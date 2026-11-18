"""Single-ESP Wi-Fi CSI preprocessing, baseline, features and heatmap.

Designed for CSI CSV files with a `data` list column and the receiver
`timestamp` in microseconds.
Supported *explicit* acquisition layouts:
  - ht20_256: 128 IQ bytes LLTF + 128 IQ bytes HT-LTF (107 active carriers)
  - lltf_128: one 128-IQ-byte LLTF field (51 active carriers)
  - htltf_128: one 128-IQ-byte HT-LTF field (56 active carriers)

The 128-byte layout cannot be identified reliably from length alone: select
LLTF/HT-LTF based on the firmware's CSI configuration and packet metadata.

Dependencies: numpy, pandas, scipy; matplotlib only for plot_heatmaps().
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

LLTF_ACTIVE_IDX = np.r_[np.arange(2, 27), np.arange(38, 64)]
HTLTF_ACTIVE_IDX = np.r_[np.arange(1, 29), np.arange(36, 64)]
LLTF_N, HTLTF_N = len(LLTF_ACTIVE_IDX), len(HTLTF_ACTIVE_IDX)
LAYOUT_BYTES = {"ht20_256": 256, "lltf_128": 128, "htltf_128": 128}
MAX_INTERP_GAP_S = 0.25
EPS = 1e-6
FEATURE_NAMES = [
    "mean_profile_mean", "mean_profile_std", "mean_profile_median",
    "temporal_std_mean", "temporal_std_median", "temporal_std_max",
    "temporal_absdiff_mean", "temporal_diff_std",
    "baseline_abs_mean", "baseline_abs_median", "baseline_abs_max",
    "baseline_rmse", "baseline_corr",
]
LABELS = {"empty": 0, "standing": 1, "walking": 2}
MOTION_FEATURE_NAMES = ['motion_' + n for n in (
    'temporal_std_mean', 'temporal_std_median', 'temporal_std_p90', 'temporal_std_max',
    'absdiff_mean', 'absdiff_median', 'absdiff_p90', 'absdiff_max',
    'rmsdiff_mean', 'rmsdiff_p90', 'range_mean', 'range_p90',
    'packet_change_mean', 'packet_change_std', 'packet_change_p90', 'packet_change_max',
    'bandpower_mean', 'bandpower_p90')]
ALL_FEATURE_NAMES = FEATURE_NAMES + MOTION_FEATURE_NAMES


@dataclass
class CSIData:
    time_s: np.ndarray
    amplitude: np.ndarray
    metadata: pd.DataFrame
    layout: str


def parse_csi_cell(cell) -> np.ndarray:
    """Decode a CSV string '[imag, real, ...]' without executing Python code."""
    value = cell if isinstance(cell, (list, tuple, np.ndarray)) else ast.literal_eval(str(cell))
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("CSI data must be a nonempty one-dimensional sequence")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("CSI data must contain numbers")
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("CSI data contains non-finite values")
    if ((arr < -128) | (arr > 127)).any() or (arr != np.trunc(arr)).any():
        raise ValueError("CSI IQ byte values must be signed integers in [-128, 127]")
    return arr


def _iq_amplitude(buf: np.ndarray) -> np.ndarray:
    if buf.ndim != 1 or len(buf) != 128:
        raise ValueError("Expected 128 bytes (64 complex values) for each LTF")
    imag, real = buf[0::2], buf[1::2]
    return np.hypot(real, imag)


def raw_iq_to_amplitude(raw: np.ndarray, layout: str = "ht20_256") -> np.ndarray:
    """Convert fixed-format ESP CSI IQ bytes to *selected* active carriers."""
    if layout not in LAYOUT_BYTES:
        raise ValueError(f"Unknown layout: {layout!r}; choose {list(LAYOUT_BYTES)}")
    raw = parse_csi_cell(raw)
    if raw.size != LAYOUT_BYTES[layout]:
        raise ValueError(f"{layout} expects {LAYOUT_BYTES[layout]} bytes, got {raw.size}")
    if layout == "ht20_256":
        return np.concatenate((
            _iq_amplitude(raw[:128])[LLTF_ACTIVE_IDX],
            _iq_amplitude(raw[128:])[HTLTF_ACTIVE_IDX],
        )).astype(np.float32)
    if layout == "lltf_128":
        return _iq_amplitude(raw)[LLTF_ACTIVE_IDX].astype(np.float32)
    return _iq_amplitude(raw)[HTLTF_ACTIVE_IDX].astype(np.float32)


def carrier_groups(layout: str) -> list[slice]:
    """Groups that must be median-normalized separately."""
    if layout == "ht20_256":
        return [slice(0, LLTF_N), slice(LLTF_N, LLTF_N + HTLTF_N)]
    if layout in ("lltf_128", "htltf_128"):
        return [slice(None)]
    raise ValueError(f"Unknown layout {layout}")


def load_receiver_csv(
    path: str | Path,
    *,
    layout: str = "ht20_256",
    time_column: str = "timestamp",
    time_scale: float = 1e-6,
    max_gap_s: float = MAX_INTERP_GAP_S,
) -> CSIData:
    """Read, validate and order one receiver's CSI series.

    The default `timestamp` column is in microseconds. For another timestamp
    in microseconds, pass its name and `time_scale=1e-6`. The time column must
    represent a *monotonic* clock over the session; capture and firmware clocks
    should not be mixed silently.
    """
    path = Path(path)
    if layout not in LAYOUT_BYTES:
        raise ValueError(f"Unknown layout {layout}")
    if time_scale <= 0 or max_gap_s <= 0:
        raise ValueError("time_scale and max_gap_s must be positive")
    df = pd.read_csv(path)
    required = {"data", "len"}
    if not required.issubset(df):
        raise ValueError(f"{path}: missing columns: {sorted(required - set(df))}")
    if len(df) < 10:
        raise ValueError(f"{path}: fewer than 10 rows")

    df = df.loc[pd.to_numeric(df["len"], errors="coerce") == LAYOUT_BYTES[layout]].copy()
    # Check optional firmware PHY metadata where the corresponding values are known.
    expected = {
        "sig_mode": 1, "sig": 1,
        "secondary_channel": 0, "second_channel": 0,
        "cwb": 0, "stbc": 0,
    }
    for col, val in expected.items():
        if col in df:
            df = df.loc[pd.to_numeric(df[col], errors="coerce") == val].copy()
    if "channel" in df and not df.empty:
        channels = pd.to_numeric(df["channel"], errors="coerce")
        if channels.dropna().empty:
            raise ValueError(f"{path}: channel column has no valid values")
        selected_channel = channels.mode().iloc[0]
        df = df.loc[channels == selected_channel].copy()

    values, good_rows = [], []
    for row_id, cell in zip(df.index, df["data"]):
        try:
            values.append(raw_iq_to_amplitude(cell, layout))
            good_rows.append(row_id)
        except (ValueError, TypeError, SyntaxError, OverflowError):
            continue
    if len(values) < 10:
        raise ValueError(f"{path}: fewer than 10 valid CSI packets after filtering")
    df = df.loc[good_rows].copy().reset_index(drop=True)
    if layout == "htltf_128" and "first_word_invalid" in df and (df.first_word_invalid != 0).any():
        raise ValueError("htltf_128 includes invalid carrier 1; use LLTF or recollect")
    t = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float) * time_scale
    A = np.vstack(values).astype(np.float32)
    good_t = np.isfinite(t)
    t, A, df = t[good_t], A[good_t], df.loc[good_t].reset_index(drop=True)
    # Never sort a clock reset into an apparently valid session.
    if time_scale == 1e-6:
        from csi.protocol import unwrap_timestamps
        t = unwrap_timestamps(t / time_scale) * time_scale
    elif (np.diff(t) <= 0).any():
        raise ValueError("Non-monotonic timestamps; split recording")
    if len(t) < 10:
        raise ValueError(f"{path}: fewer than 10 unique valid timestamps")
    gaps = np.diff(t)
    if (gaps > max_gap_s).any():
        raise ValueError(
            f"{path}: longest timestamp gap {np.max(gaps):.3f}s > {max_gap_s:.3f}s; "
            "split the recording or diagnose dropped packets before interpolation"
        )
    t = t - t[0]
    return CSIData(t, A, df, layout)


def fieldwise_ltf_normalize(A: np.ndarray, layout: str = "ht20_256") -> np.ndarray:
    """Per-packet median normalization separately for each available LTF."""
    A = np.asarray(A, dtype=np.float32)
    n = {"ht20_256": LLTF_N + HTLTF_N, "lltf_128": LLTF_N,
         "htltf_128": HTLTF_N}.get(layout)
    if A.ndim != 2 or n is None or A.shape[1] != n:
        raise ValueError(f"Invalid amplitude shape {A.shape} for layout {layout}")
    chunks = []
    for group in carrier_groups(layout):
        part = A[:, group]
        med = np.median(part, axis=1, keepdims=True)
        # Extremely small denominators are a data-quality issue, not a normal scale.
        if (med < EPS).any():
            raise ValueError("Median CSI amplitude too small for stable normalization")
        chunks.append(part / med)
    return np.concatenate(chunks, axis=1).astype(np.float32)


def temporal_smooth(A: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing independently over time for each carrier."""
    if window < 3 or len(A) < window:
        return np.asarray(A, dtype=np.float32)
    if window % 2 == 0:
        window += 1
    if len(A) < window:
        return np.asarray(A, dtype=np.float32)
    return savgol_filter(A, window_length=window, polyorder=2,
                         axis=0, mode="interp").astype(np.float32)


def resample(t: np.ndarray, A: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(grid) == 0 or grid[0] < t[0] - 1e-9 or grid[-1] > t[-1] + 1e-9:
        raise ValueError("Resampling grid outside recording timestamps")
    # Permit only sub-nanosecond roundoff already accepted by the bounds check.
    # scipy's strict bounds otherwise rejects 5.9 vs 5.8999999999999995.
    grid = np.clip(grid, t[0], t[-1])
    return interp1d(t, A, axis=0, kind="linear", bounds_error=True)(grid).astype(np.float32)


def preprocess_session(
    path: str | Path,
    *,
    layout: str = "ht20_256",
    fs: float = 25.0,
    time_column: str = "timestamp",
    time_scale: float = 1e-6,
    smoothing_window: int = 0,
    max_gap_s: float = MAX_INTERP_GAP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """CSV -> time grid, normalized and time-smoothed amplitude [T, K].

    This is offline resampling. When input sample rates exceed target fs,
    interpolation alone does NOT provide anti-alias filtering; validate the
    signal bandwidth and add a suitable anti-aliasing stage if required.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    data = load_receiver_csv(path, layout=layout, time_column=time_column,
                             time_scale=time_scale, max_gap_s=max_gap_s)
    source_fs = (len(data.time_s)-1) / data.time_s[-1]
    if source_fs > fs * 1.3:
        raise ValueError(f'Source rate {source_fs:.1f} Hz exceeds target {fs:g} Hz; '
                         'check AP rate/extra CSI traffic or configure matching fs; '
                         'downsampling without anti-alias filtering is disabled')
    dt = 1.0 / fs
    grid = np.arange(0, data.time_s[-1], dt, dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("Recording too short for target resampling rate")
    A = resample(data.time_s, data.amplitude, grid)
    A = fieldwise_ltf_normalize(A, layout)
    A = temporal_smooth(A, window=smoothing_window)
    return grid, A


def window_slices(n: int, fs: float, window_s: float, step_s: float) -> Iterable[slice]:
    if not np.isfinite([fs, window_s, step_s]).all() or fs <= 0 or window_s <= 0 or step_s <= 0:
        raise ValueError("fs, window_s, step_s must be positive")
    win, step = int(round(window_s * fs)), int(round(step_s * fs))
    if win < 2 or step < 1:
        raise ValueError("Window/step too short")
    for i in range(0, n - win + 1, step):
        yield slice(i, i + win)


def fit_empty_baseline(
    empty_csvs: Iterable[str | Path], *, layout: str = "ht20_256",
    fs: float = 25.0, time_column: str = "timestamp",
    time_scale: float = 1e-6,
) -> np.ndarray:
    """Median over time per empty session, then median across TRAIN empty sessions.

    Caller must supply ONLY TRAIN EMPTY sessions to prevent data leakage.
    """
    profiles = []
    for path in empty_csvs:
        _, A = preprocess_session(path, layout=layout, fs=fs,
                                  time_column=time_column, time_scale=time_scale)
        profiles.append(np.median(A, axis=0))
    if not profiles:
        raise ValueError("At least one TRAIN EMPTY session is required")
    return np.median(np.stack(profiles), axis=0).astype(np.float32)


def save_baseline(path: str | Path, baseline: np.ndarray, *, layout: str = "ht20_256") -> None:
    np.savez_compressed(path, amplitude=np.asarray(baseline, dtype=np.float32),
                        layout=np.asarray(layout))


def load_baseline(path: str | Path, *, expected_layout: str | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        layout = str(z["layout"].item())
        if expected_layout is not None and layout != expected_layout:
            raise ValueError(f"Baseline layout {layout} != requested {expected_layout}")
        return z["amplitude"].astype(np.float32)


def extract_window_features(W: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """13 scalar features for one CSI time window, one ESP receiver."""
    W = np.asarray(W, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if W.ndim != 2 or len(W) < 2 or baseline.shape != (W.shape[1],):
        raise ValueError(f"Window {W.shape} / baseline {baseline.shape} mismatch")
    mean_profile = W.mean(axis=0)
    temporal_std = W.std(axis=0)
    diff = np.diff(W, axis=0)
    delta = np.abs(mean_profile - baseline)
    corr = (float(np.corrcoef(mean_profile, baseline)[0, 1])
            if np.std(mean_profile) > EPS and np.std(baseline) > EPS else 0.0)
    result = np.array([
        mean_profile.mean(), mean_profile.std(), np.median(mean_profile),
        temporal_std.mean(), np.median(temporal_std), temporal_std.max(),
        np.abs(diff).mean(), diff.std(), delta.mean(), np.median(delta),
        delta.max(), np.sqrt(np.mean((mean_profile - baseline) ** 2)), corr,
    ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite extracted features")
    return result


def extract_motion_features(W: np.ndarray, fs: float) -> np.ndarray:
    """18 single-RX motion descriptors from reference phase2/phase2_utils.py.

    Bandpower follows the reference log1p(mean FFT-bin power) convention;
    it is a descriptor, not a physically calibrated power spectral density.
    """
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2 or len(W) < 4 or not np.isfinite(W).all() or fs <= 0:
        raise ValueError('Invalid motion window or sample rate')
    sd = W.std(axis=0)
    d = np.diff(W, axis=0)
    ad = np.abs(d).mean(axis=0)
    rms = np.sqrt(np.mean(d*d, axis=0))
    ranges = np.ptp(W, axis=0)
    change = np.sqrt(np.mean(d*d, axis=1))
    spec = np.fft.rfft(W-W.mean(axis=0, keepdims=True), axis=0)
    freq = np.fft.rfftfreq(len(W), d=1/fs)
    mask = (freq >= .5) & (freq <= 5)
    power = np.abs(spec)**2/len(W)
    band = np.log1p(power[mask].mean(axis=0)) if mask.any() else np.zeros(W.shape[1])
    q = lambda a: np.percentile(a, 90)
    result = np.array([
        sd.mean(), np.median(sd), q(sd), sd.max(),
        ad.mean(), np.median(ad), q(ad), ad.max(),
        rms.mean(), q(rms), ranges.mean(), q(ranges),
        change.mean(), change.std(), q(change), change.max(), band.mean(), q(band)
    ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite motion features')
    return result


def extract_two_phase_features(W, baseline, fs):
    """W is already normalized and smoothed; both offline/live call this."""
    return np.r_[extract_window_features(W, baseline), extract_motion_features(W, fs)]


def session_features(
    csv_path: str | Path, baseline: np.ndarray, *,
    layout: str = "ht20_256", fs: float = 25.0,
    window_s: float = 2.0, step_s: float = 0.5,
    time_column: str = "timestamp", time_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [windows, 31]: 13 presence + 18 motion, in a fixed order."""
    t, A = preprocess_session(csv_path, layout=layout, fs=fs,
                              time_column=time_column, time_scale=time_scale)
    slices = list(window_slices(len(t), fs, window_s, step_s))
    if not slices:
        raise ValueError(f"{csv_path}: no complete windows")
    X = np.vstack([extract_two_phase_features(temporal_smooth(A[s]), baseline, fs) for s in slices])
    centers = np.asarray([t[s].mean() for s in slices], dtype=np.float64)
    return X, centers


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Require single-receiver manifest: session_id,label,split,rx_csv."""
    df = pd.read_csv(path)
    required = {"session_id", "label", "split", "rx_csv"}
    if not required.issubset(df):
        raise ValueError(f"Manifest missing columns {sorted(required - set(df))}")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    if (~df["label"].isin(LABELS)).any():
        raise ValueError(f"Expected labels {list(LABELS)}")
    if (~df["split"].isin(("train", "val", "test"))).any():
        raise ValueError("Split must be train, val or test")
    if df["session_id"].duplicated().any():
        raise ValueError("Duplicate session_id in manifest")
    if df["rx_csv"].duplicated().any():
        raise ValueError("A recording must not appear in multiple sessions/splits")
    return df


def prepare_feature_dataset(
    manifest_path: str | Path, out_dir: str | Path, *,
    layout: str = "ht20_256", fs: float = 25.0,
    window_s: float = 2.0, step_s: float = 0.5,
    time_column: str = "timestamp", time_scale: float = 1e-6,
) -> pd.DataFrame:
    """Prepare single-RX, 3-label dataset with a TRAIN-only empty baseline."""
    manifest_path, out_dir = Path(manifest_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_manifest(manifest_path)
    def resolve(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (manifest_path.parent / p).resolve()
    df["resolved"] = df["rx_csv"].map(resolve)
    if df["resolved"].duplicated().any():
        raise ValueError("Same resolved recording appears in multiple splits")
    import hashlib
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in df["resolved"]]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate recording contents: possible data leakage")
    missing = [str(p) for p in df["resolved"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing recordings: {missing}")
    empty_train = df.loc[(df["split"] == "train") & (df["label"] == "empty")]
    baseline = fit_empty_baseline(empty_train["resolved"], layout=layout, fs=fs,
                                  time_column=time_column, time_scale=time_scale)
    save_baseline(out_dir / "empty_baseline.npz", baseline, layout=layout)
    blocks, summaries = [], []
    for r in df.itertuples(index=False):
        X, centers = session_features(r.resolved, baseline, layout=layout, fs=fs,
                                      window_s=window_s, step_s=step_s,
                                      time_column=time_column, time_scale=time_scale)
        block = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        block.insert(0, "window_center_s", centers)
        block.insert(0, "y", LABELS[r.label])
        block.insert(0, "phase1_label", 'empty' if r.label == 'empty' else 'present')
        block.insert(0, "phase1_y", int(r.label != 'empty'))
        block.insert(0, "split", r.split)
        block.insert(0, "label", r.label)
        block.insert(0, "session_id", r.session_id)
        blocks.append(block)
        summaries.append({"session_id": r.session_id, "label": r.label,
                          "split": r.split, "windows": len(block)})
    result = pd.concat(blocks, ignore_index=True)
    result.to_csv(out_dir / "all_features.csv", index=False)
    for split in ("train", "val", "test"):
        result.loc[result["split"] == split].to_csv(out_dir / f"{split}_features.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "session_summary.csv", index=False)
    metadata = {"layout": layout, "fs_hz": fs, "window_s": window_s,
                "step_s": step_s, "label_mapping": LABELS,
                "feature_names": ALL_FEATURE_NAMES,
                "phase1_feature_names": FEATURE_NAMES,
                "phase2_feature_names": MOTION_FEATURE_NAMES,
                "phase1_labels": {"empty": 0, "present": 1},
                "phase2_labels": {"standing": 1, "walking": 2},
                "baseline_source": "TRAIN EMPTY sessions only",
                "time_column": time_column, "time_scale": time_scale,
                "smoothing": "window-local savgol 5 polyorder 2", "pipeline_version": 3}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def plot_heatmaps(
    csv_path: str | Path, output_png: str | Path, *,
    layout: str = "ht20_256", baseline: np.ndarray | None = None,
    fs: float = 25.0, time_column: str = "timestamp", time_scale: float = 1e-6,
) -> None:
    """Save a two-panel amplitude and deviation Time–Subcarrier heatmap."""
    import matplotlib.pyplot as plt
    t, A = preprocess_session(csv_path, layout=layout, fs=fs,
                              time_column=time_column, time_scale=time_scale)
    if baseline is None:
        # Exploratory view only; this is NOT an EMPTY-room reference.
        reference = np.median(A, axis=0)
        ylabel2 = "Deviation from session median (not EMPTY baseline)"
    else:
        baseline = np.asarray(baseline, dtype=np.float32)
        if baseline.shape != (A.shape[1],):
            raise ValueError("Baseline does not match active subcarrier dimension")
        reference = baseline
        ylabel2 = "Deviation from TRAIN EMPTY baseline"
    D = (A - reference).T
    bound = max(float(np.quantile(np.abs(D), 0.99)), 1e-3)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    extent = (float(t[0]), float(t[-1]), A.shape[1] - .5, -.5)
    im0 = ax[0].imshow(A.T, aspect="auto", origin="upper", extent=extent,
                       cmap="viridis", interpolation="nearest")
    im1 = ax[1].imshow(D, aspect="auto", origin="upper", extent=extent,
                       cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
    for axis in ax:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Active subcarrier (index)")
        if layout == "ht20_256":
            axis.axhline(LLTF_N - .5, ls="--", lw=1, color="black", alpha=.7)
    ax[0].set_title("Normalized CSI amplitude")
    ax[1].set_title(ylabel2)
    fig.colorbar(im0, ax=ax[0], label="Relative amplitude")
    fig.colorbar(im1, ax=ax[1], label="Amplitude deviation")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Single-ESP CSI utilities")
    sp = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "plot"):
        sub = sp.add_parser(name)
        sub.add_argument("--layout", choices=list(LAYOUT_BYTES), default="ht20_256")
        sub.add_argument("--fs", type=float, default=25.0)
        sub.add_argument("--time-column", default="timestamp")
        sub.add_argument("--time-scale", type=float, default=1e-6)
        if name == "prepare":
            sub.add_argument("--manifest", required=True)
            sub.add_argument("--out", required=True)
            sub.add_argument("--window", type=float, default=2.0)
            sub.add_argument("--step", type=float, default=0.5)
        else:
            sub.add_argument("--csv", required=True)
            sub.add_argument("--out", required=True)
            sub.add_argument("--baseline", default=None)
    args = p.parse_args()
    if args.command == "prepare":
        result = prepare_feature_dataset(args.manifest, args.out,
                                         layout=args.layout, fs=args.fs,
                                         window_s=args.window, step_s=args.step,
                                         time_column=args.time_column,
                                         time_scale=args.time_scale)
        print(f"Prepared {len(result)} windows: 13 presence + 18 motion features/window")
        print(result.groupby(["split", "label"]).size().to_string())
    else:
        baseline = (load_baseline(args.baseline, expected_layout=args.layout)
                    if args.baseline else None)
        plot_heatmaps(args.csv, args.out, layout=args.layout, baseline=baseline,
                      fs=args.fs, time_column=args.time_column,
                      time_scale=args.time_scale)
        print(f"Saved heatmap to {args.out}")


if __name__ == "__main__":
    main()
