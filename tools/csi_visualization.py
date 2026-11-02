"""Shared CSI parsing and filtering helpers for CLI and web visualization."""

import re

import numpy as np

from csi_preprocessing import (
    FILTER_CUTOFF_HZ,
    FILTER_ORDER,
    RAW_SUBCARRIERS,
    SAMPLE_RATE_HZ,
    butterworth_filter,
    hampel_filter,
    min_max_scale,
)

RAW_LEN = RAW_SUBCARRIERS * 2
FILTER_MODES = ("none", "hampel", "butterworth", "hampel_butterworth", "pipeline")


def parse_csi_line(line: str) -> np.ndarray | None:
    """Parse one CSI_DATA line into 64 finite subcarrier amplitudes."""
    if "CSI_DATA" not in line or "[" not in line or "]" not in line:
        return None
    match = re.search(r"\[(.*?)\]", line)
    if match is None:
        return None
    try:
        values = [int(value.strip()) for value in match.group(1).split(",")]
    except ValueError:
        return None
    if len(values) != RAW_LEN:
        return None

    values_array = np.asarray(values, dtype=np.float64)
    amplitude = np.hypot(values_array[1::2], values_array[0::2])
    return amplitude if np.isfinite(amplitude).all() else None


def pad_for_filter(data: np.ndarray) -> np.ndarray:
    """Provide enough samples for filtfilt while preserving the live window."""
    minimum = 20
    if len(data) >= minimum:
        return data
    return np.pad(data, ((0, minimum - len(data)), (0, 0)), mode="edge")


def apply_filter(
    data: np.ndarray,
    mode: str,
    cutoff: float = FILTER_CUTOFF_HZ,
    order: int = FILTER_ORDER,
    hampel_window: int = 7,
    hampel_sigma: float = 3.0,
) -> np.ndarray:
    """Apply the selected analysis filter to a frames-by-64 array."""
    if data.size == 0 or mode == "none":
        return data
    if mode not in FILTER_MODES:
        raise ValueError(f"Unsupported filter mode: {mode}")

    working = pad_for_filter(data)
    if mode == "hampel":
        result = hampel_filter(working, hampel_window, hampel_sigma)
    elif mode == "butterworth":
        result = butterworth_filter(working, cutoff, order)
    elif mode == "hampel_butterworth":
        result = butterworth_filter(
            hampel_filter(working, hampel_window, hampel_sigma),
            cutoff,
            order,
        )
    else:
        result = min_max_scale(
            butterworth_filter(
                hampel_filter(working, hampel_window, hampel_sigma),
                cutoff,
                order,
            )
        )
    return result[: len(data)]


def summarize_series(
    data: np.ndarray,
    mode: str = "pipeline",
    duration: float = 15,
    subcarrier: int = 0,
    **filter_options,
) -> dict:
    """Return bounded JSON-friendly chart data for one selected subcarrier."""
    if data.ndim != 2 or data.shape[1] != RAW_SUBCARRIERS:
        raise ValueError(f"Expected frames-by-{RAW_SUBCARRIERS} CSI data, got {data.shape}.")
    if not 0 <= subcarrier < RAW_SUBCARRIERS:
        raise ValueError(f"subcarrier must be between 0 and {RAW_SUBCARRIERS - 1}.")
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    filtered = apply_filter(data, mode, **filter_options)
    count = max(1, int(duration * SAMPLE_RATE_HZ))
    raw_series = data[-count:, subcarrier]
    filtered_series = filtered[-count:, subcarrier]
    return {
        "frames": int(len(data)),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "subcarrier": subcarrier,
        "raw": raw_series.tolist(),
        "filtered": filtered_series.tolist(),
    }
