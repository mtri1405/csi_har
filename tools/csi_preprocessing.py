import numpy as np
from scipy.signal import butter, filtfilt
from numpy.lib.stride_tricks import sliding_window_view


SAMPLE_RATE_HZ = 100.0
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 4
PCA_COMPONENTS = 10
WINDOW_SIZE = 60

RAW_SUBCARRIERS = 64
TARGET_SUBCARRIERS = 64
REMOVED_SUBCARRIER_INDICES = (
    *range(0, 6),
    11,
    25,
    32,
    39,
    53,
    *range(58, 64),
)


def hampel_filter(data, window_size=7, n_sigma=3.0):
    """Vectorized Hampel filter removing spikes/outliers per subcarrier."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}.")
    n_frames, n_sub = data.shape
    if window_size % 2 == 0:
        window_size += 1
    k = window_size // 2
    padded = np.pad(data, ((k, k), (0, 0)), mode="median")
    windows = sliding_window_view(padded, (window_size, n_sub))[:, 0, :, :]
    medians = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - medians[:, None, :]), axis=1)
    sigma = 1.4826 * mad
    threshold = np.where(sigma == 0, np.inf, n_sigma * sigma)
    mask = np.abs(data - medians) > threshold
    out = data.copy()
    out[mask] = medians[mask]
    return out


def butterworth_filter(data, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER, fs=SAMPLE_RATE_HZ):
    """Zero-phase Butterworth low-pass filter along the time axis."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}.")
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if not 0.0 < normal_cutoff < 1.0:
        raise ValueError(
            f"cutoff must be between 0 and Nyquist ({nyquist:g} Hz), got {cutoff:g} Hz."
        )
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def preprocess_csi(data):
    """Validate 64-channel CSI and remove temporal spikes."""
    if data.ndim != 2 or data.shape[1] != TARGET_SUBCARRIERS:
        raise ValueError(
            f"Expected CSI data with shape (Total_Frames, {TARGET_SUBCARRIERS}), "
            f"got {data.shape}."
        )
    data = np.asarray(data, dtype=np.float64)
    data = hampel_filter(data)
    return data


def remove_unusable_subcarriers(data, remove_zero_columns=True):
    if data.ndim != 2 or data.shape[1] != TARGET_SUBCARRIERS:
        raise ValueError(
            f"Expected CSI amplitude data with shape "
            f"(frames, {TARGET_SUBCARRIERS}), got {data.shape}."
        )
    remove_indices = set(REMOVED_SUBCARRIER_INDICES)
    if remove_zero_columns:
        zero_columns = np.all(data == 0, axis=0)
        remove_indices.update(np.flatnonzero(zero_columns).tolist())
    keep_indices = [i for i in range(data.shape[1]) if i not in remove_indices]
    return data[:, keep_indices], sorted(remove_indices), keep_indices


def pca_reduce(window, n_components=PCA_COMPONENTS):
    if window.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {window.shape}.")
    if n_components < 1 or n_components > window.shape[1]:
        raise ValueError(
            f"n_components must be between 1 and {window.shape[1]}, got {n_components}."
        )
    centered = window - np.mean(window, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    vectors = vectors[:n_components]
    # SVD signs are arbitrary; fix them so consecutive windows use a stable orientation.
    largest_loadings = np.argmax(np.abs(vectors), axis=1)
    signs = np.sign(vectors[np.arange(n_components), largest_loadings])
    signs[signs == 0] = 1.0
    return centered @ (vectors * signs[:, None]).T


def min_max_scale(data):
    minimum = np.min(data)
    maximum = np.max(data)
    spread = maximum - minimum
    if spread <= 1e-8:
        return np.zeros_like(data)
    return (data - minimum) / spread


def normalize_window(window):
    return min_max_scale(window)


def preprocess_window(window):
    if window.ndim != 2 or window.shape[0] != WINDOW_SIZE:
        raise ValueError(f"Expected a window with shape ({WINDOW_SIZE}, subcarriers), got {window.shape}.")
    filtered = butterworth_filter(hampel_filter(window))
    reduced = pca_reduce(filtered)
    return normalize_window(reduced)
