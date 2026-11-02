# dataset_pipeline.py
import os
import re
import glob
import json
import numpy as np
from csi_preprocessing import (
    FILTER_CUTOFF_HZ, FILTER_ORDER, PCA_COMPONENTS, RAW_SUBCARRIERS,
    SAMPLE_RATE_HZ, TARGET_SUBCARRIERS, WINDOW_SIZE,
    filter_window, fit_pca, transform_pca, normalize_window,
    REMOVED_SUBCARRIER_INDICES, SUBCARRIER_INDICES,
)

# Tên thư mục -> nhãn. Đây là nguồn duy nhất định nghĩa nhãn,
# train.py sẽ đọc lại từ file .npz thay vì tự hard-code.
ACTIONS = {
    "empty": 0,
    "walking": 1,
    "standing": 2
}
LABEL_NAMES = {
    0: "Phòng trống (Empty)",
    1: "Đang đi lại (Walking)",
    2: "Đang đứng yên (Standing)"
}
RAW_LEN = RAW_SUBCARRIERS * 2


def stratified_file_split(groups, y, seed=42, val_frac=0.2, test_frac=0.2):
    """Split complete recording files while preserving label coverage."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    if groups.ndim != 1 or y.ndim != 1 or len(groups) != len(y):
        raise ValueError("groups and y must be one-dimensional arrays of equal length.")

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    unique_groups = np.unique(groups)
    group_labels = np.array([y[np.flatnonzero(groups == group)[0]] for group in unique_groups])
    for label in np.unique(group_labels):
        label_groups = unique_groups[group_labels == label]
        shuffled_groups = label_groups[rng.permutation(len(label_groups))]
        n_test = max(1, int(round(len(shuffled_groups) * test_frac)))
        n_val = max(1, int(round(len(shuffled_groups) * val_frac)))
        n_train = len(shuffled_groups) - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Nhãn {label} không đủ file để chia train/valid/test.")
        train_groups = shuffled_groups[:n_train]
        val_groups = shuffled_groups[n_train:n_train + n_val]
        test_groups = shuffled_groups[n_train + n_val:]
        train_idx.extend(np.flatnonzero(np.isin(groups, train_groups)))
        val_idx.extend(np.flatnonzero(np.isin(groups, val_groups)))
        test_idx.extend(np.flatnonzero(np.isin(groups, test_groups)))
    return tuple(
        rng.permutation(np.asarray(index, dtype=np.int64))
        for index in (train_idx, val_idx, test_idx)
    )


# --- 1. ĐỌC DỮ LIỆU TỪ FILE ---
def parse_csi_file(filepath):
    amplitudes = []
    skipped_frames = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "CSI_DATA" not in line or "[" not in line or "]" not in line:
                continue

            match = re.search(r"\[(.*?)\]", line)
            if not match:
                continue

            try:
                raw_vals = [int(x.strip()) for x in match.group(1).split(",")]
            except ValueError:
                # Chuỗi bị dính số (vd '-6-19') -> bỏ frame hỏng
                skipped_frames += 1
                continue

            if len(raw_vals) != RAW_LEN:
                skipped_frames += 1
                continue

            raw_arr = np.array(raw_vals, dtype=np.float32)
            imag = raw_arr[0::2]
            real = raw_arr[1::2]
            amp = np.sqrt(real ** 2 + imag ** 2)

            if np.isnan(amp).any() or np.isinf(amp).any():
                skipped_frames += 1
                continue

            amplitudes.append(amp)

    if skipped_frames > 0:
        print(f"    [Thông báo] Bỏ qua {skipped_frames} frame lỗi trong '{os.path.basename(filepath)}'.")

    if not amplitudes:
        return np.empty((0, RAW_LEN // 2), dtype=np.float32)
    return np.array(amplitudes, dtype=np.float32)  # (frames, 64 HT20 subcarriers)


# --- 2. QUÉT DỮ LIỆU VÀ CẮT CỬA SỔ ---
def build_dataset(base_dir="Data", window_size=WINDOW_SIZE, stride=10):
    """
    Cấu trúc thư mục:
      Data/empty/*.csv     -> nhãn 0
      Data/walking/*.csv   -> nhãn 1
      Data/standing/*.csv  -> nhãn 2
    """
    X, y, sample_groups = [], [], []
    expected_keep_indices = None
    files_per_label = {}
    rejected_files = {}

    for action_name, label in ACTIONS.items():
        folder_path = os.path.join(base_dir, action_name)

        files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        print(f"[*] '{action_name}' (nhãn {label}): tìm thấy {len(files)} file CSV.")
        used_files = 0
        rejected_files[action_name] = []

        for file in files:
            raw_amp = parse_csi_file(file)
            if raw_amp.size == 0:
                print(f"    [!] '{os.path.basename(file)}' không có frame CSI hợp lệ; bỏ qua.")
                continue


            if len(raw_amp) < window_size:
                print(
                    f"    [!] '{os.path.basename(file)}' có {len(raw_amp)} frame hợp lệ; "
                    f"đệm thêm {window_size - len(raw_amp)} frame cuối để đủ {window_size}"
                )

            raw_amp = np.asarray(raw_amp, dtype=np.float64)
            cleaned_amp = raw_amp
            keep_indices = list(SUBCARRIER_INDICES)
            if expected_keep_indices is None:
                expected_keep_indices = keep_indices
            elif keep_indices != expected_keep_indices:
                raise ValueError(
                    f"File '{file}' có layout subcarrier khác các file trước. "
                    "Các file cần có cùng mask subcarrier."
                )
            print(
                f"    [CSI] '{os.path.basename(file)}': "
                f"giữ {len(keep_indices)}/{RAW_SUBCARRIERS} subcarrier"
            )

            num_frames = cleaned_amp.shape[0]
            if num_frames < window_size:
                cleaned_amp = np.pad(
                    cleaned_amp,
                    ((0, window_size - num_frames), (0, 0)),
                    mode="edge",
                )
                num_frames = window_size
            n_windows = 0
            for start in range(0, num_frames - window_size + 1, stride):
                win = cleaned_amp[start:start + window_size, :]
                if win.shape[0] != WINDOW_SIZE:
                    raise ValueError(f"Unexpected window shape: {win.shape}")
                X.append(filter_window(win))
                y.append(label)
                sample_groups.append(os.path.abspath(file))
                n_windows += 1

            if n_windows > 0:
                used_files += 1

        files_per_label[label] = used_files





    if not X or expected_keep_indices is None:
        raise ValueError(f"Không tìm thấy file CSI hợp lệ trong '{base_dir}'.")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    groups = np.array(sample_groups)
    train_idx, val_idx, test_idx = stratified_file_split(groups, y)
    pca_mean, pca_components = fit_pca(X[train_idx])
    X = np.array(
        [normalize_window(transform_pca(window, pca_mean, pca_components)) for window in X],
        dtype=np.float32,
    )[..., np.newaxis]
    return (
        X,
        y,
        groups,
        np.array(expected_keep_indices, dtype=np.int64),
        pca_mean,
        pca_components,
        train_idx,
        val_idx,
        test_idx,
    )


# --- 3. CHẠY VÀ LƯU DATASET ---
if __name__ == "__main__":
    DATA_PATH = "Data"

    print("--- ĐANG XỬ LÝ VÀ ĐÓNG GÓI DATASET ---")
    (
        X, y, groups, keep_indices, pca_mean, pca_components,
        train_idx, val_idx, test_idx,
    ) = build_dataset(
        base_dir=DATA_PATH, window_size=WINDOW_SIZE, stride=10
    )

    print("\n--- KẾT QUẢ XỬ LÝ ---")
    print(f"Tổng số mẫu: {X.shape[0]}")
    print(f"X.shape: {X.shape} -> (N, {WINDOW_SIZE} frames, {X.shape[2]} PCA components, 1 kênh)")
    print(f"y.shape: {y.shape}")
    print(f"Số phiên thu độc lập (file): {len(np.unique(groups))}")

    for label in sorted(LABEL_NAMES):
        n_samples = int((y == label).sum())
        n_files = len(np.unique(groups[y == label]))
        print(f" - Nhãn {label} ({LABEL_NAMES[label]}): {n_samples} mẫu / {n_files} file")
    for split_name, split_indices in (
        ("train", train_idx), ("valid", val_idx), ("test", test_idx)
    ):
        split_files = np.unique(groups[split_indices])
        print(f" - {split_name}: {len(split_files)} file, {len(split_indices)} window")

    output_file = "csi_dataset.npz"
    metadata = {
        "window_size": WINDOW_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "filter_cutoff_hz": FILTER_CUTOFF_HZ,
        "filter_order": FILTER_ORDER,
        "filter_type": "butterworth_lowpass",
        "raw_subcarriers": RAW_SUBCARRIERS,
        "target_subcarriers": TARGET_SUBCARRIERS,
        "removed_subcarrier_indices": list(REMOVED_SUBCARRIER_INDICES),
        "pca_components": PCA_COMPONENTS,
        "normalization": "min_max_0_1",
        "subcarrier_indices": keep_indices.tolist(),
    }
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        groups=groups,
        subcarrier_indices=keep_indices,
        pca_mean=pca_mean,
        pca_components=pca_components,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_strategy="stratified_file_60_20_20",
        split_seed=42,
        label_names_json=json.dumps({str(k): v for k, v in LABEL_NAMES.items()}, ensure_ascii=False),
        metadata_json=json.dumps(metadata),
    )
    print(f"\n[OK] Đã lưu dataset vào: {output_file}")