# dataset_pipeline.py
import os
import re
import glob
import json
import numpy as np
from csi_preprocessing import (
    FILTER_CUTOFF_HZ, FILTER_ORDER, PCA_COMPONENTS, RAW_SUBCARRIERS,
    SAMPLE_RATE_HZ, TARGET_SUBCARRIERS, WINDOW_SIZE, preprocess_csi,
    remove_unusable_subcarriers, preprocess_window, REMOVED_SUBCARRIER_INDICES,
)

# Tên thư mục -> nhãn. Đây là nguồn duy nhất định nghĩa nhãn,
# train.py sẽ đọc lại từ file .npz thay vì tự hard-code.
ACTIONS = {
    "empty": 0,
    "walking": 1,
    "standing": 2,
}
LABEL_NAMES = {
    0: "Phòng trống (Empty)",
    1: "Đang đi lại (Walking)",
    2: "Đang đứng yên (Standing)",
}
RAW_LEN = RAW_SUBCARRIERS * 2


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
def build_dataset(base_dir="Data", window_size=WINDOW_SIZE, stride=25):
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

            raw_amp = preprocess_csi(raw_amp)
            cleaned_amp, removed_indices, keep_indices = remove_unusable_subcarriers(raw_amp)
            keep_indices = [int(i) for i in np.asarray(keep_indices).tolist()]
            if expected_keep_indices is None:
                expected_keep_indices = keep_indices
            elif keep_indices != expected_keep_indices:
                raise ValueError(
                    f"File '{file}' có layout subcarrier khác các file trước. "
                    "Các file cần có cùng mask subcarrier."
                )
            print(
                f"    [CSI] '{os.path.basename(file)}': "
                f"giữ {cleaned_amp.shape[1]}/{TARGET_SUBCARRIERS} subcarrier"
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
                # Giống hệt realtime: lọc, PCA và Min-Max từng cửa sổ
                X.append(preprocess_window(win))
                y.append(label)
                sample_groups.append(os.path.abspath(file))
                n_windows += 1

            if n_windows > 0:
                used_files += 1

        files_per_label[label] = used_files





    if not X or expected_keep_indices is None:
        raise ValueError(f"Không tìm thấy file CSI hợp lệ trong '{base_dir}'.")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, WINDOW_SIZE, 10, 1)
    y = np.array(y, dtype=np.int64)
    return X, y, np.array(sample_groups), np.array(expected_keep_indices, dtype=np.int64)


# --- 3. CHẠY VÀ LƯU DATASET ---
if __name__ == "__main__":
    DATA_PATH = "Data"

    print("--- ĐANG XỬ LÝ VÀ ĐÓNG GÓI DATASET ---")
    X, y, groups, keep_indices = build_dataset(
        base_dir=DATA_PATH, window_size=WINDOW_SIZE, stride=5
    )

    print("\n--- KẾT QUẢ XỬ LÝ ---")
    print(f"Tổng số mẫu: {X.shape[0]}")
    print(f"X.shape: {X.shape} -> (N, {WINDOW_SIZE} frames, {X.shape[2]} subcarrier, 1 kênh)")
    print(f"y.shape: {y.shape}")
    print(f"Số phiên thu độc lập (file): {len(np.unique(groups))}")

    for label in sorted(LABEL_NAMES):
        n_samples = int((y == label).sum())
        n_files = len(np.unique(groups[y == label]))
        print(f" - Nhãn {label} ({LABEL_NAMES[label]}): {n_samples} mẫu / {n_files} file")

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
        label_names_json=json.dumps({str(k): v for k, v in LABEL_NAMES.items()}, ensure_ascii=False),
        metadata_json=json.dumps(metadata),
    )
    print(f"\n[OK] Đã lưu dataset vào: {output_file}")