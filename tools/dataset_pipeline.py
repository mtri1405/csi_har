import os
import re
import glob
import numpy as np
from scipy.signal import butter, filtfilt
from csi_preprocessing import remove_unusable_subcarriers

# --- 1. ĐỌC DỮ LIỆU TỪ FILE ---
def parse_csi_file(filepath):
    amplitudes = []
    skipped_frames = 0
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1. Kiểm tra cấu trúc cơ bản của dòng
            if "CSI_DATA" not in line or "[" not in line or "]" not in line:
                continue
            
            match = re.search(r'\[(.*?)\]', line)
            if not match:
                continue
            
            # 2. Ép kiểu an toàn (Safe Parsing)
            try:
                # Tách chuỗi theo dấu phẩy và chuyển đổi sang số nguyên
                raw_vals = [int(x.strip()) for x in match.group(1).split(',')]
            except ValueError:
                # Bắt lỗi khi chuỗi bị dính số (như '-6-19') -> Bỏ qua frame hỏng này
                skipped_frames += 1
                continue
            
            # 3. Kiểm tra độ dài bắt buộc (384 bytes cho HT40)
            if len(raw_vals) != 384:
                skipped_frames += 1
                continue
            
            # 4. Tách phần thực và phần ảo để tính biên độ
            raw_arr = np.array(raw_vals, dtype=np.float32)
            imag = raw_arr[0::2]
            real = raw_arr[1::2]
            amp = np.sqrt(real**2 + imag**2)
            
            # 5. Kiểm tra giá trị bất thường (NaN / Inf)
            if np.isnan(amp).any() or np.isinf(amp).any():
                skipped_frames += 1
                continue

            amplitudes.append(amp)
            
    if skipped_frames > 0:
        print(f"    [Thông báo] Bỏ qua {skipped_frames} frame bị lỗi dữ liệu trong '{os.path.basename(filepath)}'.")
        
    return np.array(amplitudes)  # Kích thước ban đầu: (Số_frames_hợp_lệ, 192)

# --- 2. LỌC NHIỄU TÍN HIỆU ---
def filter_csi(data, cutoff_hz=3.0, fs=10.0, order=4):
    """
    Lọc thông thấp dọc theo trục thời gian (axis=0) cho từng subcarrier.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Lọc thông thấp Butterworth
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

# --- 3. QUÉT DỮ LIỆU VÀ CẮT CỬA SỔ (DATASET BUILDER) ---

def build_dataset(base_dir="Data", window_size=30, stride=5):
    """
    Quét dữ liệu từ các thư mục con trong 'Data/':
      - Data/empty/*.csv      -> Nhãn 0
      - Data/sit_stand/*.csv  -> Nhãn 1
      - Data/walking/*.csv    -> Nhãn 2
    """
    actions = {
        "empty": 0,
        "walking": 1,
        "standing": 2,
        "left_hand": 3
    }

    X, y = [], []
    valid_subcarrier_count = None

    for action_name, label in actions.items():
        # Trỏ thẳng vào thư mục con: Data/<action_name>/*.csv
        folder_path = os.path.join(base_dir, action_name)
        
        if not os.path.exists(folder_path):
            print(f"[!] CẢNH BÁO: Không tìm thấy thư mục '{folder_path}'. Bỏ qua nhãn {action_name}.")
            continue

        # Tìm toàn bộ file .csv trong thư mục hành động
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        print(f"[*] Hành động '{action_name}' (Nhãn {label}): Tìm thấy {len(files)} file CSV.")

        for file in files:
            raw_amp = parse_csi_file(file)
            
            # Kiểm tra độ dài file có đủ cắt ít nhất 1 cửa sổ không
            if len(raw_amp) < window_size:
                print(f"    [!] Bỏ qua file '{os.path.basename(file)}' (chỉ có {len(raw_amp)} frame, cần tối thiểu {window_size})")
                continue

            cleaned_amp, removed_indices, _ = remove_unusable_subcarriers(raw_amp)
            if valid_subcarrier_count is None:
                valid_subcarrier_count = cleaned_amp.shape[1]
            elif cleaned_amp.shape[1] != valid_subcarrier_count:
                raise ValueError(
                    f"File '{file}' giữ lại {cleaned_amp.shape[1]} subcarrier, "
                    f"nhưng dataset trước đó dùng {valid_subcarrier_count}. "
                    "Các file cần có cùng layout subcarrier."
                )
            print(
                f"    [CSI] Giữ {cleaned_amp.shape[1]}/192 subcarrier, "
                f"loại bỏ index: {removed_indices}"
            )

            # Lọc nhiễu thông thấp Butterworth
            filtered = filter_csi(cleaned_amp)

            # Cắt cửa sổ trượt (Sliding Window)
            num_frames = filtered.shape[0]
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                win = filtered[start:end, :]

                # Chuẩn hóa Z-Score từng cửa sổ để loại bỏ nhiễu phản xạ tĩnh
                win_norm = (win - np.mean(win)) / (np.std(win) + 1e-8)

                X.append(win_norm)
                y.append(label)

    if len(X) == 0:
        raise ValueError("Không tìm thấy dữ liệu hợp lệ! Vui lòng kiểm tra lại cấu trúc thư mục 'Data/'.")

    # Chuyển thành định dạng đầu vào cho CNN: (Số_mẫu, Height=30, Width=valid_subcarriers, Channels=1)
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)

    return X, y

# --- 4. CHẠY THỬ NGHIỆM VÀ KIỂM TRA DỮ LIỆU ---
if __name__ == "__main__":
    DATA_PATH = "Data"  # Đường dẫn đến thư mục chứa dữ liệu của bạn
    
    print("--- ĐANG TIẾN HÀNH XỬ LÝ VÀ ĐÓNG GÓI DATASET ---")
    X, y = build_dataset(base_dir=DATA_PATH, window_size=30, stride=5)
    
    print("\n--- KẾT QUẢ XỬ LÝ ---")
    print(f"Tổng số mẫu tạo được (Samples): {X.shape[0]}")
    print(
        f"Kích thước mỗi mẫu (X.shape)   : {X.shape}  -> "
        f"(N, 30 frames, {X.shape[2]} valid subcarriers, 1 channel)"
    )
    print(f"Kích thước nhãn    (y.shape)   : {y.shape}")
    
    # Đếm số lượng mẫu của từng nhãn để kiểm tra cân bằng dữ liệu
    unique, counts = np.unique(y, return_counts=True)
    label_map = {0: "empty", 1: "walking", 2: "standing", 3: "left_hand"}
    for u, c in zip(unique, counts):
        print(f" - Nhãn {u} ({label_map.get(u, 'unknown')}): {c} mẫu")

    # Lưu dataset ra file .npz để tái sử dụng nhanh khi train mà không cần parse lại
    output_file = "csi_dataset.npz"
    np.savez_compressed(output_file, X=X, y=y)
    print(f"\n[OK] Đã lưu dataset hoàn chỉnh vào file: {output_file}")