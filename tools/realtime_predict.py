# realtime_predict.py
import re
import time
import argparse
import serial
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy.signal import butter, filtfilt
from csi_preprocessing import remove_unusable_subcarriers

# --- 1. KHỞI TẠO CẤU HÌNH ---
PORT = "COM8"
BAUD = 2000000
WINDOW_SIZE = 30        # Cần 30 frame (3 giây ở 10Hz) để nhận diện
MODEL_PATH = "csi_cnn_model.pth"

# Danh sách nhãn hành động (Index phải khớp với file lúc huấn luyện)
LABELS = {
    0: "Phòng trống (Empty)",
    1: "Đang đi lại (Walking)",
    2: "Đang đứng yên",
    3: "Đang giơ tay trái"
}

# --- 2. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH ---
class CSI_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CSI_CNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. THIẾT LẬP THIẾT BỊ (GPU / CPU) VÀ NẠP MÔ HÌNH ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f">> Thiết bị suy luận: GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print(">> Không tìm thấy GPU, tự động chuyển sang chế độ: CPU")

num_classes = len(LABELS)
model = CSI_CNN(num_classes=num_classes).to(device)

# Nạp trọng số mô hình đã huấn luyện
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f">> Đã nạp thành công mô hình từ '{MODEL_PATH}' ({num_classes} lớp).")
except Exception as e:
    print(f"[LỖI NẠP MODEL]: {e}")
    exit(1)

model.eval()  # Khóa BatchNorm và tắt Dropout khi suy luận

# --- 4. CÁC HÀM TIỀN XỬ LÝ AN TOÀN ---
def parse_csi_raw(line_str):
    if "CSI_DATA" not in line_str or "[" not in line_str or "]" not in line_str:
        return None
    
    match = re.search(r'\[(.*?)\]', line_str)
    if not match:
        return None

    try:
        tokens = match.group(1).split(',')
        if len(tokens) != 384:
            return None
        raw_vals = [int(x.strip()) for x in tokens]
    except ValueError:
        return None

    raw_arr = np.array(raw_vals, dtype=np.float32)
    imag = raw_arr[0::2]
    real = raw_arr[1::2]
    amp = np.sqrt(real**2 + imag**2)

    if np.isnan(amp).any() or np.isinf(amp).any():
        return None

    cleaned_amp, _, _ = remove_unusable_subcarriers(
        amp.reshape(1, -1), remove_zero_columns=False
    )
    return cleaned_amp[0]

def preprocess_window(buffer_array):
    # Lọc thông thấp Butterworth 3Hz
    b, a = butter(4, 3.0 / (0.5 * 10.0), btype='low', analog=False)
    filtered = filtfilt(b, a, buffer_array, axis=0)
    
    # Chuẩn hóa Z-Score theo từng cửa sổ
    normed = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    # Chuyển đổi sang PyTorch Tensor và đưa lên thiết bị (GPU hoặc CPU)
    tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

# --- 5. VÒNG LẶP SUY LUẬN REAL-TIME ---
def run_realtime():
    # Vô hiệu hóa DTR/RTS để tránh kích hoạt mạch reset tự động trên ESP32
    ser = serial.Serial(PORT, BAUD, timeout=1, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    ser.reset_input_buffer()

    buffer = deque(maxlen=WINDOW_SIZE)
    frame_count = 0

    print(f">> Đã kết nối với cổng {PORT} tại baudrate {BAUD}.")
    print(">> Đang thu thập và chờ đủ 30 frames đầu tiên...")

    try:
        while True:
            raw_line = ser.readline()
            if not raw_line:
                continue

            line = raw_line.decode('utf-8', errors='ignore').strip()
            amp = parse_csi_raw(line)
            
            if amp is not None:
                buffer.append(amp)
                frame_count += 1

                # Hiển thị số lượng frames đang tích lũy
                if len(buffer) < WINDOW_SIZE:
                    print(f"\r>> [Đang tích lũy]: {len(buffer)}/{WINDOW_SIZE} frames...", end="", flush=True)
                    continue

                # Khi đã đủ 30 frames: cứ mỗi 2 frames mới (khoảng 0.2s) thực hiện dự đoán 1 lần
                if frame_count % 2 == 0:
                    input_tensor = preprocess_window(np.array(buffer))
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        pred_idx = np.argmax(probs)
                        confidence = probs[pred_idx] * 100

                    predicted_action = LABELS.get(pred_idx, "Không xác định")
                    print(f"\r[DỰ ĐOÁN]: {predicted_action:<25} | Độ tin cậy: {confidence:5.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        print("\n>> Đã dừng chương trình nhận diện.")
    finally:
        ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run realtime CSI action prediction.")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--baudrate", "--baud", type=int, default=BAUD)
    args = parser.parse_args()
    PORT = args.port
    BAUD = args.baudrate
    run_realtime()