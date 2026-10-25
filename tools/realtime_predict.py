# realtime_predict.py
import re
import argparse
from collections import deque

import numpy as np
import serial
import torch

from csi_model import CSI_CNN
from csi_preprocessing import (
    RAW_SUBCARRIERS,
    REMOVED_SUBCARRIER_INDICES,
    TARGET_SUBCARRIERS,
    WINDOW_SIZE,
    preprocess_window,
)

PORT = "COM8"
BAUD = 2000000
MODEL_PATH = "csi_cnn_model.pth"
RAW_LEN = RAW_SUBCARRIERS * 2
PREDICT_EVERY = 2   # dự đoán mỗi 2 frame mới
SMOOTH_WINDOW = 5   # trung bình xác suất của 5 lần dự đoán gần nhất


# --- NẠP MÔ HÌNH ---
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint cũ không có metadata preprocessing.")

    metadata = checkpoint.get("metadata", {})
    indices = metadata.get("subcarrier_indices")
    if (
        metadata.get("window_size") != WINDOW_SIZE
        or metadata.get("raw_subcarriers") != RAW_SUBCARRIERS
        or metadata.get("target_subcarriers") != TARGET_SUBCARRIERS
        or metadata.get("removed_subcarrier_indices")
        != list(REMOVED_SUBCARRIER_INDICES)
        or not indices
    ):
        raise ValueError("Checkpoint thiếu cấu hình cửa sổ hoặc subcarrier.")

    labels = {int(k): v for k, v in metadata["labels"].items()}
    num_classes = int(metadata["num_classes"])
    if sorted(labels) != list(range(num_classes)):
        raise ValueError(f"Nhãn trong checkpoint không khớp num_classes: {labels}")

    model = CSI_CNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # khóa BatchNorm, tắt Dropout
    return model, labels, np.array(indices, dtype=np.int64)


# --- PARSE 1 DÒNG CSI ---
def parse_csi_raw(line_str, keep_indices):
    if "CSI_DATA" not in line_str or "[" not in line_str or "]" not in line_str:
        return None

    match = re.search(r"\[(.*?)\]", line_str)
    if not match:
        return None

    try:
        tokens = match.group(1).split(",")
        if len(tokens) != RAW_LEN:
            return None
        raw_vals = [int(x.strip()) for x in tokens]
    except ValueError:
        return None

    raw_arr = np.array(raw_vals, dtype=np.float32)
    imag = raw_arr[0::2]
    real = raw_arr[1::2]
    amp = np.sqrt(real ** 2 + imag ** 2)

    if np.isnan(amp).any() or np.isinf(amp).any():
        return None

    if not keep_indices or max(keep_indices) >= TARGET_SUBCARRIERS:
        raise ValueError(
            "Checkpoint không chứa indices subcarrier hợp lệ trên layout 64 kênh."
        )
    return amp[keep_indices]


def tensorize_window(buffer_array, device):
    normed = preprocess_window(buffer_array)
    tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)  # (1, 1, WINDOW_SIZE, W)


# --- VÒNG LẶP SUY LUẬN REAL-TIME ---
def run_realtime(port, baud, model, labels, keep_indices, device, smooth):
    # Tắt DTR/RTS để không kích hoạt reset tự động trên ESP32
    ser = serial.Serial(port, baud, timeout=1, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    ser.reset_input_buffer()

    buffer = deque(maxlen=WINDOW_SIZE)
    prob_history = deque(maxlen=max(1, smooth))
    frame_count = 0

    print(f">> Đã kết nối cổng {port} tại baudrate {baud}.")
    print(f">> Đang thu thập, chờ đủ {WINDOW_SIZE} frame đầu tiên...")

    try:
        while True:
            raw_line = ser.readline()
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="ignore").strip()
            amp = parse_csi_raw(line, keep_indices)
            if amp is None:
                continue

            buffer.append(amp)
            frame_count += 1

            if len(buffer) < WINDOW_SIZE:
                print(f"\r>> [Đang tích lũy]: {len(buffer)}/{WINDOW_SIZE} frames...", end="", flush=True)
                continue

            if frame_count % PREDICT_EVERY != 0:
                continue

            input_tensor = tensorize_window(np.array(buffer), device)
            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1).cpu().numpy()[0]

            # Làm mượt: trung bình xác suất vài lần gần nhất để tránh nhấp nháy nhãn
            prob_history.append(probs)
            avg_probs = np.mean(prob_history, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx]) * 100

            name = labels.get(pred_idx, "Không xác định")
            print(f"\r[DỰ ĐOÁN]: {name:<28} | Độ tin cậy: {confidence:5.1f}%   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n>> Đã dừng chương trình nhận diện.")
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="Run realtime CSI action prediction.")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--baudrate", "--baud", type=int, default=BAUD)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="Số lần dự đoán gần nhất để lấy trung bình (1 = tắt)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f">> Thiết bị suy luận: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(">> Không có GPU, dùng CPU")

    try:
        model, labels, keep_indices = load_model(args.model, device)
        print(f">> Đã nạp mô hình từ '{args.model}' ({len(labels)} lớp): {labels}")
    except Exception as e:
        print(f"[LỖI NẠP MODEL]: {e}")
        raise SystemExit(1)

    run_realtime(args.port, args.baudrate, model, labels, keep_indices, device, args.smooth)


if __name__ == "__main__":
    main()