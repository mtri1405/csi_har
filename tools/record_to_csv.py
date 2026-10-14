import os
import time
import argparse
import serial

DEFAULT_PORT = "COM8"
DEFAULT_BAUD = 2000000
DEFAULT_ACTION_LABEL = "standing"
DEFAULT_DURATION_SEC = 30
CSV_HEADER = (
    "type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
    "aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,"
    "secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data"
)

def main():
    parser = argparse.ArgumentParser(description="Record CSI frames to CSV.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", "--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--action", default=DEFAULT_ACTION_LABEL)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    args = parser.parse_args()

    save_dir = os.path.join("Data", args.action)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{args.action}_{int(time.time())}.csv"
    file_path = os.path.join(save_dir, filename)
    ser = serial.Serial(args.port, args.baudrate, timeout=1)

    print(f"Bắt đầu thu thập cho hành động: {args.action} trong {args.duration:g}s...", flush=True)
    print(f"Đường dẫn lưu trữ: {file_path}", flush=True)
    start_time = time.time()
    sample_count = 0

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER + "\n")
            while time.time() - start_time < args.duration:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if "CSI_DATA" in line and "[" in line and "]" in line:
                    f.write(line + "\n")
                    sample_count += 1
                    if sample_count % 100 == 0:
                        print(f"Đã ghi {sample_count} gói tin...", flush=True)
    finally:
        ser.close()
    print(f"Hoàn tất! Đã ghi nhận {sample_count} gói tin. File lưu tại: {file_path}", flush=True)


if __name__ == "__main__":
    main()