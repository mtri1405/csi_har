import os
import time
import argparse
import csv
import json
import math
import serial
from csi.protocol import parse_packet, unwrap_timestamps
from csi.paths import DATA_ROOT

DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 2000000
DEFAULT_ACTION_LABEL = "empty"
DEFAULT_DURATION_SEC = 30

CSV_HEADER = ",".join((
    "type", "seq", "timestamp", "rssi", "noise_floor", "channel",
    "second_channel", "sig", "mcs", "cwb", "stbc", "len",
    "first_word_invalid", "dropped", "data", "pc_elapsed_s", "pc_received_unix_s"
))

def main():
    parser = argparse.ArgumentParser(description="Record CSI frames to CSV.")
    parser.add_argument("--port", "--p", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", "--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--action", default=DEFAULT_ACTION_LABEL)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument('--countdown', type=int, default=0,
                        help='Seconds to wait before recording (0..3600)')
    args = parser.parse_args()
    if not 0 <= args.countdown <= 3600:
        parser.error('Countdown must be 0..3600 seconds')

    if args.action not in ('empty', 'standing', 'walking') or not math.isfinite(args.duration) or not 0 < args.duration <= 86400:
        parser.error('Use empty/standing/walking and duration in (0,86400]')
    save_dir = str(DATA_ROOT / args.action)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{args.action}_{time.time_ns()}.csv"
    file_path = os.path.join(save_dir, filename)
    ser = serial.Serial(args.port, args.baudrate, timeout=0.1)
    try:
        wait_for_countdown(args.countdown)
    except BaseException:
        ser.close()
        raise
    # Remove OS-buffered bytes from before this recording. ESP queue is separate.
    ser.reset_input_buffer()

    print(f"Bắt đầu thu thập cho hành động: {args.action} trong {args.duration:g}s...", flush=True)
    print(f"Đường dẫn lưu trữ: {file_path}", flush=True)
    start_time = time.monotonic()
    sample_count = 0
    skipped_count = 0
    previous_dropped = None
    first_dropped = None
    received_times = []
    pending = bytearray()
    next_progress = 10.0
    serial_bytes = 0
    serial_lines = 0
    non_csi_lines = 0

    try:
        with open(file_path, "w", encoding="utf-8", newline='') as f:
            f.write(CSV_HEADER + "\n")
            writer = csv.writer(f, lineterminator='\n')
            while time.monotonic() - start_time < args.duration:
                chunk = ser.read(max(1, min(ser.in_waiting, 65536)))
                serial_bytes += len(chunk)
                pc_elapsed = time.monotonic() - start_time
                pc_unix = time.time()
                pending.extend(chunk)
                lines = pending.split(b'\n')
                pending = bytearray(lines.pop())
                for raw_line in lines:
                    serial_lines += 1
                    line = raw_line.decode('utf-8', errors='replace').strip()
                    if not line.startswith('CSI_DATA,'):
                        if 'CSI_DATA' in line:
                            skipped_count += 1
                        elif line:
                            non_csi_lines += 1
                        continue
                    packet = parse_packet(line)
                    if packet is None:
                        skipped_count += 1
                        continue
                    fields = next(csv.reader([line]))
                    writer.writerow(fields + [f'{pc_elapsed:.6f}', f'{pc_unix:.6f}'])
                    sample_count += 1
                    received_times.append((pc_elapsed, packet['timestamp']))
                    dropped = packet['dropped']
                    if first_dropped is None:
                        first_dropped = dropped
                    previous_dropped = dropped
                if len(pending) > 65536:
                    skipped_count += 1
                    pending.clear()
                if pc_elapsed >= next_progress:
                    f.flush()
                    print(f'{int(next_progress)}s: {sample_count} dong', flush=True)
                    next_progress = (int(pc_elapsed // 10) + 1) * 10
    finally:
        ser.close()
    elapsed = max(time.monotonic() - start_time, 1e-6)
    summary = capture_timing_summary(received_times, elapsed, args.duration)
    summary.update(packets=sample_count, serial_bytes=serial_bytes,
                   serial_lines=serial_lines, rejected_frames=skipped_count,
                   non_csi_lines=non_csi_lines, partial_bytes_at_stop=len(pending))
    print(json.dumps({'type': 'capture_summary', **summary}, ensure_ascii=False), flush=True)
    print(f"Thời gian PC: {elapsed:.3f}s; khoảng thời gian CSI theo ESP: "
          f"{summary['esp_packet_span_s']}s. Host timestamp là lúc nhận dòng Serial.", flush=True)
    for warning in summary['warnings']:
        print(f'CẢNH BÁO: {warning}', flush=True)
    if sample_count == 0:
        reason = ('No serial bytes received: check COM, board power and firmware.'
                  if serial_bytes == 0 else
                  'Serial data received but no valid CSI: check baud, AP link and layout.')
        raise SystemExit(reason)
    total_dropped = max(0, (previous_dropped or 0) - (first_dropped or 0))
    print(
        f"Hoàn tất! Đã ghi nhận {sample_count} gói tin 128-subcarrier. "
        f"Bỏ qua {skipped_count} frame không hợp lệ. "
        f"Average rate: receive={sample_count / elapsed:.1f}/s, "
        f"drop={total_dropped / elapsed:.1f}/s. "
        f"File lưu tại: {file_path}",
        flush=True,
    )


def wait_for_countdown(seconds):
    """Countdown is outside capture duration; buffered serial is cleared after it."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = max(0, math.ceil(deadline-time.monotonic()))
        print(json.dumps(dict(type='capture_countdown', remaining_s=remaining)), flush=True)
        if remaining == 0:
            return
        time.sleep(min(1, max(0, deadline-time.monotonic())))


def capture_timing_summary(received_times, elapsed, requested):
    """Compare receipt and acquisition clocks without altering either clock."""
    summary = dict(pc_elapsed_s=elapsed, requested_duration_s=requested,
                   pc_packet_span_s=None, esp_packet_span_s=None,
                   first_packet_pc_s=None, last_packet_pc_s=None, warnings=[])
    if not received_times:
        summary['warnings'].append('No valid CSI packets received during capture.')
        return summary
    first, last = received_times[0][0], received_times[-1][0]
    summary.update(first_packet_pc_s=first, last_packet_pc_s=last,
                   pc_packet_span_s=last-first)
    try:
        t = unwrap_timestamps([r[1] for r in received_times])
        span = float((t[-1]-t[0])*1e-6)
        summary['esp_packet_span_s'] = round(span, 6)
        if last-first > 5 and abs(span-(last-first)) > max(1, (last-first)*.1):
            summary['warnings'].append('Clock ESP lệch thời gian nhận PC: kiểm tra backlog Serial và clock firmware.')
    except ValueError as exc:
        summary['warnings'].append(f'Timestamp ESP không liên tục: {exc}')
    if last-first < requested*.9:
        summary['warnings'].append('Các gói CSI không phủ đủ thời gian thu trên PC; kiểm tra khởi động và mất tín hiệu.')
    return summary


if __name__ == "__main__":
    main()
