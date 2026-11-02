import json
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Data"
HOST = "127.0.0.1"
PORT = 8765
ALLOWED_SCRIPTS = {
    "record": "record_to_csv.py",
    "dataset": "dataset_pipeline.py",
    "train": "train.py",
    "predict": "realtime_predict.py",
}

process_lock = threading.Lock()
processes = {}
monitor_lock = threading.Lock()
monitor_state = {
    "thread": None,
    "stop_event": None,
    "samples": deque(maxlen=300),
    "error": None,
    "port": None,
    "baudrate": None,
}


def python_executable():
    candidate = ROOT / "venv" / "Scripts" / "python.exe"
    return str(candidate if candidate.exists() else Path(sys.executable))


def list_ports():
    try:
        from serial.tools import list_ports
        return [
            {"device": port.device, "description": port.description}
            for port in list_ports.comports()
        ]
    except ImportError as exc:
        raise RuntimeError("Không thể tải pyserial để liệt kê cổng COM.") from exc


def running_process_exists():
    with process_lock:
        return any(state["process"].poll() is None for state in processes.values())


def _safe_data_file(relative_name):
    candidate = (DATA_ROOT / unquote(relative_name)).resolve()
    if DATA_ROOT.resolve() not in candidate.parents or candidate.suffix.lower() != ".csv":
        raise ValueError("Chỉ được xem file CSV bên trong thư mục Data.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {relative_name}")
    return candidate


def list_recordings():
    if not DATA_ROOT.exists():
        return []
    recordings = []
    for file_path in sorted(DATA_ROOT.glob("*/*.csv")):
        recordings.append({
            "path": file_path.relative_to(DATA_ROOT).as_posix(),
            "label": file_path.parent.name,
            "name": file_path.name,
        })
    return recordings


def filter_options(query):
    def number(name, default, cast):
        raw = query.get(name, [str(default)])[0]
        try:
            return cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} không hợp lệ.") from exc

    return {
        "mode": query.get("filter", ["pipeline"])[0],
        "duration": number("duration", 15.0, float),
        "subcarrier": number("subcarrier", 0, int),
        "cutoff": number("cutoff", 4.0, float),
        "order": number("order", 4, int),
        "hampel_window": number("hampel_window", 7, int),
        "hampel_sigma": number("hampel_sigma", 3.0, float),
    }


def start_monitor(port, baudrate):
    import serial
    from csi_visualization import parse_csi_line

    if running_process_exists():
        raise ValueError("Hãy dừng process hiện tại trước khi mở bộ xem live.")
    with monitor_lock:
        if monitor_state["thread"] and monitor_state["thread"].is_alive():
            raise ValueError("Bộ xem live đã chạy.")

    connection = serial.Serial(
        port, baudrate, timeout=0.2, dsrdtr=False, rtscts=False
    )
    try:
        connection.setDTR(False)
        connection.setRTS(False)
    except (AttributeError, serial.SerialException):
        pass
    connection.reset_input_buffer()

    with monitor_lock:
        try:
            if monitor_state["thread"] and monitor_state["thread"].is_alive():
                raise ValueError("Bộ xem live đã chạy.")
            stop_event = threading.Event()
            monitor_state.update({
                "thread": None,
                "stop_event": stop_event,
                "samples": deque(maxlen=300),
                "error": None,
                "port": port,
                "baudrate": baudrate,
            })
        except ValueError:
            connection.close()
            raise

    def read_serial():
        try:
            while not stop_event.is_set():
                raw_line = connection.readline()
                if not raw_line:
                    continue
                sample = parse_csi_line(raw_line.decode("utf-8", errors="ignore"))
                if sample is not None:
                    with monitor_lock:
                        monitor_state["samples"].append(sample)
        except (serial.SerialException, OSError, ValueError) as exc:
            with monitor_lock:
                monitor_state["error"] = str(exc)
        finally:
            connection.close()
            with monitor_lock:
                monitor_state["thread"] = None

    thread = threading.Thread(target=read_serial, daemon=True)
    with monitor_lock:
        monitor_state["thread"] = thread
    thread.start()


def stop_monitor():
    with monitor_lock:
        stop_event = monitor_state["stop_event"]
        thread = monitor_state["thread"]
    if stop_event:
        stop_event.set()
    if thread:
        thread.join(timeout=2)


def monitor_payload(query):
    from csi_visualization import summarize_series
    options = filter_options(query)
    with monitor_lock:
        samples = np.asarray(monitor_state["samples"], dtype=np.float64)
        error = monitor_state["error"]
        running = bool(monitor_state["thread"] and monitor_state["thread"].is_alive())
    if samples.size == 0:
        return {
            "running": running,
            "error": error,
            "frames": 0,
            "sample_rate_hz": 10.0,
            "subcarrier": options["subcarrier"],
            "raw": [],
            "filtered": [],
        }
    payload = summarize_series(samples, **options)
    payload.update({"running": running, "error": error})
    return payload


def start_process(kind, args):
    script = ALLOWED_SCRIPTS.get(kind)
    if not script:
        raise ValueError("Tác vụ không hợp lệ.")
    if kind in {"record", "predict"} and monitor_state["thread"] and monitor_state["thread"].is_alive():
        raise ValueError("Hãy dừng bộ xem live trước khi dùng serial cho process.")
    command = [python_executable(), str(ROOT / script)]
    if kind == "record":
        command += [
            "--port", str(args.get("port", "COM8")),
            "--baudrate", str(int(args.get("baudrate", 2000000))),
            "--action", str(args.get("action", "standing")),
            "--duration", str(float(args.get("duration", 30))),
        ]
    elif kind == "predict":
        command += [
            "--port", str(args.get("port", "COM8")),
            "--baudrate", str(int(args.get("baudrate", 2000000))),
        ]

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    run_id = uuid.uuid4().hex
    with process_lock:
        processes[run_id] = {"process": proc, "lines": [], "done": False}

    def collect():
        state = processes[run_id]
        for line in proc.stdout:
            state["lines"].append(line.rstrip("\r\n"))
        proc.wait()
        state["done"] = True
        state["returncode"] = proc.returncode

    threading.Thread(target=collect, daemon=True).start()
    return run_id


def extract_evaluation(lines):
    prefix = "EVALUATION_JSON: "
    for line in reversed(lines):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                return None
    return None


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/ports":
            try:
                self.send_json({"ports": list_ports()})
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/config":
            from csi_preprocessing import (
                FILTER_CUTOFF_HZ,
                FILTER_ORDER,
                RAW_SUBCARRIERS,
                SAMPLE_RATE_HZ,
                WINDOW_SIZE,
            )
            self.send_json({
                "raw_subcarriers": RAW_SUBCARRIERS,
                "window_size": WINDOW_SIZE,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "filter_cutoff_hz": FILTER_CUTOFF_HZ,
                "filter_order": FILTER_ORDER,
                "filters": [
                    {"value": "none", "label": "Không lọc"},
                    {"value": "hampel", "label": "Hampel"},
                    {"value": "butterworth", "label": "Butterworth"},
                    {"value": "hampel_butterworth", "label": "Hampel + Butterworth"},
                    {"value": "pipeline", "label": "Full pipeline"},
                ],
            })
            return
        if parsed.path == "/api/recordings":
            self.send_json({"recordings": list_recordings()})
            return
        if parsed.path == "/api/preview":
            try:
                relative_name = query.get("file", [""])[0]
                from dataset_pipeline import parse_csi_file
                data = parse_csi_file(_safe_data_file(relative_name))
                if data.size == 0:
                    raise ValueError("CSV không có frame CSI hợp lệ.")
                from csi_visualization import summarize_series
                self.send_json(summarize_series(data, **filter_options(query)))
            except (FileNotFoundError, ValueError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/monitor":
            try:
                self.send_json(monitor_payload(query))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/")
            state = processes.get(run_id)
            if not state:
                self.send_json({"error": "Không tìm thấy tiến trình."}, 404)
                return
            self.send_json({
                "running": state["process"].poll() is None,
                "done": state["done"],
                "returncode": state.get("returncode"),
                "lines": state["lines"][-500:],
                "evaluation": extract_evaluation(state["lines"]),
            })
            return
        if parsed.path in ("/", "/csi_web.html"):
            content = (ROOT / "csi_web.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json({"error": "Dữ liệu JSON không hợp lệ."}, 400)
            return

        if parsed.path == "/api/run":
            try:
                run_id = start_process(data.get("kind"), data)
                self.send_json({"runId": run_id})
            except (ValueError, TypeError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/monitor/start":
            try:
                port = str(data.get("port", "")).strip()
                baudrate = int(data.get("baudrate", 0))
                if not port or baudrate <= 0:
                    raise ValueError("Cần cổng serial và baudrate hợp lệ.")
                start_monitor(port, baudrate)
                self.send_json({"running": True})
            except (ValueError, TypeError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/monitor/stop":
            stop_monitor()
            self.send_json({"running": False})
            return
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
            run_id = parsed.path.split("/")[3]
            state = processes.get(run_id)
            if not state:
                self.send_json({"error": "Không tìm thấy tiến trình."}, 404)
                return
            if state["process"].poll() is None:
                state["process"].terminate()
            self.send_json({"stopped": True})
            return
        self.send_error(404)


if __name__ == "__main__":
    print(f"CSI web UI: http://{HOST}:{PORT}", flush=True)
    print(f"Python runtime: {python_executable()}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
