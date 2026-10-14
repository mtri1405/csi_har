import json
import os
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
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


def start_process(kind, args):
    script = ALLOWED_SCRIPTS.get(kind)
    if not script:
        raise ValueError("Tác vụ không hợp lệ.")
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
        if parsed.path == "/api/ports":
            try:
                self.send_json({"ports": list_ports()})
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 500)
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
