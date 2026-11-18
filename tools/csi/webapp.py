"""Local control panel for CSI recording, dataset preparation, training and prediction."""
from __future__ import annotations

import json
import os
import math
import subprocess
import sys
import threading
import time
import uuid
import socket
import ipaddress
from functools import lru_cache
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from csi.paths import TOOLS_ROOT as ROOT, PROJECT_ROOT, WEB_ROOT, DATA_ROOT, REPORT_ROOT
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@lru_cache(maxsize=1)
def lan_ipv4_addresses() -> tuple[str, ...]:
    """Discover local LAN addresses without contacting an external service."""
    try:
        candidates = {item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return ()
    addresses = []
    for value in candidates:
        address = ipaddress.IPv4Address(value)
        if address.is_private and not (address.is_loopback or address.is_link_local or address.is_unspecified):
            addresses.append(value)
    return tuple(sorted(addresses))


def server_access_info(server) -> dict:
    host, port = server.server_address[:2]
    if host == '0.0.0.0':
        addresses = lan_ipv4_addresses()
    elif host not in ('127.0.0.1', 'localhost'):
        addresses = (host,)
    else:
        addresses = ()
    return dict(local_url=f'http://127.0.0.1:{port}',
                lan_urls=[f'http://{address}:{port}' for address in addresses],
                port=port, lan_enabled=host not in ('127.0.0.1', 'localhost'))


def run_command(job_id: str, command: list[str]) -> None:
    try:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        with jobs_lock:
            jobs[job_id]["pid"] = process.pid
            jobs[job_id]["process"] = process
            if jobs[job_id]["status"] == "stopped":
                process.terminate()
        assert process.stdout is not None
        for line in process.stdout:
            with jobs_lock:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and event.get('type') == 'capture_countdown':
                        jobs[job_id]['countdown_remaining'] = event['remaining_s']
                        continue
                    if isinstance(event, dict) and event.get('type') == 'capture_summary':
                        jobs[job_id]['capture_summary'] = event
                        continue
                    if isinstance(event, dict) and event.get('type') == 'csi_frame':
                        jobs[job_id].setdefault('frames', deque(maxlen=200)).append(event)
                        jobs[job_id]['last_frame'] = time.time()
                        continue
                except ValueError:
                    pass
                jobs[job_id]["log"] = (jobs[job_id]["log"] + line)[-64000:]
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and jobs[job_id]['action'] == 'predict':
                        jobs[job_id]["prediction"] = event
                        jobs[job_id]['prediction_updated'] = time.time()
                except ValueError:
                    pass
        process.stdout.close()
        code = process.wait()
        with jobs_lock:
            if jobs[job_id]["status"] != "stopped":
                jobs[job_id]["status"] = "done" if code == 0 else "failed"
            jobs[job_id]["returncode"] = code
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["log"] += f"\nLỗi khởi chạy tác vụ: {exc}\n"


def list_recordings() -> list[dict]:
    result = []
    for path in sorted(DATA_ROOT.glob("*/*.csv")):
        result.append({
            "label": path.parent.name,
            "name": path.name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
        })
    return result


def create_manifest() -> Path:
    """Create a session-level manifest without mixing windows between splits."""
    import csv

    files = sorted(DATA_ROOT.glob("*/*.csv"))
    if not files:
        raise ValueError("Chưa có file CSV trong tools/Data.")
    rows = []
    by_label: dict[str, list[Path]] = {}
    for path in files:
        label = path.parent.name.lower().strip()
        if label == "static":
            label = "standing"
        if path.stem.split('_')[0] not in {label, 'static'}:
            raise ValueError(f'Nhãn tên file không khớp thư mục: {path}')
        if label in {"empty", "standing", "walking"}:
            by_label.setdefault(label, []).append(path)
    if not by_label:
        raise ValueError("Tên thư mục dữ liệu phải là empty, standing hoặc walking.")
    missing = {label: len(by_label.get(label, [])) for label in ('empty','standing','walking')
               if len(by_label.get(label, [])) < 3}
    if missing:
        raise ValueError(f'Cần ít nhất 3 recording độc lập mỗi nhãn (train/val/test). Hiện có: {missing}')
    for label, paths in sorted(by_label.items()):
        for index, path in enumerate(paths):
            # Keep complete recordings together; at least three files per class
            # are needed for train/validation/test coverage.
            split = ("train", "val", "test")[index % 3] if len(paths) >= 3 else "train"
            rows.append((f"{label}_{index + 1:03d}", label, split, str(path.resolve())))
    manifest = REPORT_ROOT / "web_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("session_id", "label", "split", "rx_csv"))
        writer.writerows(rows)
    return manifest


def make_command(action: str, params: dict) -> list[str]:
    if action == 'monitor':
        port = str(params.get('port', '')).strip()
        if not port:
            raise ValueError('Cần chọn cổng serial để xem trực tiếp.')
        return [str(PYTHON), str(ROOT / 'run.py'), 'monitor', '--port', port]
    if action == "audit":
        return [str(PYTHON), str(ROOT / 'run.py'), 'audit', '--roots', str(DATA_ROOT)]
    if action == "record":
        label = str(params.get("label", "empty")).strip().lower()
        if label not in {"empty", "standing", "walking"}:
            raise ValueError("Nhãn chỉ được chứa chữ, số, gạch ngang hoặc gạch dưới.")
        duration = float(params.get("duration", 30))
        countdown = int(params.get("countdown", 5))
        if not 0 <= countdown <= 3600:
            raise ValueError("Countdown must be 0..3600 seconds")
        baudrate = int(params.get("baudrate", 2_000_000))
        port = str(params.get("port", "")).strip()
        if not port or not math.isfinite(duration) or duration <= 0 or duration > 86_400:
            raise ValueError("Cần port hợp lệ và thời lượng từ 0 đến 86400 giây.")
        return [str(PYTHON), str(ROOT / "run.py"), "record", "--port", port,
                "--baudrate", str(baudrate), "--action", label,
                "--duration", str(duration), "--countdown", str(countdown)]
    if action == "dataset":
        manifest = create_manifest()
        layout = str(params.get("layout", "ht20_256"))
        fs = float(params.get("fs", 25))
        window = float(params.get("window", 2))
        step = float(params.get("step", 0.5))
        return [str(PYTHON), str(ROOT / "run.py"), "prepare",
                "--manifest", str(manifest), "--out", str(REPORT_ROOT / "features"),
                "--layout", layout, "--fs", str(fs), "--window", str(window),
                "--step", str(step)]
    if action == "train":
        trees = int(params.get("trees", 400))
        if trees < 1 or trees > 10_000:
            raise ValueError("Số cây phải trong khoảng 1 đến 10000.")
        return [str(PYTHON), str(ROOT / "run.py"), "train",
                "--features-dir", str(REPORT_ROOT / "features"),
                "--out", str(REPORT_ROOT / "two_phase_rf"), "--trees", str(trees)]
    if action == "predict":
        port = str(params.get("port", "")).strip()
        baudrate = int(params.get("baudrate", 2_000_000))
        calibration = float(params.get('calibration', 0))
        vote = int(params.get('vote', 5))
        countdown = int(params.get('countdown', 5))
        if not math.isfinite(calibration) or not 0 <= calibration <= 120:
            raise ValueError('Calibration must be 0..120 seconds; keep room EMPTY')
        if vote < 1 or vote > 31 or vote % 2 == 0:
            raise ValueError('Vote must be odd, in 1..31')
        if not 0 <= countdown <= 3600:
            raise ValueError('Countdown must be 0..3600 seconds')
        if not port:
            raise ValueError("Cần chọn port cho dự đoán realtime.")
        return [str(PYTHON), str(ROOT / "run.py"), "predict", "--port", port,
                "--baudrate", str(baudrate),
                "--model", str(REPORT_ROOT / "two_phase_rf" / "model.joblib"),
                '--calibration',str(calibration),'--vote',str(vote),'--countdown',str(countdown)]
    raise ValueError("Tác vụ không hợp lệ.")


def read_metrics():
    path = REPORT_ROOT / 'two_phase_rf' / 'metrics.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        return

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/api/server-info':
            self.send_json(server_access_info(self.server))
            return
        if parsed.path == '/api/signal':
            try:
                from csi.viewer import csv_preview, preview
                query = parse_qs(parsed.query)
                if 'file' in query:
                    path = (PROJECT_ROOT / query['file'][0]).resolve()
                    if not path.is_relative_to(DATA_ROOT.resolve()) or path.suffix.lower() != '.csv':
                        raise ValueError('Chỉ được xem CSV trong tools/Data.')
                    self.send_json(csv_preview(path))
                else:
                    with jobs_lock:
                        job = jobs.get(query.get('job', [''])[0], {})
                        frames = list(job.get('frames', []))
                        last = job.get('last_frame', 0)
                        state = job.get('status', 'idle')
                    import numpy as np
                    # Restarted receiver clocks start a new visible segment.
                    start = 0
                    for i in range(1, len(frames)):
                        if (frames[i]['channel'] != frames[i-1]['channel'] or
                            (frames[i]['timestamp'] <= frames[i-1]['timestamp'] and
                             frames[i-1]['timestamp']-frames[i]['timestamp'] < 2**31)):
                            start = i
                    packets = [{**f, 'amplitude': np.asarray(f['amplitude'])} for f in frames[start:]]
                    result = preview(packets)
                    result.update(status=state, stale=time.time()-last>3)
                    self.send_json(result)
            except (ValueError, OSError) as exc:
                self.send_json({'error': str(exc)}, 400)
            return
        if parsed.path in {'/', '/signal.js', '/prediction.js'}:
            content = (WEB_ROOT / ('index.html' if parsed.path == '/' else parsed.path[1:])).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8" if parsed.path.endswith('.js') else "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
            return
        if parsed.path == "/api/status":
            now = time.time()
            with jobs_lock:
                current = {key: {k: v for k, v in value.items() if k not in {"process", "frames"}}
                           for key, value in jobs.items()}
                for job in current.values():
                    if 'prediction_updated' in job:
                        job['prediction_age_s'] = max(0, now-job['prediction_updated'])
            self.send_json({"jobs": current, "ports": self.serial_ports(),
                            "recordings": list_recordings(), "metrics": read_metrics(),
                            'server_time': now, 'access': server_access_info(self.server)})
            return
        self.send_json({"error": "Không tìm thấy tài nguyên."}, 404)

    @staticmethod
    def serial_ports() -> list[dict]:
        try:
            from serial.tools import list_ports
            return [{"device": p.device, "description": p.description}
                    for p in list_ports.comports()]
        except ImportError:
            return []

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        try:
            if not 0 <= length <= 16384:
                raise ValueError('Request too large')
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError('Expected JSON object')
            if parsed.path == "/api/run":
                action = str(body.get("action", ""))
                job_id = uuid.uuid4().hex[:10]
                with jobs_lock:
                    if any(j['status'] == 'running' for j in jobs.values()):
                        raise ValueError('Hãy dừng hoặc chờ tác vụ hiện tại hoàn tất.')
                    command = make_command(action, body)
                    jobs[job_id] = {"action": action, "status": "running",
                                    "started": time.time(), "log": ""}
                threading.Thread(target=run_command, args=(job_id, command),
                                 daemon=True).start()
                self.send_json({"job_id": job_id})
                return
            if parsed.path == "/api/stop":
                job_id = str(body.get("job_id", ""))
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job or job['status'] != 'running':
                        raise ValueError('Tác vụ không còn đang chạy.')
                    job['status'] = 'stopped'
                    process = job.get('process')
                    if process is not None and process.poll() is None:
                        process.terminate()
                self.send_json({"ok": True})
                return
            self.send_json({"error": "Không tìm thấy API."}, 404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0",
                        help='0.0.0.0 allows phones on LAN; 127.0.0.1 restricts to this PC')
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'Cannot listen on {args.host}:{args.port}: {exc}. Stop the old server or choose --port.\n')
    access = server_access_info(server)
    print(f"CSI control panel (PC): {access['local_url']}", flush=True)
    for url in access['lan_urls']:
        print(f'Phone / same Wi-Fi: {url}', flush=True)
    if access['lan_enabled'] and not access['lan_urls']:
        print('LAN enabled. Check ipconfig for the PC Wi-Fi IPv4 address.', flush=True)
    print('Phone controls the PC; ESP stays connected to the PC USB/Serial port.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
