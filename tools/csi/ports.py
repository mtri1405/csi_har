"""Scan available serial ports for CSI receiver/sender devices.

Examples:
  tools\\venv\\Scripts\\python tools\\run.py ports
  tools\\venv\\Scripts\\python tools\\run.py ports --verbose
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    try:
        from serial.tools import list_ports
    except Exception as exc:
        print(f"pyserial is required: {exc}")
        return 2
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return 1
    for p in ports:
        if args.verbose:
            print(f"{p.device}\t{p.description}\t{p.hwid}")
        else:
            print(p.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
