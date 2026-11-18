"""Stable project paths, independent of the launch working directory."""
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
DATA_ROOT = TOOLS_ROOT / 'Data'
REPORT_ROOT = TOOLS_ROOT / 'reports'
WEB_ROOT = TOOLS_ROOT / 'web'
