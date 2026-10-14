@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  start "CSI Web UI" "venv\Scripts\python.exe" "csi_web_server.py"
) else (
  start "CSI Web UI" python "csi_web_server.py"
)
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765/"
endlocal
