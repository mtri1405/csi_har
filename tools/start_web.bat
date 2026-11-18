@echo off
cd /d "%~dp0\.."
if exist "tools\venv\Scripts\python.exe" (
  "tools\venv\Scripts\python.exe" -X utf8 "tools\run.py" web --host 0.0.0.0 %*
) else (
  py -X utf8 "tools\run.py" web --host 0.0.0.0 %*
)
