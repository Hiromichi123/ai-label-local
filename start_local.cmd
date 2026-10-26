@echo off
setlocal
cd /d "%~dp0"
set "YOLO_LABEL_WORKSPACE=%~dp0workspace"
echo Open http://127.0.0.1:8765 in your browser. Keep this window open.
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765
if errorlevel 1 pause
