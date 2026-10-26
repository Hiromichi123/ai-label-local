@echo off
setlocal
cd /d "%~dp0"
set "YOLO_LABEL_WORKSPACE=%~dp0workspace"
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\sync_to_remote.py" %*
pause
