@echo off
setlocal EnableDelayedExpansion

title MarineVision - Sonar Intelligence

cd /d "%~dp0"

echo.
echo ================================================================================
echo   MARINEVISION - UNDERWATER INTELLIGENCE AND SONAR ANALYTICS
echo   Side-Scan Sonar (SSS) Marine Debris and Acoustic Anomaly Detection
echo ================================================================================
echo.

set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo [*] Using local virtual environment (.venv)
)

if not defined PYTHON_EXE if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    echo [*] Using local virtual environment (venv)
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
        echo [*] Using system Python
    )
)

if not defined PYTHON_EXE (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3"
        echo [*] Using Windows Python Launcher
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Python executable: !PYTHON_EXE!

set "ULTRALYTICS_AUTOINSTALL=0"
set "YOLO_OFFLINE=1"
set "ULTRALYTICS_OFFLINE=1"
set "YOLO_VERBOSE=False"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PIP_NO_INDEX=1"
set "PYTHONUNBUFFERED=1"
set "FLASK_APP=app.py"
set "FLASK_ENV=production"

echo [*] Offline mode active (Ultralytics, YOLO, PyTorch network checks disabled)
echo [*] Opening browser at http://127.0.0.1:5000 in 3 seconds...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

echo.
echo ================================================================================
echo   Status   : Server running locally (100%% Offline)
echo   URL      : http://127.0.0.1:5000
echo   Controls : Press Ctrl+C in this window to stop the server
echo ================================================================================
echo.

!PYTHON_EXE! app.py

if !errorlevel! neq 0 (
    echo.
    echo [!] Server exited with code !errorlevel!.
    echo.
    pause
)
