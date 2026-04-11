@echo off
cd /d "%~dp0"

REM Activate .venv if present, otherwise fall back to venv
if exist ".venv\Scripts\activate.bat" (
	call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
	call venv\Scripts\activate.bat
)

REM Run the desktop control panel by default
python src\main.py --ui

REM Keep window open to see messages
pause
