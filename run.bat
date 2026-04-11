@echo off
cd /d "%~dp0"

REM Activate .venv if present, otherwise fall back to venv
if exist ".venv\Scripts\activate.bat" (
	call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
	call venv\Scripts\activate.bat
)

REM Run the demo
python src\main.py

REM Keep window open to see messages
pause
