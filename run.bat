@echo off
cd /d "%~dp0"
REM Activate the virtual environment
call venv\Scripts\activate
REM Run the demo
python src\main.py
REM Keep window open to see messages
pause
