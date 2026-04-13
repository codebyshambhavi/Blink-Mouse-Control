@echo off
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

if not defined PYTHON_EXE (
	echo [ERROR] No virtual environment python found in .venv or venv.
	echo Create one first, for example: python -m venv .venv
	exit /b 1
)

"%PYTHON_EXE%" src\main.py
exit /b %errorlevel%
