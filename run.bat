@echo off
cd /d "%~dp0"

if /I "%TERM_PROGRAM%"=="vscode" (
	start "" /B powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
	exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
exit /b %errorlevel%
