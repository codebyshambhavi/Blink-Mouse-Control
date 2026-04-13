$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$pythonExe = $null
if (Test-Path ".venv\Scripts\python.exe") {
    $pythonExe = ".venv\Scripts\python.exe"
} elseif (Test-Path "venv\Scripts\python.exe") {
    $pythonExe = "venv\Scripts\python.exe"
}

if (-not $pythonExe) {
    Write-Error "No virtual environment python found in .venv or venv. Create one with: python -m venv .venv"
    exit 1
}

& $pythonExe "src/main.py"
exit $LASTEXITCODE
