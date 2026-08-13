$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Here ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Dexter tools venv was not found at: $Python"
}

& $Python -m pip install -r (Join-Path $Here "requirements.txt")
