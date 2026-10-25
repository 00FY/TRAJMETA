param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Set up the project virtual environment first; see README.md.'
}
Push-Location $ProjectRoot
try { & $Python webapp/server.py --port $Port }
finally { Pop-Location }
