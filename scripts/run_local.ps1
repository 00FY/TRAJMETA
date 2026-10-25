param([ValidateSet('run','bank','audit','test')][string]$Command = 'run')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment not found. Follow README setup instructions.'
}
Push-Location $projectRoot
try {
    if ($Command -eq 'test') { & $pythonPath -m pytest -q }
    else { & $pythonPath -m trajmeta.cli $Command }
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
