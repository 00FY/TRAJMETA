$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CommandPattern = '^"?' + [regex]::Escape($Python) + '"?\s+webapp[/\\]server\.py(?:\s|$)'
$Servers = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -match $CommandPattern }
foreach ($Server in $Servers) {
    & taskkill /PID $Server.ProcessId /T /F
}
if (-not $Servers) { Write-Output 'No TrajMeta UI server is running from this environment.' }
