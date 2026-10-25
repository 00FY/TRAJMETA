$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create the .venv environment using README.md first.' }
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath 'reports/study-v1/experiment_evidence.zip') {
        & $pythonExe scripts/study_evidence.py restore
        if ($LASTEXITCODE -ne 0) { throw 'Published evidence restoration failed.' }
    }
    $manifest = Import-Csv -LiteralPath 'data/audits/study-v1/dataset_manifest.csv'
    $missing = @($manifest | Where-Object { -not (Test-Path -LiteralPath $_.path) })
    if ($missing.Count -gt 0) {
        & $pythonExe -m trajmeta.benchmark restore --output data/audits/study-v1
        if ($LASTEXITCODE -ne 0) { throw 'Dataset restoration failed; inspect upstream version/access errors.' }
    }
    & $pythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    & $pythonExe -m trajmeta.study --phase all
    if ($LASTEXITCODE -ne 0) { throw 'Study stopped; inspect the reported diagnostic or incomplete run.' }
    & $pythonExe scripts/report_study.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
    & $pythonExe scripts/export_results.py
    if ($LASTEXITCODE -ne 0) { throw 'CSV export failed.' }
    & $pythonExe scripts/verify_study.py
    if ($LASTEXITCODE -ne 0) { throw 'Completion verification failed.' }
    & $pythonExe scripts/timing_audit.py
    if ($LASTEXITCODE -ne 0) { throw 'Timing audit failed.' }
    & $pythonExe scripts/finalize_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report finalization failed.' }
} finally {
    Pop-Location
}
