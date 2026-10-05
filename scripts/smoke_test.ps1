# Prove the pipeline moves before committing hours to it. NOT a result.
param([string[]]$Rungs = @("v1","v2"))
$ErrorActionPreference = "Stop"
$py = if ($env:SICKLE_PY) { $env:SICKLE_PY } else { "D:\Final Project M.Tech\.venv\Scripts\python.exe" }
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = "utf-8"
& $py scripts/check_configs.py
foreach ($v in $Rungs) {
  Write-Host "=== smoke $v ==="
  & $py -m src.train --config $v --seed 0 --smoke --max-batches 2
}
Pop-Location
