# Prove the pipeline moves before committing hours to it. NOT a result.
param([string[]]$Rungs = @("v1","v2"))
$ErrorActionPreference = "Stop"
# Interpreter resolution, in order: $env:SICKLE_PY, the project's own .venv,
# then `python` on PATH. The project must not depend on another project's env.
$py = if ($env:SICKLE_PY) { $env:SICKLE_PY }
      else { Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe" }
if (-not (Test-Path $py)) { $py = "python" }
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = "utf-8"
& $py scripts/check_configs.py
foreach ($v in $Rungs) {
  Write-Host "=== smoke $v ==="
  & $py -m src.train --config $v --seed 0 --smoke --max-batches 2
}
Pop-Location
