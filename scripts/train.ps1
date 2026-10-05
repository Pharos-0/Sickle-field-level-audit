# Train one ladder rung at one seed.
#   .\scripts\train.ps1 -Config v1 -Seed 0
param(
  [Parameter(Mandatory=$true)][string]$Config,
  [int]$Seed = 0,
  [int]$Epochs = 0,
  [string]$Device = ""
)
$ErrorActionPreference = "Stop"
$py = if ($env:SICKLE_PY) { $env:SICKLE_PY } else { "D:\Final Project M.Tech\.venv\Scripts\python.exe" }
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = "utf-8"
$a = @("-m","src.train","--config",$Config,"--seed",$Seed)
if ($Epochs -gt 0) { $a += @("--epochs",$Epochs) }
if ($Device)       { $a += @("--device",$Device) }
& $py @a
Pop-Location
