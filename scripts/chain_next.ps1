# Wait for a running training job to exit, then start the next rung.
#
#   .\scripts\chain_next.ps1 -WaitPid 18516 -Next v2 -Seed 0
#
# These are multi-hour CPU runs and two of them at once would halve each
# other's throughput, so the ladder is strictly sequential. This is the same
# discipline that scripts/run_ladder.sh enforces, applied to a job that is
# already running detached.
#
# The rungs are INDEPENDENT -- v2 does not consume v1's output -- so v2 starts
# whether or not v1 finished cleanly. What v1 did is recorded, never inferred.

param(
  [Parameter(Mandatory=$true)][int]$WaitPid,
  [Parameter(Mandatory=$true)][string]$Next,
  [int]$Seed = 0,
  [string]$Prev = "",
  [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py   = if ($env:SICKLE_PY) { $env:SICKLE_PY } else { "D:\Final Project M.Tech\.venv\Scripts\python.exe" }
$logs = Join-Path $root "outputs\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$chainLog = Join-Path $logs "chain_$Next`_seed$Seed.status"

function Say($msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  $line | Tee-Object -FilePath $chainLog -Append | Out-Null
}

Say "waiting for PID $WaitPid to exit before starting $Next seed $Seed"

while ($true) {
  $p = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
  if (-not $p) { break }
  Start-Sleep -Seconds $PollSeconds
}
Say "PID $WaitPid has exited"

# --- report what the previous rung actually achieved, without guessing -------
if ($Prev) {
  $bestPath = Join-Path $root "outputs\crop_type\$Prev`_seed$Seed\best.json"
  if (Test-Path $bestPath) {
    try {
      $b = Get-Content $bestPath -Raw | ConvertFrom-Json
      $state = if ($b.complete) { "COMPLETE" } else { "INCOMPLETE" }
      Say ("{0}: {1} -- best epoch {2}, pixel macro IoU {3}, paddy {4}" -f `
           $Prev, $state, $b.best_epoch,
           [math]::Round($b.macro_iou,4), [math]::Round($b.paddy_iou,4))
      if ($b.paddy_iou -lt 1e-6) { Say "$Prev DEGENERATE: paddy IoU is 0 (one class predicted everywhere)" }
    } catch { Say "$Prev best.json present but unreadable: $_" }
  } else {
    Say "$Prev produced NO best.json -- treat as FAILED, do not interpolate"
  }
}

# --- start the next rung ----------------------------------------------------
$log = Join-Path $logs "$Next`_seed$Seed.log"
$err = Join-Path $logs "$Next`_seed$Seed.err"
$env:PYTHONIOENCODING = "utf-8"

$proc = Start-Process -FilePath $py `
        -ArgumentList @("-u","-m","src.train","--config",$Next,"--seed",$Seed) `
        -WorkingDirectory $root -RedirectStandardOutput $log `
        -RedirectStandardError $err -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 20
# Start-Process returns a thin launcher; find the python actually doing work.
$worker = Get-Process python* -ErrorAction SilentlyContinue |
          Where-Object { $_.Id -ne $proc.Id -and $_.StartTime -gt (Get-Date).AddMinutes(-3) } |
          Sort-Object WorkingSet64 -Descending | Select-Object -First 1
$wid = if ($worker) { $worker.Id } else { $proc.Id }

$proc.Id | Out-File -Encoding ascii (Join-Path $logs "$Next`_seed$Seed.pid")
$wid     | Out-File -Encoding ascii (Join-Path $logs "$Next`_seed$Seed.worker.pid")
Say "started $Next seed $Seed -- launcher PID $($proc.Id), worker PID $wid"
Say "log: $log"
