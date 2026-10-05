# Stage, commit and push in one step.
#
#   .\scripts\save.ps1 "what changed"
#   .\scripts\save.ps1 "what changed" -NoPush
#
# Refuses to commit a checkpoint or dataset file even if .gitignore is wrong,
# because a 12 MB .pth.tar pushed once stays in git history forever.

param(
  [Parameter(Mandatory=$true, Position=0)][string]$Message,
  [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# --- refuse things that must never enter history ----------------------------
git add -A
$staged = git diff --cached --name-only
if (-not $staged) { Write-Host "nothing to commit"; exit 0 }

$banned = $staged | Where-Object {
  $_ -match '\.(pth|pth\.tar|npz|tif|zip)$' -or $_ -match '^sickle_data/'
}
if ($banned) {
  Write-Host "REFUSING TO COMMIT -- these do not belong in git:" -ForegroundColor Red
  $banned | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
  Write-Host "Unstage them (git reset) and fix .gitignore first."
  exit 1
}

# --- warn about anything large ---------------------------------------------
foreach ($f in $staged) {
  if (Test-Path -LiteralPath $f) {
    $kb = [math]::Round((Get-Item -LiteralPath $f).Length / 1KB, 0)
    if ($kb -gt 1024) { Write-Host "  large file: $f  ($kb KB)" -ForegroundColor Yellow }
  }
}

Write-Host "`nstaging $($staged.Count) file(s)"
git commit -m $Message
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($NoPush) { Write-Host "`ncommitted, not pushed (-NoPush)"; exit 0 }

$remote = git remote
if (-not $remote) {
  Write-Host "`nNo git remote configured, so nothing was pushed." -ForegroundColor Yellow
  Write-Host "Set one up once with:"
  Write-Host "    git remote add origin https://github.com/<you>/<repo>.git"
  Write-Host "    git push -u origin main"
  exit 0
}

$branch = git rev-parse --abbrev-ref HEAD
git push origin $branch
if ($LASTEXITCODE -eq 0) { Write-Host "`npushed $branch" -ForegroundColor Green }
