#!/usr/bin/env bash
# What happens when the ladder seeds land, without anyone present.
#
# Ten runs (v3 and v4 at seeds 3 to 7) take about 24 hours. Nothing
# downstream updates by itself: the ladder CSV, the power analysis, the
# tolerance registry and the manuscript all have to be re-derived, and the
# sweep has to be resumed. This script does that in order and writes a
# report, so the work is not waiting on someone being awake.
#
# It is deliberately NOT silent about the one thing it must not decide on
# its own. Adding seeds legitimately changes seed-dependent headline
# numbers, and drift_check.py is built to stop the build when that happens.
# This script runs drift_check twice: once to record exactly what moved,
# and then --update to adopt the new values, leaving a dated amendment stub
# that a human has to complete. Adopting a value is not the same as
# agreeing with it, and the stub says so.
#
#   bash scripts/after_seeds.sh
#
# Safe to run more than once: every step is idempotent, and the training
# loop skips runs already marked complete.

set -u
cd "D:/FPD/sickle-project" || exit 1
PY=".venv/Scripts/python.exe"
LOG="logs_after_seeds.log"
: > "$LOG"

say() { echo "$@" | tee -a "$LOG"; }
step() {
  say ""
  say "=== $1 $(date +%H:%M:%S) ==========================================="
  shift
  "$@" >> "$LOG" 2>&1
  local rc=$?
  say "--- exit $rc"
  return $rc
}

say "after_seeds: started $(date)"

# ---------------------------------------------------------------- 1. wait
say ""
say "Waiting for logs_ladder_seeds.log to report completion."
# A plain wait-for-the-marker loop waits for ever if the training job dies,
# which is how the earlier out-of-memory kill would have left this script
# hanging silently. So the wait also gives up when the job is gone AND its
# log has stopped growing, and says which of the two happened.
STALL_LIMIT=5400          # 90 minutes without log growth, against ~87 s/epoch
last_size=-1
stalled=0
reason="completed normally"
while true; do
  if grep -q "LADDER SEEDS COMPLETE" logs_ladder_seeds.log 2>/dev/null; then
    break
  fi
  size=$(wc -c < logs_ladder_seeds.log 2>/dev/null || echo 0)
  running=$(ps -W 2>/dev/null | grep -c "src.train" || echo 0)
  if [ "$size" = "$last_size" ] && [ "$running" -eq 0 ]; then
    stalled=$((stalled + 60))
    if [ "$stalled" -ge "$STALL_LIMIT" ]; then
      reason="ABANDONED: no training process and no log growth for $((STALL_LIMIT / 60)) minutes"
      break
    fi
  else
    stalled=0
  fi
  last_size=$size
  sleep 60
done
say "Ladder seeds wait ended at $(date): $reason"
if [ "$reason" != "completed normally" ]; then
  say ""
  say "WARNING. The seed job did not report completion. Everything below"
  say "runs on whatever seeds DID finish: the completeness filter excludes"
  say "interrupted runs, so the ladder stays correct, but it will carry"
  say "fewer seeds than intended and the paper will say so."
fi

# Which of the ten actually completed. A run killed part way leaves a
# best.json behind, so this asks the trainer's flag rather than the
# directory listing.
"$PY" - <<'PYEOF' | tee -a "$LOG"
import sys
sys.path.insert(0, ".")
from src.utils import completed_seed_runs
for rung in ("v3", "v4"):
    kept, dropped = completed_seed_runs("crop_type", rung)
    print(f"{rung}: {len(kept)} complete "
          f"({', '.join(sorted(d.name for d, _ in kept))})")
    if dropped:
        print(f"    INCOMPLETE, excluded: "
              f"{', '.join(sorted(d.name for d, _ in dropped))}")
PYEOF

# ------------------------------------------------- 2. re-derive the ladder
step "aggregate ladder"        "$PY" scripts/aggregate_ladder.py
step "ablation power"          "$PY" scripts/ladder_power.py

# --------------------------------------- 3. record the drift, then adopt it
say ""
say "=== drift BEFORE adopting the new seed counts ========================"
"$PY" scripts/drift_check.py >> "$LOG" 2>&1
say "--- exit $? (a non-zero exit here is expected: seed-dependent values"
say "    move when seeds are added, and the registry is built to stop for it)"
cp results/drift_check.json results/drift_check_before_new_seeds.json

step "adopt new expected values" "$PY" scripts/drift_check.py --update

# leave an amendment stub rather than a silent edit
"$PY" - <<'PYEOF' | tee -a "$LOG"
import datetime
import json
import pathlib

reg = pathlib.Path("configs/tolerances.yaml")
s = reg.read_text(encoding="utf8").replace("\r\n", "\n")
before = json.loads(pathlib.Path(
    "results/drift_check_before_new_seeds.json").read_text(encoding="utf8"))
moved = [e for e in before["entries"]
         if e["status"] in ("DRIFT", "SEEDS-CHANGED")]
if not moved:
    print("no seed-dependent value moved; no amendment needed")
else:
    rows = "".join(
        f"\n        {e['id']}: {e['expected']!r} -> {e['found']!r}"
        for e in moved)
    stub = f"""  - date: "{datetime.date.today().isoformat()}"
    entries: [{', '.join(e['id'] for e in moved)}]
    change: >
      Expected values adopted automatically by scripts/after_seeds.sh after
      v3 and v4 were trained at seeds 3 to 7, taking those rungs from three
      completed seeds to eight.{rows}
    reason: >
      NOT YET REVIEWED BY A HUMAN. These values moved because the ladder
      gained seeds, which is new evidence rather than drift, and the script
      adopted them so that the rebuild could proceed unattended. Adopting a
      value is not the same as agreeing with it. Before this build is
      circulated, someone must confirm that each change is the expected
      consequence of the added seeds and replace this paragraph with that
      confirmation.
    measured_values_unchanged: false
    reviewed: false
"""
    i = s.index("amendments:")
    j = s.index("\n", i) + 1
    reg.write_text(s[:j] + stub + s[j:], encoding="utf8", newline="\n")
    print(f"amendment stub written for {len(moved)} entry/entries, "
          f"marked reviewed: false")
PYEOF

# ------------------------------------------------- 4. rebuild and re-check
step "full rebuild" "$PY" scripts/build_all.py
BUILD_RC=$?

# ----------------------------------------------------- 5. resume the sweep
say ""
say "=== resuming the data-regime sweep $(date +%H:%M:%S) ================="
say "It skips any cell already marked complete; see docs/SWEEP_PAUSED.md."
nohup "$PY" -u scripts/run_sweep.py >> logs_sweep_restart.log 2>&1 &
say "sweep restarted, pid $!"

# ------------------------------------------------------------- 6. report
say ""
say "====================================================================="
say "after_seeds finished $(date)"
say "  build exit code        $BUILD_RC"
say "  drift before adopting  results/drift_check_before_new_seeds.json"
say "  amendment              configs/tolerances.yaml, marked reviewed: false"
say "  sweep                  resumed in the background"
say ""
say "NOT DONE, and needing a person:"
say "  - confirm the adopted expected values are the expected consequence"
say "    of the added seeds, then mark the amendment reviewed"
say "  - the paper still has no author name, email or target venue"
if [ "$BUILD_RC" -ne 0 ]; then
  say "  - THE BUILD DID NOT PASS ITS OWN CHECKS. Read the section above"
  say "    before circulating anything from it."
fi
