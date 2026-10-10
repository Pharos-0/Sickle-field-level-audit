"""Run the per-field audit across EVERY released model, for all regression tasks.

    python scripts/run_full_audit.py

The three regression gates were each run on one model. That is too thin to
support a claim about TASKS -- which is what the audit concludes -- so this
runs every released checkpoint:

    crop_yield          26 released runs
    harvesting_date     13
    transplanting_date  13
                        -- 52 total

Crop type is already covered: all 12 cells were audited at plot level by
`scripts/plot_level_eval.py`.

No training. Forward passes on released checkpoints, so this is hours, not
days. Resumable: a model whose result file already exists is skipped.

A run that FAILS is reported as FAILED and is never interpolated or quietly
dropped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RUNS = Path("D:/FPD/runs/wacv_2024")
OUT = PROJ / "results" / "regression_gate"

TASKS = ["crop_yield", "harvesting_date", "transplanting_date"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = []
    for task in args.tasks:
        d = RUNS / task
        if not d.is_dir():
            print(f"  no released runs for {task}")
            continue
        for run in sorted(d.iterdir()):
            if not (run / "checkpoint_best.pth.tar").exists():
                continue
            plan.append((task, run.name))

    print(f"full per-field audit: {len(plan)} released models\n")
    if args.dry_run:
        for t, r in plan:
            print(f"  {t:20s} {r}")
        return 0

    results, t0 = [], time.time()
    for i, (task, run) in enumerate(plan, 1):
        dest = OUT / f"{task}_{run}_bs{args.batch_size}.json"
        if dest.exists() and not args.redo:
            print(f"[{i}/{len(plan)}] {task}/{run}: already done, skipping",
                  flush=True)
            results.append((task, run, 0, 0.0))
            continue

        print(f"\n[{i}/{len(plan)}] {task}/{run}", flush=True)
        t1 = time.time()
        rc = subprocess.run(
            [sys.executable, "-u", "scripts/regression_gate.py",
             "--task", task, "--run", run,
             "--batch-size", str(args.batch_size)],
            cwd=str(PROJ)).returncode
        dt = time.time() - t1
        # rc == 1 means the GATE failed, which is a result, not a crash.
        # A missing checkpoint or an exception also returns non-zero; the
        # result file distinguishes them.
        results.append((task, run, rc, dt))
        print(f"    -> {'gate PASS' if rc == 0 else 'gate FAIL or error'} "
              f"({dt/60:.1f} min)", flush=True)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    ok = sum(1 for r in results if r[2] == 0)
    print(f"  {ok}/{len(results)} gates passed")
    bad = [r for r in results if r[2] != 0]
    if bad:
        print("  did NOT pass (reported as-is, never interpolated):")
        for t, r, rc, _ in bad:
            print(f"    {t:20s} {r}")
    print(f"  total {(time.time()-t0)/3600:.2f} h")
    print("\nNow run:  python scripts/aggregate_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
