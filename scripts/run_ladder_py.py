"""Run the ladder sequentially. Cross-shell replacement for run_ladder.sh.

    python scripts/run_ladder_py.py                      # v1..v10, seed 0
    python scripts/run_ladder_py.py --seeds 0 1 2
    python scripts/run_ladder_py.py --rungs v1 v2 --epochs 10

STRICTLY SEQUENTIAL, deliberately. Two multi-hour CPU jobs halve each other's
throughput, and the first reproduction-gate sweep in this project had to be
thrown away because concurrent loops overwrote each other's output
(docs/REPRODUCTION_GATE.md section 4a).

A rung that fails does NOT stop the ladder -- the remaining rungs still run and
the failure is reported at the end. A failed rung is reported as FAILED and is
never interpolated or quietly skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
LADDER = [f"v{i}" for i in range(1, 11)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=LADDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--redo-complete", action="store_true",
                    help="re-run rungs that already completed (default: skip)")
    args = ap.parse_args()

    results = []
    t_all = time.time()
    # SEED-MAJOR, deliberately: finish the whole ladder at seed 0 before
    # starting seed 1. If this is interrupted -- and it runs for days -- what
    # survives is a COMPLETE ladder at fewer seeds, which is analysable, rather
    # than three seeds of the first few rungs, which is not.
    for seed in args.seeds:
        for rung in args.rungs:
            # RESUMABLE. A multi-day ladder will be interrupted -- by a reboot,
            # a sleep, or a stray kill -- and redoing finished rungs can cost
            # tens of hours. A rung counts as done only when its best.json says
            # complete=true, which the trainer writes after the final epoch.
            done = PROJ / "outputs" / "crop_type" / f"{rung}_seed{seed}" / "best.json"
            if not args.redo_complete and done.exists():
                try:
                    if json.loads(done.read_text(encoding="utf8")).get("complete"):
                        print(f"--> {rung} seed {seed}: already COMPLETE, skipping",
                              flush=True)
                        results.append((rung, seed, 0, 0.0))
                        continue
                except json.JSONDecodeError:
                    pass   # unreadable means not trustworthy; re-run it

            cmd = [sys.executable, "-u", "-m", "src.train",
                   "--config", rung, "--seed", str(seed)]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.num_workers is not None:
                cmd += ["--num-workers", str(args.num_workers)]

            print("=" * 72, flush=True)
            print(f"{rung} seed {seed}   {time.strftime('%Y-%m-%d %H:%M:%S')}",
                  flush=True)
            print("=" * 72, flush=True)

            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(PROJ)).returncode
            dt = time.time() - t0
            results.append((rung, seed, rc, dt))
            print(f"\n--> {rung} seed {seed}: "
                  f"{'OK' if rc == 0 else f'FAILED (exit {rc})'} "
                  f"in {dt/60:.1f} min\n", flush=True)

    print("=" * 72)
    print("LADDER SUMMARY")
    print("=" * 72)
    for rung, seed, rc, dt in results:
        print(f"  {rung:5s} seed {seed}  "
              f"{'OK    ' if rc == 0 else 'FAILED'}  {dt/60:7.1f} min")
    bad = [r for r in results if r[2] != 0]
    print(f"\n{len(results) - len(bad)}/{len(results)} succeeded, "
          f"{len(bad)} FAILED, total {(time.time()-t_all)/3600:.1f} h")
    if bad:
        print("Failed rungs must be reported as FAILED, not interpolated.")
    print("\nNow run:  python scripts/aggregate_ladder.py")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
