"""PHASE 15 — the data-regime sweep. 90 runs.

    python scripts/run_sweep.py

Tests the study's PRE-REGISTERED hypothesis, stated before any of this ran:

    If temporal complexity requires independent spatial diversity to
    generalise, its advantage over single-date modelling should increase with
    plot count. If it does not, the data-quantity explanation is weakened.

Design, from docs/SICKLE_MASTER_PROMPT.md:

    6 plot counts  x  3 configurations  x  5 seeds  =  90 runs
    counts : 50, 100, 150, 200, 250, 315   (train split has 315 plots)
    configs: v1 single-acquisition, v2 temporal, v5 multi-sensor
    eval   : the SAME 36 validation plots every time, never subsetted

NAMING TRAP. The master prompt calls these configurations "V2, V6, V7". That
numbering predates the v1..v10 ladder and does NOT correspond to it. Translated
into the ladder's configs:

    prompt "V2" single-image   -> v1   (single acquisition, zero temporal info)
    prompt "V6" temporal       -> v2   (full temporal sequence)
    prompt "V7" multi-sensor   -> v5   (S2 + S1 radar)

v5 rather than v6 for multi-sensor: the ladder measured v5->v6 (adding
Landsat-8) as INCONCLUSIVE at three seeds (+0.0075 pixel, -0.0270 plot), so v5
is where the multi-sensor effect actually lives and it costs less to run.

Subsets are NESTED -- the 100-plot set contains the 50-plot set -- so the curve
is not confounded by which plots were drawn. Paddy fraction stays 0.385-0.426
across all counts, so it is not confounded by class balance drift either.

Resumable: a run whose best.json says complete=true is skipped. Sequential, for
the same reason the ladder is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

COUNTS = [50, 100, 150, 200, 250, 315]
CONFIGS = ["v1", "v2", "v5"]
SEEDS = [1, 2, 3, 4, 5]          # subset manifests are seed001..seed005


def subset_name(count: int, seed: int) -> str:
    return f"plots_{count:03d}_seed{seed:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", nargs="+", type=int, default=COUNTS)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--redo-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Fail before burning two days if a manifest is missing.
    missing = [subset_name(c, s) for c in args.counts for s in args.seeds
               if not (PROJ / "artifacts" / "subsets"
                       / f"{subset_name(c, s)}.json").exists()]
    if missing:
        print(f"MISSING {len(missing)} subset manifest(s), e.g. {missing[:3]}")
        print("Run: python scripts/make_subsets.py")
        return 1

    plan = [(c, cfg, s) for s in args.seeds
            for c in args.counts for cfg in args.configs]
    print(f"data-regime sweep: {len(plan)} runs "
          f"({len(args.counts)} counts x {len(args.configs)} configs "
          f"x {len(args.seeds)} seeds)\n")

    results, t_all = [], time.time()
    for count, cfg, seed in plan:
        sub = subset_name(count, seed)
        out = (PROJ / "outputs" / "crop_type" / "sweep" / sub
               / f"{cfg}_seed{seed}" / "best.json")
        if not args.redo_complete and out.exists():
            try:
                if json.loads(out.read_text(encoding="utf8")).get("complete"):
                    print(f"--> {cfg} {sub}: already COMPLETE, skipping",
                          flush=True)
                    results.append((cfg, sub, seed, 0, 0.0))
                    continue
            except json.JSONDecodeError:
                pass

        cmd = [sys.executable, "-u", "-m", "src.train", "--config", cfg,
               "--seed", str(seed), "--subset", sub]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        if args.dry_run:
            print("  " + " ".join(cmd[2:]))
            continue

        print("=" * 72, flush=True)
        print(f"{cfg}  {count} plots  seed {seed}   "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("=" * 72, flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(PROJ)).returncode
        dt = time.time() - t0
        results.append((cfg, sub, seed, rc, dt))
        print(f"\n--> {cfg} {sub}: "
              f"{'OK' if rc == 0 else f'FAILED (exit {rc})'} "
              f"in {dt/60:.1f} min\n", flush=True)

    if args.dry_run:
        print(f"\n{len(plan)} runs planned")
        return 0

    print("=" * 72)
    print("SWEEP SUMMARY")
    bad = [r for r in results if r[3] != 0]
    for cfg, sub, seed, rc, dt in results:
        print(f"  {cfg:4s} {sub:22s} "
              f"{'OK    ' if rc == 0 else 'FAILED'} {dt/60:7.1f} min")
    print(f"\n{len(results)-len(bad)}/{len(results)} succeeded, "
          f"{len(bad)} FAILED, total {(time.time()-t_all)/3600:.1f} h")
    if bad:
        print("Failed runs must be reported as FAILED, not interpolated.")
    print("\nNow run:  python scripts/aggregate_sweep.py")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
