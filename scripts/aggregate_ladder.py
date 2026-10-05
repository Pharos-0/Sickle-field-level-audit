"""Aggregate the ladder into the table the write-up needs.

Reads every `outputs/<task>/v*_seed*/best.json`, reports each rung under both
metric families, and computes the rung-to-rung deltas that are the actual
contribution of this study.

Honesty rules enforced here rather than left to the writer:

  * A rung with no run on disk prints **NOT RUN**. It is never interpolated,
    never dropped, and never silently omitted from the table.
  * A rung whose paddy IoU is 0 prints **DEGENERATE** and is excluded from the
    delta analysis, because a model predicting one class everywhere is not a
    point on a performance curve.
  * A delta smaller than the seed-to-seed spread prints **inconclusive**. With
    three seeds the spread is the only honest yardstick for whether a rung did
    anything.

    python scripts/aggregate_ladder.py
    python scripts/aggregate_ladder.py --task crop_type
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

LADDER = [f"v{i}" for i in range(1, 11)]
SKIPPED_SMOKE: list[str] = []


def collect(task: str) -> dict[str, dict[int, dict]]:
    base = PROJ / "outputs" / task
    found: dict[str, dict[int, dict]] = {v: {} for v in LADDER}
    if not base.exists():
        return found
    for d in sorted(base.glob("v*_seed*")):
        best = d / "best.json"
        if not best.exists():
            continue
        v, _, s = d.name.partition("_seed")
        if v not in found:
            continue
        try:
            rec = json.loads(best.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            continue
        man = {}
        mp = d / "manifest.json"
        if mp.exists():
            try:
                man = json.loads(mp.read_text(encoding="utf8"))
            except json.JSONDecodeError:
                pass
        # A smoke run trains on two batches to prove the pipeline moves. Its
        # metrics are meaningless and must never reach a results table. The
        # trainer records smoke=True in the manifest; honour it.
        if man.get("smoke"):
            SKIPPED_SMOKE.append(d.name)
            continue
        rec["_manifest"] = man
        found[v][int(s)] = rec
    return found


def agg(runs: dict[int, dict], key: str):
    vals = [r[key] for r in runs.values() if key in r and r[key] is not None]
    if not vals:
        return None, None, 0
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return st.mean(vals), sd, len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="crop_type")
    args = ap.parse_args()

    data = collect(args.task)
    rows = []
    if SKIPPED_SMOKE:
        names = ", ".join(SKIPPED_SMOKE)
        print(f"EXCLUDED {len(SKIPPED_SMOKE)} smoke run(s) -- two-batch "
              f"pipeline checks, not results: {names}\n")

    print(f"LADDER -- task {args.task}\n")
    print(f"{'rung':6s}{'seeds':>6s}{'pixel macro IoU':>20s}"
          f"{'plot macro IoU':>20s}{'paddy IoU':>13s}  status")
    print("-" * 80)

    for v in LADDER:
        runs = data[v]
        if not runs:
            print(f"{v:6s}{'--':>6s}{'--':>20s}{'--':>20s}{'--':>13s}  NOT RUN")
            rows.append({"rung": v, "status": "NOT RUN"})
            continue

        pm, psd, n = agg(runs, "macro_iou")
        gm, gsd, _ = agg(runs, "plot_macro_iou")
        dm, _, _ = agg(runs, "paddy_iou")

        status = "ok"
        # best.json is rewritten whenever the best improves, so a killed run
        # still leaves a usable record -- but it is NOT a finished result and
        # must not be compared against completed rungs without saying so.
        partial = [s for s, r in runs.items() if r.get("complete") is False]
        if partial:
            got = max((r.get("epochs_run_so_far") or 0) for r in runs.values())
            plan = max((r.get("epochs_planned") or 0) for r in runs.values())
            status = f"INCOMPLETE ({got}/{plan} epochs, seeds {sorted(partial)})"
        if dm is not None and dm < 1e-6:
            status = "DEGENERATE (predicts one class)"
        # "best epoch 1" means a run never improved past its first epoch --
        # how three of the four released UTAE baselines failed. It is only
        # meaningful once the run has FINISHED: a run currently on epoch 1 of
        # 100 trivially has its best at epoch 1 and is not degenerate.
        finished = [r for r in runs.values() if r.get("complete")]
        eps = [r.get("best_epoch") for r in finished]
        if eps and all(e == 1 for e in eps if e is not None):
            status = ("DEGENERATE (best epoch 1)" if status == "ok"
                      else status + "; best epoch 1")

        print(f"{v:6s}{n:6d}"
              + (f"{pm:14.4f}+-{psd:.3f}" if pm is not None else f"{'--':>20s}")
              + (f"{gm:14.4f}+-{gsd:.3f}" if gm is not None else f"{'--':>20s}")
              + (f"{dm:13.4f}" if dm is not None else f"{'--':>13s}")
              + f"  {status}")
        rows.append({"rung": v, "status": status, "n_seeds": n,
                     "pixel_macro_iou": pm, "pixel_sd": psd,
                     "plot_macro_iou": gm, "plot_sd": gsd, "paddy_iou": dm})

    # --- rung-to-rung deltas ------------------------------------------------
    print("\nRUNG-TO-RUNG DELTAS (what each added component bought)")
    print("Controlled component attribution WITHIN this experimental system.")
    print("Not causal. Not a claim about these components in general.\n")
    print(f"{'step':10s}{'d pixel':>12s}{'d plot':>12s}  verdict")
    print("-" * 62)

    by = {r["rung"]: r for r in rows}
    for a, b in zip(LADDER, LADDER[1:]):
        ra, rb = by.get(a, {}), by.get(b, {})
        if ra.get("status") == "NOT RUN" or rb.get("status") == "NOT RUN":
            print(f"{a}->{b:6s}{'--':>12s}{'--':>12s}  NOT RUN")
            continue
        bad = [n for n, r in ((a, ra), (b, rb))
               if "DEGENERATE" in str(r.get("status"))
               or "INCOMPLETE" in str(r.get("status"))]
        if bad:
            why = ("degenerate" if any("DEGENERATE" in str(r.get("status"))
                                       for r in (ra, rb)) else "incomplete")
            print(f"{a}->{b:6s}{'--':>12s}{'--':>12s}  "
                  f"skipped, {', '.join(bad)} {why}")
            continue
        dp = (rb.get("pixel_macro_iou") or 0) - (ra.get("pixel_macro_iou") or 0)
        dg = ((rb.get("plot_macro_iou") or 0) - (ra.get("plot_macro_iou") or 0)
              if rb.get("plot_macro_iou") is not None
              and ra.get("plot_macro_iou") is not None else None)
        spread = max(ra.get("pixel_sd") or 0, rb.get("pixel_sd") or 0)
        if spread > 0 and abs(dp) < spread:
            verdict = "inconclusive: smaller than seed-to-seed spread"
        elif dp > 0:
            verdict = "positive"
        else:
            verdict = "negative or flat"
        print(f"{a}->{b:6s}{dp:+12.4f}"
              + (f"{dg:+12.4f}" if dg is not None else f"{'--':>12s}")
              + f"  {verdict}")
        if dg is not None and dp * dg < 0:
            print(f"{'':10s}{'':12s}{'':12s}  "
                  f"NOTE: the two metrics move in OPPOSITE directions here")

    out = PROJ / "results" / f"ladder_{args.task}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")

    n_run = sum(1 for r in rows if r.get("status") != "NOT RUN")
    print(f"\n{n_run}/{len(LADDER)} rungs have results. "
          f"{len(LADDER) - n_run} NOT RUN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
