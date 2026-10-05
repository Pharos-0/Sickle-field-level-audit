"""Aggregate the reproduction-gate results into one table.

Reads every `results/gate/*.json` written by `scripts/phase5_gate.py` and
reports, per released cell, the released value, the reproduced value, the
absolute difference and the verdict against the pre-declared tolerance.

    python scripts/aggregate_gate.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

GATE = Path(__file__).resolve().parents[1] / "results" / "gate"
OUT = Path(__file__).resolve().parents[1] / "results"

SENSOR_ORDER = ["S2", "S1", "L8", "S2,L8,S1"]
ARCH_ORDER = ["unet3d", "utae", "convlstm"]


def main() -> int:
    files = sorted(GATE.glob("gate_*.json"))
    if not files:
        print(f"no gate results in {GATE}")
        return 1

    recs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf8"))
        run = d["run"]
        sensors = run.split("]")[0].lstrip("[")
        arch = run.split("_", 1)[1] if "_" in run else run
        recs.append({
            "task": d["task"], "run": run, "sensors": sensors, "arch": arch,
            "batch_size": d.get("batch_size"),
            "released": d["released_best_val_iou"],
            "released_epoch": d["released_best_epoch"],
            "reproduced": d["reproduced"],
            "abs_delta": d["abs_delta"],
            "tolerance": d["tolerance"],
            "verdict": d["verdict"],
            "paddy_iou": d["all_metrics"].get("paddy_iou"),
            "non_paddy_iou": d["all_metrics"].get("non_paddy_iou"),
            "macro_f1": d["all_metrics"].get("f1_macro"),
        })

    def key(r):
        s = SENSOR_ORDER.index(r["sensors"]) if r["sensors"] in SENSOR_ORDER else 9
        a = ARCH_ORDER.index(r["arch"]) if r["arch"] in ARCH_ORDER else 9
        return (a, s)

    recs.sort(key=key)

    with (OUT / "gate_summary.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    print(f"{'run':24s}{'ep':>4s}{'released':>10s}{'repro':>9s}"
          f"{'|delta|':>9s}{'verdict':>8s}{'paddy':>8s}{'non-pad':>9s}")
    print("-" * 81)
    for r in recs:
        rep = r["reproduced"]
        print(f"{r['run']:24s}{r['released_epoch']:4d}{r['released']:10.4f}"
              + (f"{rep:9.4f}" if rep is not None else f"{'--':>9s}")
              + (f"{r['abs_delta']:9.4f}" if r["abs_delta"] is not None
                 else f"{'--':>9s}")
              + f"{r['verdict']:>8s}"
              + (f"{r['paddy_iou']:8.3f}" if r["paddy_iou"] is not None
                 else f"{'--':>8s}")
              + (f"{r['non_paddy_iou']:9.3f}" if r["non_paddy_iou"] is not None
                 else f"{'--':>9s}"))

    deltas = np.array([r["abs_delta"] for r in recs
                       if r["abs_delta"] is not None], dtype=float)
    npass = sum(1 for r in recs if r["verdict"] == "PASS")
    print("-" * 81)
    print(f"cells reproduced : {len(recs)}")
    print(f"PASS at +/-{recs[0]['tolerance']}  : {npass}/{len(recs)}")
    if deltas.size:
        print(f"abs delta        : median {np.median(deltas):.4f}  "
              f"max {deltas.max():.4f}  min {deltas.min():.4f}")

    # degenerate runs: a model stuck at a single class
    degen = [r for r in recs if r["paddy_iou"] is not None
             and r["paddy_iou"] < 1e-6]
    if degen:
        print(f"\nDEGENERATE runs (paddy IoU == 0, i.e. one class predicted "
              f"everywhere): {len(degen)}")
        for r in degen:
            print(f"   {r['run']:24s} best epoch {r['released_epoch']:3d}  "
                  f"IoU {r['released']:.4f}")

    print(f"\nwrote {OUT / 'gate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
