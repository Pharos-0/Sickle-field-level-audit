"""Test whether batch composition explains the three crop-type gate misses.

    python scripts/diagnose_gate_misses.py --device cpu

The manuscript reports that three of twelve crop-type cells miss the declared
0.01 IoU tolerance, by 0.0114, 0.0123 and 0.0345, and says the cause is NOT
established -- only that it is consistent with the batch-size sensitivity of
Section V-A. This script turns that from a hypothesis into a measurement.

THE EXPERIMENT
--------------
Defect 1 says evaluation depends on batch size, because the temporal axis is
padded to the per-batch maximum and the pad mask is realigned by truncation.
If that is also what produces these three residuals, then sweeping the
evaluation batch size should move each cell's score, and for at least some
cell some batch size should land within tolerance of the released value.

So for each failing cell we re-evaluate the released checkpoint at several
batch sizes and report the distance to the released number at each.

HOW TO READ THE RESULT
----------------------
  * If a batch size brings a cell within tolerance, batch composition
    explains that cell, and the released number was produced at a different
    effective batching than ours.
  * If the score moves with batch size but never reaches the released value,
    batch composition contributes but does not account for the gap.
  * If the score barely moves, batch composition is NOT the explanation for
    that cell and something else differs -- in which case the manuscript's
    "candidate, not a diagnosis" wording is the correct thing to keep.

A negative result here is a real result. Do not discard it.

This writes results/gate/diagnosis_misses.json and prints a table. It never
edits the manuscript.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
GATE = PROJ / "results" / "gate"
PY = sys.executable

TOL = 0.01


def failing_cells() -> list[dict]:
    """The crop-type cells whose bs32 gate did not pass, read from disk."""
    out = []
    for f in sorted(GATE.glob("gate_crop_type_*_bs32.json")):
        g = json.loads(f.read_text(encoding="utf8"))
        if g["verdict"] != "PASS":
            out.append({"run": g["run"],
                        "released": g["released_best_val_iou"],
                        "bs32": g["reproduced"],
                        "delta_bs32": g["abs_delta"]})
    return out


def evaluate(run: str, bs: int, device: str,
             reuse: bool = True, cached_only: bool = False) -> float | None:
    """Return the cell's IoU at this batch size, running the gate if needed.

    phase5_gate exits 1 whenever the verdict is FAIL, which for the cells
    being diagnosed here is the expected outcome, not an error. An earlier
    version of this function read that exit code as a crash and discarded
    valid results, then reported a spread of 0.0000 and concluded that the
    score does not move with batch size -- for two cells whose scores move
    by 0.18 and 0.33 IoU. Success is therefore decided by the artefact on
    disk, never by the return code.
    """
    f = GATE / f"gate_crop_type_{run}_bs{bs}.json"
    if reuse and f.exists():
        return json.loads(f.read_text(encoding="utf8"))["reproduced"]
    if cached_only:
        print(f"    bs {bs}: no cached artefact, and --cached-only is set")
        return None
    before = f.stat().st_mtime if f.exists() else None
    p = subprocess.run(
        [PY, "scripts/phase5_gate.py", "--run", run, "--task", "crop_type",
         "--batch-size", str(bs), "--device", device],
        cwd=PROJ, capture_output=True, text=True)
    if not f.exists() or (before is not None and f.stat().st_mtime == before):
        print(f"    bs {bs}: no fresh artefact (gate exited {p.returncode})")
        tail = (p.stderr or p.stdout).rstrip().split("\n")
        print("   ", tail[-1][:160] if tail else "(no output)")
        return None
    return json.loads(f.read_text(encoding="utf8"))["reproduced"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached-only", action="store_true",
                    help=("build the report from artefacts already on disk "
                          "and launch nothing; use when the remaining "
                          "evaluations are too slow to be worth repeating"))
    ap.add_argument("--no-reuse", action="store_true",
                    help="recompute even where an artefact exists")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps this off the GPU if training is running")
    ap.add_argument("--batch-sizes", nargs="+", type=int,
                    default=[8, 16, 64],
                    help=("default omits 1 and 4: on the first cell both "
                          "moved the score far AWAY from the released value "
                          "and each cost ~45 min of CPU on ConvLSTM, so they "
                          "cannot answer whether any batch size closes the "
                          "gap"))
    args = ap.parse_args()

    cells = failing_cells()
    if not cells:
        print("No failing crop-type cells found. Nothing to diagnose.")
        return 0
    print(f"{len(cells)} crop-type cells missed the {TOL} tolerance at bs32.")
    print(f"Sweeping batch sizes {args.batch_sizes} on {args.device}.\n")

    report = {"tolerance": TOL, "batch_sizes": args.batch_sizes, "cells": []}
    for c in cells:
        print(f"{c['run']}: released {c['released']:.4f}, "
              f"bs32 {c['bs32']:.4f} (off by {c['delta_bs32']:.4f})")
        scores = {32: c["bs32"]}
        for bs in args.batch_sizes:
            v = evaluate(c["run"], bs, args.device,
                         reuse=not args.no_reuse,
                         cached_only=args.cached_only)
            if v is not None:
                scores[bs] = v
                d = abs(v - c["released"])
                flag = "  <-- WITHIN TOLERANCE" if d <= TOL else ""
                print(f"    bs {bs:<3d} {v:.4f}   off by {d:.4f}{flag}")
        best = min(scores.items(), key=lambda kv: abs(kv[1] - c["released"]))
        spread = max(scores.values()) - min(scores.values())
        verdict = ("batch composition explains this cell"
                   if abs(best[1] - c["released"]) <= TOL else
                   "batch size moves the score but does NOT close the gap"
                   if spread > TOL else
                   "score barely moves with batch size; another cause")
        print(f"    spread across batch sizes {spread:.4f} | {verdict}\n")
        report["cells"].append({
            "run": c["run"], "released": c["released"],
            "scores_by_batch_size": scores,
            "best_batch_size": best[0],
            "best_abs_delta": abs(best[1] - c["released"]),
            "spread": spread, "verdict": verdict})

    out = GATE / "diagnosis_misses.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"wrote {out}")
    print("\nWhatever this shows, report it. A result that fails to explain "
          "the residuals is evidence about the residuals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
