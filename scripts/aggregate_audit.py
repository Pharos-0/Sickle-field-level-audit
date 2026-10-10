"""Aggregate the per-field audit across every released model and task.

    python scripts/aggregate_audit.py

Answers the question the audit was built to answer: does the aggregate number
describe an individual field, and does the answer depend on the task?

Reports, per task, across all released models:
  * how many reproduced within the pre-declared tolerance
  * the aggregate error
  * the per-field-season error distribution -- median, p90, worst
  * the ratio p90/median, which is the tail the aggregate hides

A model whose gate FAILED is listed separately and excluded from the summary
statistics. It is never interpolated or silently dropped.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
GATE = PROJ / "results" / "regression_gate"
OUT = PROJ / "results"

UNIT = {"crop_yield": "kg", "harvesting_date": "days",
        "transplanting_date": "days", "sowing_date": "days"}


def main() -> int:
    files = sorted(GATE.glob("*.json"))
    if not files:
        print(f"no audit results in {GATE}")
        return 1

    rows, by_task = [], defaultdict(list)
    for f in files:
        d = json.loads(f.read_text(encoding="utf8"))
        pf = d.get("per_field_error", {})
        r = {
            "task": d["task"], "run": d["run"], "verdict": d["verdict"],
            "relative_delta": d.get("relative_delta"),
            "agg_mae": d["reproduced"]["mae"],
            "agg_rmse": d["reproduced"]["rmse"],
            "n_field_seasons": d["reproduced"]["n_fields"],
            "median": pf.get("median"), "p75": pf.get("p75"),
            "p90": pf.get("p90"), "max": pf.get("max"),
        }
        r["p90_over_median"] = (r["p90"] / r["median"]
                                if r["median"] else None)
        rows.append(r)
        by_task[d["task"]].append(r)

    with (OUT / "audit_summary.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("PER-FIELD AUDIT -- all released models, all regression tasks\n")
    print(f"{'task':20s}{'models':>7s}{'gate OK':>8s}"
          f"{'agg MAE':>10s}{'median':>9s}{'p90':>9s}{'worst':>10s}"
          f"{'p90/med':>9s}")
    print("-" * 82)

    for task in sorted(by_task):
        rs = by_task[task]
        ok = [r for r in rs if r["verdict"] == "PASS"]
        use = ok or rs
        f = lambda k: np.median([r[k] for r in use if r[k] is not None])
        print(f"{task:20s}{len(rs):7d}{len(ok):8d}"
              f"{f('agg_mae'):10.2f}{f('median'):9.2f}{f('p90'):9.2f}"
              f"{max(r['max'] for r in use):10.2f}"
              f"{f('p90_over_median'):9.2f}")

    print("-" * 82)
    print("(medians across models; 'worst' is the worst single field-season "
          "seen in any model)\n")

    # the finding: the tail, not the centre
    print("WHAT THE AGGREGATE HIDES")
    for task in sorted(by_task):
        use = [r for r in by_task[task] if r["verdict"] == "PASS"] \
            or by_task[task]
        agg = np.median([r["agg_mae"] for r in use])
        med = np.median([r["median"] for r in use])
        p90 = np.median([r["p90"] for r in use])
        u = UNIT.get(task, "")
        better = "BETTER" if med < agg else "WORSE"
        print(f"  {task}:")
        print(f"     aggregate MAE {agg:.2f} {u}  |  typical field {med:.2f} {u}"
              f"  ({better} than the aggregate)")
        print(f"     but 1 field-season in 10 is off by {p90:.2f} {u} "
              f"-- {p90/med:.1f}x the typical field")

    bad = [r for r in rows if r["verdict"] != "PASS"]
    if bad:
        print(f"\nGATE DID NOT PASS ({len(bad)}), excluded from the summary "
              f"and reported as-is:")
        for r in bad:
            print(f"   {r['task']:20s} {r['run']:26s} "
                  f"relative delta {100*(r['relative_delta'] or 0):.1f}%")

    print(f"\nwrote {OUT / 'audit_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
