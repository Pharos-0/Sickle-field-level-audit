"""The data-regime sweep, analysed exactly as configs/sweep_analysis_protocol.yaml
says (frozen 2026-10-14, before any cross-run result was computed).

    python scripts/sweep_rescore.py     # first: field IoU for every run
    python scripts/sweep_analysis.py

Runs the protocol's validity checks and REFUSES to report anything while one
fails. Then, for the primary metric (pixel-pooled macro IoU, best.json) and
the secondary metric (field IoU, mean over the 36 validation fields):
  - the curves: mean and sd over seeds, per configuration and plot count
  - paired seed-wise deltas: temporal = v2 - v1, radar = v2_radar - v2
  - per-count summaries with 95 % t-intervals (n = 5) and the master prompt's
    inconclusive rule (|mean| < sd)
  - the trend test that decides the hypothesis: per-seed OLS slope of the
    temporal advantage on log2(plot count), mean and 95 % t-interval
    (df = 4); sensitivity on plots/100 and the paired 315-minus-50 change.

Writes results/deltas.csv (master prompt format) and
results/sweep/sweep_analysis.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
SWEEP = PROJ / "outputs" / "crop_type" / "sweep"
RES = PROJ / "results" / "sweep"
PROTO = yaml.safe_load((PROJ / "configs" / "sweep_analysis_protocol.yaml")
                       .read_text(encoding="utf8"))
COUNTS = PROTO["design"]["plot_counts"]
CONFIGS = list(PROTO["design"]["configs"])
SEEDS = PROTO["design"]["seeds"]
METRICS = {"primary": "pooled_macro_iou_training",
           "secondary": "field_iou_mean"}


def tci(x):
    """Mean, sd and 95 % t-interval of a small sample."""
    x = np.asarray(x, float)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "n": len(x),
            "inconclusive": abs(m) < sd}


def checks(rows):
    out, ok = [], True

    def rec(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        out.append({"check": name, "pass": bool(passed), "detail": detail})

    grid = {(r["plot_count"], r["config"], r["seed"]) for r in rows}
    want = {(c, k, s) for c in COUNTS for k in CONFIGS for s in SEEDS}
    rec("all 90 runs complete and re-scored", grid == want and len(rows) == 90,
        f"{len(rows)} rows; missing {sorted(want - grid)}")

    bad = []
    for r in rows:
        d = SWEEP / r["subset"] / r["run"]
        cfg = json.loads((d / "config_resolved.json").read_text(encoding="utf8"))
        man = json.loads((d / "manifest.json").read_text(encoding="utf8"))
        if (cfg["data"].get("subset") != r["subset"]
                or cfg["project"].get("version") != r["config"]
                or int(man.get("seed")) != r["seed"]):
            bad.append(f"{r['subset']}/{r['run']}")
    rec("configs name the expected subset, version and seed", not bad,
        f"{len(rows) - len(bad)}/{len(rows)} match" + (f"; bad {bad}" if bad else ""))

    val = set()
    root = cfg["data"]["root"]
    with open(f"{root}/sickle_dataset_tabular.csv", encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            if r["SPLIT"] == cfg["data"]["eval_split"] and r["PLOT_ID"]:
                val.add(int(float(r["PLOT_ID"])))
    nest_bad, leak = [], []
    for s in SEEDS:
        prev = set()
        for c in COUNTS:
            ids = set(int(float(p)) for p in json.loads(
                (PROJ / "artifacts" / "subsets" / f"plots_{c:03d}_seed{s:03d}.json")
                .read_text(encoding="utf8"))["plot_ids"])
            if len(ids) != c or not prev <= ids:
                nest_bad.append(f"seed {s} count {c}")
            if ids & val:
                leak.append(f"seed {s} count {c}")
            prev = ids
    rec("subsets nested within each seed, sized as named", not nest_bad,
        "all nested" if not nest_bad else f"broken: {nest_bad}")
    rec("no subset shares a plot with the validation split", not leak,
        f"{len(val)} validation plots; overlap in {leak}" if leak
        else f"{len(val)} validation plots, no overlap")

    inv = {}
    with open(PROJ / "docs" / "CHECKPOINT_INVENTORY.csv", encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            inv[r["file"]] = r["sha256"]
    miss = [f"{r['subset']}/{r['run']}" for r in rows
            if not inv.get(f"outputs/crop_type/sweep/{r['subset']}/{r['run']}"
                           f"/checkpoint_best.pth.tar")]
    rec("every checkpoint present and checksummed in the inventory", not miss,
        f"{len(rows) - len(miss)}/{len(rows)}" + (f"; missing {miss}" if miss else ""))

    fails = [f"{r['subset']}/{r['run']} ({r['abs_diff']:.1e})"
             for r in rows if not r["pooled_check"]]
    rec("recomputed pooled macro IoU within 2e-3 of best.json", not fails,
        f"max diff {max(r['abs_diff'] for r in rows):.1e}"
        + (f"; failing {fails}" if fails else ""))

    sets = set()
    for r in rows:
        fc = json.loads((RES / "field_confusions" / f"{r['subset']}__{r['run']}.json")
                        .read_text(encoding="utf8"))
        sets.add(frozenset(fc["fields"]))
    rec("every run scores the same validation fields", len(sets) == 1,
        f"{len(next(iter(sets)))} fields" if len(sets) == 1
        else f"{len(sets)} different field sets")
    return ok, out


def slope(x, y):
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def analyse(rows, key):
    v = {(r["plot_count"], r["config"], r["seed"]): float(r[key]) for r in rows}
    curves = {k: {str(c): tci([v[(c, k, s)] for s in SEEDS]) for c in COUNTS}
              for k in CONFIGS}
    comps = {"temporal_advantage": ("v1", "v2"),
             "radar_increment": ("v2", "v2_radar")}
    res, deltas = {"curves": curves}, []
    for name, (base, var) in comps.items():
        d = {(c, s): v[(c, var, s)] - v[(c, base, s)] for c in COUNTS for s in SEEDS}
        for c in COUNTS:
            for s in SEEDS:
                deltas.append({"task_arm": "A", "plot_count": c, "seed": s,
                               "baseline": base, "variant": var, "metric": key,
                               "baseline_value": v[(c, base, s)],
                               "variant_value": v[(c, var, s)],
                               "delta": d[(c, s)]})
        per_count = {str(c): tci([d[(c, s)] for s in SEEDS]) for c in COUNTS}
        sl = [slope(np.log2(COUNTS), [d[(c, s)] for c in COUNTS]) for s in SEEDS]
        sl100 = [slope(np.array(COUNTS) / 100, [d[(c, s)] for c in COUNTS])
                 for s in SEEDS]
        end = [d[(COUNTS[-1], s)] - d[(COUNTS[0], s)] for s in SEEDS]
        t = tci(sl)
        verdict = ("SUPPORTED" if t["mean"] > 0 and t["ci95"][0] > 0 else
                   "CONTRADICTED" if t["mean"] < 0 and t["ci95"][1] < 0 else
                   "NOT SUPPORTED" + (" (inconclusive)" if t["inconclusive"]
                                      else ""))
        res[name] = {
            "baseline": base, "variant": var, "per_count": per_count,
            "slope_per_doubling": {**t, "per_seed": sl},
            "slope_per_100_plots": {**tci(sl100), "per_seed": sl100},
            "change_315_minus_50": {**tci(end), "per_seed": end},
            "verdict": verdict,
            "role": "decides the hypothesis" if name == "temporal_advantage"
                    else "secondary, exploratory",
        }
    return res, deltas


def main() -> int:
    with open(RES / "rescore.csv", encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("plot_count", "seed"):
            r[k] = int(r[k])
        for k in ("pooled_macro_iou_training", "pooled_macro_iou_recomputed",
                  "abs_diff", "field_iou_mean", "field_iou_median"):
            r[k] = float(r[k])
        r["pooled_check"] = r["pooled_check"] == "True"

    # protocol configurations only; anything else is listed, never analysed
    extra = sorted({(r["subset"], r["run"]) for r in rows
                    if r["config"] not in CONFIGS})
    rows = [r for r in rows if r["config"] in CONFIGS]
    excluded = [f"{s_}/{r_}" for s_, r_ in extra] + [
        str(d.relative_to(SWEEP)).replace("\\", "/")
        for d in sorted(SWEEP.glob("plots_*_seed*/*_seed*"))
        if d.name.rsplit("_seed", 1)[0] not in CONFIGS]
    ok, chk = checks(rows)
    for c in chk:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}: {c['detail']}")
    if not ok:
        print("\nREFUSING TO REPORT: a validity check failed (protocol).")
        (RES / "sweep_analysis.json").write_text(json.dumps(
            {"protocol_version": PROTO["protocol_version"], "checks": chk,
             "reported": False}, indent=2), encoding="utf8")
        return 1

    out = {"protocol_version": PROTO["protocol_version"],
           "protocol_frozen": PROTO["frozen"], "checks": chk, "reported": True,
           "excluded_runs": {
               "runs": sorted(set(excluded)),
               "reason": "config v5, superseded by v2_radar in commit a8aa0dc (64f48cb before the 2026-10-23 history rewrite) "
                         "(2026-10-10) because v5 also carries augmentation "
                         "and class weighting; not in the protocol's design"}}
    all_deltas = []
    for role, key in METRICS.items():
        res, deltas = analyse(rows, key)
        out[role] = {"metric": key, **res}
        all_deltas += deltas
    p, s = (out["primary"]["temporal_advantage"]["verdict"],
            out["secondary"]["temporal_advantage"]["verdict"])
    out["hypothesis_verdict"] = p
    out["metrics_agree"] = p == s
    (RES / "sweep_analysis.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf8")
    with open(PROJ / "results" / "deltas.csv", "w", newline="",
              encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_deltas[0]))
        w.writeheader()
        w.writerows(all_deltas)

    for role in METRICS:
        a = out[role]
        print(f"\n== {role}: {a['metric']}")
        for k in CONFIGS:
            print(f"  {k:<9}" + "  ".join(
                f"{c}:{a['curves'][k][str(c)]['mean']:.3f}" for c in COUNTS))
        for name in ("temporal_advantage", "radar_increment"):
            x = a[name]
            print(f"  {name} ({x['variant']} - {x['baseline']}), {x['role']}")
            print("    per count: " + "  ".join(
                f"{c}:{x['per_count'][str(c)]['mean']:+.3f}" for c in COUNTS))
            sp = x["slope_per_doubling"]
            print(f"    slope per doubling {sp['mean']:+.4f} "
                  f"[{sp['ci95'][0]:+.4f}, {sp['ci95'][1]:+.4f}] -> {x['verdict']}")
    print(f"\nhypothesis verdict (primary metric): {p}; metrics agree: "
          f"{out['metrics_agree']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
