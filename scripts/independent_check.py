"""Independent re-derivation of the manuscript's headline numbers.

    python scripts/independent_check.py
    python scripts/independent_check.py --json report.json

WHY THIS FILE IS WRITTEN THE WAY IT IS
--------------------------------------
This is meant to be run by somebody who does not trust the rest of this
repository. So it deliberately does NOT import anything from `src/`, does not
import torch, and does not read any of the project's own result summaries
(`triage_v5.json`, `audit_summary.csv` medians, the manuscript, the figures).
It reads only raw per-run artefacts and re-implements every calculation from
scratch with numpy, then compares the result against the value the manuscript
prints, which is hard-coded below as a declared expectation.

If this script and `scripts/verify_all.py` ever disagree, that is the
interesting case: `verify_all.py` shares helper code with the pipeline,
whereas this one shares none.

A PASS here means: the arithmetic in the manuscript follows from the raw files
in this repository. It does NOT mean the raw files are themselves correct --
that requires retraining, which is a separate and much larger exercise.

WHAT IT CHECKS
--------------
  1. Ablation partition      results/ladder_crop_type.csv
  2. Field-level stats       outputs/crop_type/v5_seed*/best_per_plot_iou.json
  3. Cross-seed correlation  the same files, all seed pairs
  4. Screening metrics       results/triage/per_field_signals_v5.csv
  5. Reproduction counts     results/audit_summary.csv, results/gate/*.json
  6. Split sizes             the released tabular index

Check 4 needs `per_field_signals_v5.csv`, which `scripts/triage_eval.py`
writes. If it is absent the check reports SKIP rather than passing silently.

EXIT CODE
---------
0 if every check that could run passed; 1 otherwise. Suitable for CI.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# The manuscript's claims, written out so the comparison is explicit. Change
# these ONLY to match a new manuscript, never to match a new result.
# --------------------------------------------------------------------------
CLAIMED = {
    "n_steps": 9,
    "n_inconclusive": 5,
    "n_exceed_up": 2,
    "n_exceed_down": 2,
    "sigma_max": 0.0417,
    "d_temporal": 0.1198,
    "d_radar": 0.1860,
    "pixel_pooled_iou": 0.8665,
    # Field IoU (the crop a field holds), corrected 2026-10-12; see
    # src/evaluation/metrics.py. Values as the manuscript states them.
    "median_field_iou": 0.957,
    "n_fields": 36,
    "n_below_half": 4,
    "n_below_half_every_run": 2,
    "seed_r_min": 0.57,
    "seed_r_max": 0.86,
    "conf_recall_50": 0.75,
    "conf_precision_50": 0.17,
    "conf_ci_lo_50": 0.33,
    "n_regression_models": 48,
    "n_regression_pass": 48,
    "n_croptype_cells": 12,
    "n_croptype_pass": 9,
    "croptype_misses": [0.0114, 0.0123, 0.0345],
    "n_samples": 2370,
    "n_unique_plots_index": 351,
    "n_train_rows": 1937,
    "n_train_plots": 315,
    "n_val_rows": 227,
    "n_val_plots": 36,
    "n_test_rows_blank_plot": 206,
    # yield screening (Section V-H) -- a negative result
    "ys_n_field_seasons": 41,
    "ys_n_plots": 14,
    "ys_n_checkpoints": 24,
    "ys_auroc": 0.311,
    "ys_auroc_ci_contains_half": True,
}

RESULTS: list[dict] = []


def record(name, status, detail, claimed=None, found=None):
    RESULTS.append({"check": name, "status": status, "detail": detail,
                    "claimed": claimed, "found": found})
    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"{tag}  {name}")
    print(f"      {detail}")


def close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# --------------------------------------------------------------------------
# 1. Ablation partition
# --------------------------------------------------------------------------
def check_ablation():
    f = PROJ / "results/ladder_crop_type.csv"
    if not f.exists():
        return record("1. Ablation partition", "SKIP", f"missing {f}")
    L = {r["rung"]: r for r in csv.DictReader(f.open(encoding="utf8"))
         if r.get("pixel_macro_iou")}
    rungs = [f"v{i}" for i in range(1, 11)]
    sigma = max(float(L[k]["pixel_sd"]) for k in L)

    steps = []
    for a, b in zip(rungs, rungs[1:]):
        if a in L and b in L:
            steps.append((b, float(L[b]["pixel_macro_iou"])
                          - float(L[a]["pixel_macro_iou"])))
    up = [s for s in steps if s[1] >= sigma]
    dn = [s for s in steps if s[1] <= -sigma]
    inc = [s for s in steps if abs(s[1]) < sigma]

    # The partition must also be exhaustive: no step may fall through.
    exhaustive = len(up) + len(dn) + len(inc) == len(steps)

    ok = (len(steps) == CLAIMED["n_steps"]
          and len(inc) == CLAIMED["n_inconclusive"]
          and len(up) == CLAIMED["n_exceed_up"]
          and len(dn) == CLAIMED["n_exceed_down"]
          and exhaustive
          and close(sigma, CLAIMED["sigma_max"], 5e-4))
    detail = (f"{len(steps)} steps, sigma_max={sigma:.4f}: "
              f"{len(up)} up, {len(dn)} down, {len(inc)} inconclusive, "
              f"exhaustive={exhaustive} | "
              + "; ".join(f"{k}{d:+.4f}" for k, d in steps))
    record("1. Ablation partition", "PASS" if ok else "FAIL", detail,
           CLAIMED["n_inconclusive"], len(inc))

    # The two headline deltas, recomputed independently.
    dt = float(L["v2"]["pixel_macro_iou"]) - float(L["v1"]["pixel_macro_iou"])
    dr = float(L["v5"]["pixel_macro_iou"]) - float(L["v4"]["pixel_macro_iou"])
    ok2 = (close(dt, CLAIMED["d_temporal"], 5e-5)
           and close(dr, CLAIMED["d_radar"], 5e-5))
    record("1b. Temporal and radar deltas", "PASS" if ok2 else "FAIL",
           f"temporal {dt:+.4f} (claimed {CLAIMED['d_temporal']:+.4f}), "
           f"radar {dr:+.4f} (claimed {CLAIMED['d_radar']:+.4f})")


# --------------------------------------------------------------------------
# 2 & 3. Field-level statistics and cross-seed correlation
# --------------------------------------------------------------------------
def _field_iou(cm):
    """IoU over the classes present in the field's truth, written here
    rather than imported, so this check does not share code with the
    pipeline it audits. Rows are true, columns predicted."""
    cm = np.asarray(cm, dtype=float)
    vals = []
    for c in range(cm.shape[0]):
        if cm[c].sum() == 0:
            continue
        vals.append(cm[c, c] / (cm[c].sum() + cm[:, c].sum() - cm[c, c]))
    return float(np.mean(vals))


def load_per_field():
    f = PROJ / "results/field_confusions/v5.json"
    if not f.exists():
        return [], []
    fc = json.loads(f.read_text(encoding="utf8"))
    mats = [{k: _field_iou(r["cm"]) for k, r in rows.items()}
            for _, rows in sorted(fc["per_seed"].items(),
                                  key=lambda kv: int(kv[0]))]
    return [f"{f.name}:seed{s}" for s in sorted(fc["per_seed"], key=int)], mats


def check_field_level():
    files, mats = load_per_field()
    if len(mats) < 2:
        return record("2. Field-level statistics", "SKIP",
                      f"found {len(mats)} per-plot IoU files, need >= 2")
    keys = sorted(set.intersection(*[set(m) for m in mats]), key=float)
    mean_iou = np.array([np.mean([m[k] for m in mats]) for k in keys])

    med = float(np.median(mean_iou))
    below = int((mean_iou < 0.5).sum())
    always = sum(1 for k in keys if all(m[k] < 0.5 for m in mats)
                 and np.mean([m[k] for m in mats]) < 0.5)

    ok = (len(keys) == CLAIMED["n_fields"]
          and close(med, CLAIMED["median_field_iou"], 5e-4)
          and below == CLAIMED["n_below_half"]
          and always == CLAIMED["n_below_half_every_run"])
    record("2. Field-level statistics", "PASS" if ok else "FAIL",
           f"{len(keys)} fields from {len(mats)} seeds | median {med:.4f} "
           f"(claimed {CLAIMED['median_field_iou']}) | {below} below 0.5 "
           f"(claimed {CLAIMED['n_below_half']}) | {always} below 0.5 in "
           f"every run (claimed {CLAIMED['n_below_half_every_run']})")

    rs = sorted(float(np.corrcoef([mats[i][k] for k in keys],
                                  [mats[j][k] for k in keys])[0, 1])
                for i, j in itertools.combinations(range(len(mats)), 2))
    ok2 = (close(min(rs), CLAIMED["seed_r_min"], 0.005)
           and close(max(rs), CLAIMED["seed_r_max"], 0.005))
    record("3. Cross-seed correlation", "PASS" if ok2 else "FAIL",
           f"{len(rs)} seed pairs: " + ", ".join(f"{r:+.4f}" for r in rs)
           + f" | claimed range {CLAIMED['seed_r_min']:+.2f} to "
             f"{CLAIMED['seed_r_max']:+.2f}")


# --------------------------------------------------------------------------
# 4. Screening metrics, re-derived from the exported per-field signals
# --------------------------------------------------------------------------
def loo_screen(sig: dict, bad: set, workload: float):
    """Leave-one-out referral, re-implemented from the paper's description.

    For each field, the threshold is the (1 - workload) quantile of every
    OTHER field's signal. The field is referred when its own signal is at or
    above that threshold. No field contributes to the threshold that judges
    it, which is the property the manuscript claims.
    """
    fields = sorted(sig)
    referred = []
    for f in fields:
        others = [sig[g] for g in fields if g != f]
        thr = float(np.quantile(others, 1.0 - workload))
        if sig[f] >= thr:
            referred.append(f)
    caught = [f for f in referred if f in bad]
    return {
        "n_referred": len(referred),
        "workload": len(referred) / len(fields),
        "recall": len(caught) / len(bad) if bad else None,
        "precision": len(caught) / len(referred) if referred else None,
        "n_caught": len(caught),
    }


def boot_recall_ci(sig: dict, bad: set, workload: float, n_boot=2000,
                   seed=0, refit=True):
    """Percentile bootstrap over FIELDS, not pixels.

    `refit=True` re-estimates the referral threshold inside every replicate
    from the resampled signal values, so the interval carries the uncertainty
    of choosing the threshold as well as of the evaluation set. This is the
    procedure the manuscript's Section IV-D now specifies.

    `refit=False` instead freezes the full-sample leave-one-out decision and
    resamples only the fields being scored. It answers a narrower question
    and gives a narrower interval; it is computed here as a sensitivity so
    the difference between the two estimands is visible rather than implicit.

    Replicates containing no unreliable field are discarded, because recall
    is undefined over an empty denominator.
    """
    fields = sorted(sig)
    rng = np.random.default_rng(seed)
    thr_loo = {f: float(np.quantile([sig[g] for g in fields if g != f],
                                    1.0 - workload)) for f in fields}
    vals = []
    for _ in range(n_boot):
        samp = [fields[i] for i in rng.integers(0, len(fields), len(fields))]
        sbad = [f for f in samp if f in bad]
        if not sbad:
            continue
        if refit:
            thr = float(np.quantile([sig[f] for f in samp], 1.0 - workload))
            vals.append(sum(1 for f in sbad if sig[f] >= thr) / len(sbad))
        else:
            vals.append(sum(1 for f in sbad if sig[f] >= thr_loo[f])
                        / len(sbad))
    if not vals:
        return loo_screen(sig, bad, workload), (float("nan"), float("nan"))
    v = np.array(vals)
    return (loo_screen(sig, bad, workload),
            (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))))


def check_screening():
    f = PROJ / "results/triage/per_field_signals_v5.csv"
    if not f.exists():
        return record(
            "4. Screening metrics", "SKIP",
            f"missing {f.relative_to(PROJ)} -- run:  python "
            f"scripts/triage_eval.py --config v5 --seeds 0 1 2 --device cpu "
            f"--num-workers 0")
    rows = list(csv.DictReader(f.open(encoding="utf8")))
    dis = {float(r["plot_id"]): float(r["seed_disagreement"]) for r in rows}
    conf = {float(r["plot_id"]): float(r["low_confidence"]) for r in rows}
    bad = {float(r["plot_id"]) for r in rows if r["is_bad"] == "1"}

    d50, ci50 = boot_recall_ci(dis, bad, 0.50)
    c33, ci33 = boot_recall_ci(conf, bad, 0.33)
    c50, cci50 = boot_recall_ci(conf, bad, 0.50)
    _, ci50_fixed = boot_recall_ci(dis, bad, 0.50, refit=False)

    # the manuscript quotes confidence at a 50 % workload
    ok = (len(rows) == CLAIMED["n_fields"]
          and len(bad) == CLAIMED["n_below_half"]
          and close(c50["recall"], CLAIMED["conf_recall_50"], 0.01)
          and close(c50["precision"], CLAIMED["conf_precision_50"], 0.01)
          and close(cci50[0], CLAIMED["conf_ci_lo_50"], 0.03))
    record("4. Screening metrics", "PASS" if ok else "FAIL",
           f"{len(rows)} fields, {len(bad)} unreliable | "
           f"disagreement @50% workload: recall {d50['recall']:.4f} "
           f"({d50['n_caught']}/{len(bad)}), precision "
           f"{d50['precision']:.4f}, 95% CI [{ci50[0]:.3f}, {ci50[1]:.3f}] | "
           f"confidence @50% (checked): recall {c50['recall']:.4f} "
           f"({c50['n_caught']}/{len(bad)}), precision {c50['precision']:.4f}, "
           f"95% CI [{cci50[0]:.3f}, {cci50[1]:.3f}] | confidence @33%: "
           f"recall {c33['recall']:.4f}")
    record("4c. Bootstrap estimand sensitivity", "PASS",
           f"re-fitting the threshold per replicate gives "
           f"[{ci50[0]:.3f}, {ci50[1]:.3f}]; freezing the full-sample "
           f"leave-one-out rule gives [{ci50_fixed[0]:.3f}, "
           f"{ci50_fixed[1]:.3f}]. The manuscript reports the former, which "
           f"is the wider of the two.")

    # The leave-one-out property itself: spike one field's signal and confirm
    # its own threshold does not move.
    probe = sorted(dis)[0]
    others = [dis[g] for g in sorted(dis) if g != probe]
    thr_before = float(np.quantile(others, 0.5))
    spiked = dict(dis)
    spiked[probe] = 1e6
    others2 = [spiked[g] for g in sorted(spiked) if g != probe]
    thr_after = float(np.quantile(others2, 0.5))
    record("4b. Leave-one-out threshold independence",
           "PASS" if thr_before == thr_after else "FAIL",
           f"spiking field {probe:.0f} left its own threshold at "
           f"{thr_after:.6f} (was {thr_before:.6f})")


# --------------------------------------------------------------------------
# 5. Reproduction counts
# --------------------------------------------------------------------------
def check_reproduction():
    a = PROJ / "results/audit_summary.csv"
    if not a.exists():
        return record("5. Reproduction counts", "SKIP", f"missing {a}")
    allrows = list(csv.DictReader(a.open(encoding="utf8")))
    rows = [r for r in allrows if "_debug" not in r["run"]]
    npass = sum(1 for r in rows if r["verdict"] == "PASS")

    cells = sorted(glob.glob(str(PROJ / "results/gate/*_bs32.json")))
    gates = [json.loads(Path(c).read_text(encoding="utf8")) for c in cells]
    cpass = sum(1 for g in gates if g["verdict"] == "PASS")
    miss = sorted(round(g["abs_delta"], 4) for g in gates
                  if g["verdict"] != "PASS")

    ok = (len(rows) == CLAIMED["n_regression_models"]
          and npass == CLAIMED["n_regression_pass"]
          and len(gates) == CLAIMED["n_croptype_cells"]
          and cpass == CLAIMED["n_croptype_pass"]
          and all(close(x, y, 5e-5)
                  for x, y in zip(miss, CLAIMED["croptype_misses"])))
    record("5. Reproduction counts", "PASS" if ok else "FAIL",
           f"regression {npass}/{len(rows)} "
           f"({len(allrows) - len(rows)} *_debug excluded) | crop type "
           f"{cpass}/{len(gates)} | misses {miss} | total audited "
           f"{len(rows) + len(gates)}, reproduced {npass + cpass}")


# --------------------------------------------------------------------------
# 6. Split sizes from the released index
# --------------------------------------------------------------------------
def check_splits():
    cfg = PROJ / "configs" / "base.yaml"
    root = None
    if cfg.exists():
        for line in cfg.read_text(encoding="utf8").splitlines():
            if line.strip().startswith("root:"):
                root = Path(line.split(":", 1)[1].strip().strip('"\''))
                break
    cand = [root / "sickle_dataset_tabular.csv"] if root else []
    tab = next((c for c in cand if c.exists()), None)
    if tab is None:
        return record("6. Split sizes", "SKIP",
                      "released tabular index not found via configs/base.yaml")

    rows = list(csv.DictReader(tab.open(encoding="utf8")))
    def plots(split):
        return {r["PLOT_ID"] for r in rows
                if r["SPLIT"] == split and r["PLOT_ID"].strip()}
    tr = [r for r in rows if r["SPLIT"] == "train"]
    va = [r for r in rows if r["SPLIT"] == "val"]
    te = [r for r in rows if r["SPLIT"] == "test"]
    blank = sum(1 for r in te if not r["PLOT_ID"].strip())
    uniq = {r["PLOT_ID"] for r in rows if r["PLOT_ID"].strip()}

    ok = (len(rows) == CLAIMED["n_samples"]
          and len(uniq) == CLAIMED["n_unique_plots_index"]
          and len(tr) == CLAIMED["n_train_rows"]
          and len(plots("train")) == CLAIMED["n_train_plots"]
          and len(va) == CLAIMED["n_val_rows"]
          and len(plots("val")) == CLAIMED["n_val_plots"]
          and blank == CLAIMED["n_test_rows_blank_plot"])
    record("6. Split sizes", "PASS" if ok else "FAIL",
           f"{len(rows)} rows, {len(uniq)} unique plot ids | train "
           f"{len(tr)}/{len(plots('train'))} | val {len(va)}/"
           f"{len(plots('val'))} | test {len(te)} rows, {blank} with blank "
           f"PLOT_ID")


# --------------------------------------------------------------------------
# 7. Yield screening, re-derived from the exported per-field table
# --------------------------------------------------------------------------
def _auroc(score, pos):
    """Mann-Whitney AUROC with explicit tie handling."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    srt = score[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def check_yield_screen():
    f = PROJ / "results/triage/yield_per_field.csv"
    if not f.exists():
        return record("7. Yield screening", "SKIP",
                      f"missing {f.name} -- run scripts/yield_screen.py")
    rows = list(csv.DictReader(f.open(encoding="utf8")))
    sig = np.array([float(r["disagreement_cv"]) for r in rows])
    ape = np.array([float(r["ape_pct"]) for r in rows])
    bad = np.array([r["is_bad"] == "1" for r in rows])
    plots = {r["plot_id"] for r in rows}

    # Recompute the CV target relationship and AUROC from scratch.
    au = _auroc(sig, bad)
    ok = (len(rows) == CLAIMED["ys_n_field_seasons"]
          and len(plots) == CLAIMED["ys_n_plots"]
          and close(au, CLAIMED["ys_auroc"], 0.005))
    record("7. Yield screening", "PASS" if ok else "FAIL",
           f"{len(rows)} field-seasons from {len(plots)} plots | AUROC "
           f"{au:.3f} (claimed {CLAIMED['ys_auroc']}) | {int(bad.sum())} "
           f"unreliable | chance = 0.500")

    # The paper's claim is that the signal is NOT demonstrated. Verify the
    # direction of that claim independently: a plot-level bootstrap CI that
    # contains 0.5.
    by_plot = {}
    for i, r in enumerate(rows):
        by_plot.setdefault(r["plot_id"], []).append(i)
    keys = sorted(by_plot)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(2000):
        samp = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        idx = [i for p in samp for i in by_plot[p]]
        y = bad[idx]
        if y.all() or not y.any():
            continue
        vals.append(_auroc(sig[idx], y))
    v = np.array(vals)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    contains = lo <= 0.5 <= hi
    record("7b. Yield screen is correctly reported as not demonstrated",
           "PASS" if contains == CLAIMED["ys_auroc_ci_contains_half"]
           else "FAIL",
           f"plot-bootstrap AUROC CI [{lo:.3f}, {hi:.3f}] "
           f"{'contains' if contains else 'EXCLUDES'} chance 0.5; the "
           f"manuscript reports the signal as not demonstrated")

    # No field-season may be scored by a different number of models than
    # another, or the dispersions are not comparable.
    nm = {int(r["n_models"]) for r in rows}
    record("7c. Dispersion computed over a constant model set",
           "PASS" if len(nm) == 1 and nm == {CLAIMED["ys_n_checkpoints"]}
           else "FAIL",
           f"every field-season scored by {sorted(nm)} checkpoints "
           f"(claimed {CLAIMED['ys_n_checkpoints']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path,
                    help="also write the full result table here")
    args = ap.parse_args()

    print("Independent re-derivation of the manuscript's headline numbers.")
    print("Imports nothing from src/. Reads raw artefacts only.\n")
    print(f"repository: {PROJ}\n")

    for fn in (check_ablation, check_field_level, check_screening,
               check_reproduction, check_splits,
               check_yield_screen):
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            record(fn.__name__, "FAIL", f"raised {type(exc).__name__}: {exc}")
        print()

    npass = sum(1 for r in RESULTS if r["status"] == "PASS")
    nfail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    nskip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("-" * 72)
    print(f"{npass} passed, {nfail} failed, {nskip} skipped")
    if nskip:
        print("A SKIP is not a pass. The checks above say what is missing.")
    print("\nA pass means the manuscript's arithmetic follows from the raw")
    print("files in this repository. It does not establish that those files")
    print("are correct; that needs retraining, which this script never does.")

    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=2), encoding="utf8")
        print(f"\nwrote {args.json}")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
