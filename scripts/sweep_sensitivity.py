"""Sensitivity of the sweep's trend results to individual fields and seeds.

    python scripts/sweep_rescore.py && python scripts/sweep_analysis.py
    python scripts/sweep_sensitivity.py

EXPLORATORY and POST HOC. The verdict is the one configs/sweep_analysis_protocol.yaml
fixed and scripts/sweep_analysis.py reported; nothing here changes it. These
analyses were chosen after seeing that verdict, to ask how much it depends on
a few fields or a few seeds.

  1. Fields. The field-IoU metric is the mean over the 36 validation fields,
     and a slope is linear, so the field-IoU trend slope is exactly the mean
     of 36 per-field slopes. Each field's slope of the temporal advantage
     (v2 - v1) on log2(plot count), averaged over seeds, shows whether the
     negative trend is spread across fields or carried by a few. Leave-one-
     field-out and drop-the-k-most-negative versions follow.
  2. Seeds. Leave-one-seed-out mean slope and 95 % t-interval (df = 3), for
     both metrics.
  3. Both sources of uncertainty. The protocol's interval resamples seeds
     only, treating the 36 validation fields as fixed. Here a two-level
     bootstrap resamples seeds (5, with replacement) AND fields (36, with
     replacement); pooled macro IoU is recomputed from the summed per-field
     confusion matrices of the drawn fields. Five seeds make any bootstrap
     over seeds coarse, so field-only and seed-only versions are given too.

Writes results/sweep/sweep_sensitivity.json (per-field slopes, not labels).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
from sweep_analysis import COUNTS, CONFIGS, SEEDS, RES  # noqa: E402
from src.evaluation.metrics import own_class_iou_from_confusion  # noqa: E402

X = np.log2(np.array(COUNTS, float))
B = 10_000


def slope_last(y):
    """OLS slope over the last axis (the six plot counts)."""
    xc = X - X.mean()
    return (np.asarray(y) - np.asarray(y).mean(-1, keepdims=True)) @ xc / (xc @ xc)


def macro_from_cm(cm):
    """Macro IoU of a 2x2 confusion matrix (rows true, columns predicted);
    a class absent from both truth and prediction is skipped."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(-1) + cm.sum(-2) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    return np.nanmean(iou, -1)


def tci(v):
    v = np.asarray(v, float)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return {"mean": float(m), "sd": float(sd), "ci95": [float(m - h), float(m + h)]}


def main() -> int:
    with open(RES / "rescore.csv", encoding="utf8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["config"] in CONFIGS]
    idx = {(int(r["plot_count"]), r["config"], int(r["seed"])): r for r in rows}
    fields = None
    FI = np.zeros((len(COUNTS), len(CONFIGS), len(SEEDS), 36))
    CM = np.zeros((len(COUNTS), len(CONFIGS), len(SEEDS), 36, 2, 2))
    for ci, c in enumerate(COUNTS):
        for ki, k in enumerate(CONFIGS):
            for si, s in enumerate(SEEDS):
                r = idx[(c, k, s)]
                fc = json.loads((RES / "field_confusions" /
                                 f"{r['subset']}__{r['run']}.json")
                                .read_text(encoding="utf8"))["fields"]
                if fields is None:
                    fields = sorted(fc, key=float)
                assert sorted(fc, key=float) == fields
                for fi, f in enumerate(fields):
                    cm = np.asarray(fc[f]["cm"], float)
                    CM[ci, ki, si, fi] = cm
                    FI[ci, ki, si, fi] = own_class_iou_from_confusion(cm)
                # the confusion matrices must reproduce the recorded scores
                assert abs(FI[ci, ki, si].mean() - float(r["field_iou_mean"])) < 1e-9
                assert abs(macro_from_cm(CM[ci, ki, si].sum(0))
                           - float(r["pooled_macro_iou_training"])) < 1e-9
    v1, v2 = CONFIGS.index("v1"), CONFIGS.index("v2")

    # ---- 1. fields
    adv = FI[:, v2] - FI[:, v1]                    # (count, seed, field)
    per_field = slope_last(np.moveaxis(adv, 0, -1)).mean(0)   # (field,)
    overall = float(per_field.mean())
    order = np.argsort(per_field)
    neg_total = per_field[per_field < 0].sum()
    loo = np.array([np.delete(per_field, i).mean() for i in range(36)])
    drop = {str(k): float(np.delete(per_field, order[:k]).mean())
            for k in (1, 3, 5, 10)}
    # how each field's single-image and temporal scores move
    s_v1 = slope_last(np.moveaxis(FI[:, v1], 0, -1)).mean(0)
    s_v2 = slope_last(np.moveaxis(FI[:, v2], 0, -1)).mean(0)
    fields_out = {
        "slope_check_equals_protocol_field_slope": overall,
        "n_fields": 36,
        "n_negative": int((per_field < -0.01).sum()),
        "n_near_zero": int((abs(per_field) <= 0.01).sum()),
        "n_positive": int((per_field > 0.01).sum()),
        "share_of_negative_mass_in_5_most_negative":
            float(per_field[order[:5]].sum() / neg_total) if neg_total < 0 else None,
        "leave_one_field_out_range": [float(loo.min()), float(loo.max())],
        "mean_slope_after_dropping_k_most_negative": drop,
        "most_negative": [{"field": fields[i], "slope": float(per_field[i]),
                           "v1_slope": float(s_v1[i]), "v2_slope": float(s_v2[i])}
                          for i in order[:8]],
        "per_field": {fields[i]: {"advantage_slope": float(per_field[i]),
                                  "v1_slope": float(s_v1[i]),
                                  "v2_slope": float(s_v2[i])}
                      for i in range(36)},
        "fields_where_v1_rises_and_v2_flat": int(((s_v1 > 0.02) & (abs(s_v2) <= 0.02)).sum()),
    }

    # ---- 2. seeds (both metrics, per-seed slopes from the protocol's data)
    metric_vals = {
        "field_iou": FI.mean(-1),                                  # (c,k,s)
        "pooled_macro_iou": macro_from_cm(CM.sum(3)),              # (c,k,s)
    }
    seeds_out = {}
    for name, M in metric_vals.items():
        sl = slope_last(np.moveaxis(M[:, v2] - M[:, v1], 0, -1))   # (seed,)
        seeds_out[name] = {
            "per_seed": [float(v) for v in sl],
            "all": tci(sl),
            "leave_one_seed_out": {str(SEEDS[i]): tci(np.delete(sl, i))
                                   for i in range(len(SEEDS))},
        }

    # ---- 3. bootstrap over seeds and/or fields
    rng = np.random.default_rng(0)
    boot = {}
    for mode in ("seeds_and_fields", "fields_only", "seeds_only"):
        res = {"field_iou": [], "pooled_macro_iou": []}
        for _ in range(B):
            sd = (rng.integers(0, 5, 5) if mode != "fields_only"
                  else np.arange(5))
            fd = (rng.integers(0, 36, 36) if mode != "seeds_only"
                  else np.arange(36))
            w = np.bincount(fd, minlength=36).astype(float)
            fi = (FI[:, :, sd] * w).sum(-1) / w.sum()              # (c,k,s)
            pm = macro_from_cm(np.tensordot(CM[:, :, sd], w, axes=([3], [0])))
            for name, M in (("field_iou", fi), ("pooled_macro_iou", pm)):
                res[name].append(float(slope_last(
                    np.moveaxis(M[:, v2] - M[:, v1], 0, -1)).mean()))
        boot[mode] = {k: {"ci95": [float(np.percentile(v, 2.5)),
                                   float(np.percentile(v, 97.5))],
                          "share_below_zero": float(np.mean(np.array(v) < 0))}
                      for k, v in res.items()}

    out = {"status": "exploratory, post hoc; the protocol verdict stands",
           "slope_unit": "temporal advantage (v2 - v1) per doubling of plots",
           "fields": fields_out, "seeds": seeds_out,
           "bootstrap": {"B": B, "seed": 0, **boot}}
    (RES / "sweep_sensitivity.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf8")

    f = fields_out
    print(f"FIELDS  mean per-field slope {f['slope_check_equals_protocol_field_slope']:+.4f}"
          f" | negative {f['n_negative']}, near zero {f['n_near_zero']}, "
          f"positive {f['n_positive']} of 36")
    print(f"  5 most negative fields hold {f['share_of_negative_mass_in_5_most_negative']:.0%}"
          f" of the negative slope mass; leave-one-field-out range "
          f"[{f['leave_one_field_out_range'][0]:+.4f}, {f['leave_one_field_out_range'][1]:+.4f}]")
    print("  after dropping the k most negative: " + ", ".join(
        f"k={k}: {v:+.4f}" for k, v in f["mean_slope_after_dropping_k_most_negative"].items()))
    print(f"  fields where single-image rises and temporal stays flat: "
          f"{f['fields_where_v1_rises_and_v2_flat']}")
    for name, s in seeds_out.items():
        lo = s["leave_one_seed_out"]
        print(f"SEEDS {name}: per seed " + " ".join(f"{v:+.3f}" for v in s["per_seed"])
              + " | leave-one-out CIs: " + "; ".join(
                  f"-{k}: [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for k, v in lo.items()))
    for mode, d in boot.items():
        print(f"BOOT {mode:<17} " + "  ".join(
            f"{k} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] P(<0) {v['share_below_zero']:.2f}"
            for k, v in d.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
