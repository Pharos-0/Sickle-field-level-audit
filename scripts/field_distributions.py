"""Item 3: the per-field error DISTRIBUTION, and what carries the aggregate.

    python scripts/field_distributions.py

The manuscript's central claim is that a pooled metric misrepresents
individual fields. A median is itself an aggregate, so quoting the gap as
"0.8665 pooled vs 0.6755 per-field" repeats the error it criticises one
level down. This reports the whole distribution, with intervals, with the
fraction below stated thresholds, and with the mechanism.

THE MECHANISM, MEASURED RATHER THAN ASSERTED
--------------------------------------------
The pooled confusion matrix is the plain SUM of the per-field confusion
matrices (verified in scripts/plot_level_eval.py, which recomputes the
pooled figure from them and checks it against the reference). So a field's
influence on the pooled metric is its pixel count, and dropping a field and
recomputing is exact arithmetic, not a reweighting argument.

Field sizes here are extremely unequal (scripts/field_size.py): one
validation field holds 47% of all scored pixels. The pooled number is
therefore close to a measurement of a few fields, and that is reported as a
leave-fields-out shift, which is the claim in its least deniable form.

WHICH PER-FIELD SCORE (corrected 2026-10-12)
--------------------------------------------
The per-field score is the IoU of the crop(s) the field holds
(src/evaluation/metrics.py, own_class_iou_from_confusion), computed from the
per-field confusion matrices of scripts/field_confusions.py and
scripts/plot_level_eval.py. An earlier version used macro IoU over both
classes. 35 of 36 validation fields hold one crop, and for such a field the
macro value is 1.0 when every pixel is right and at most 0.5 once one pixel
is wrong, so it reported most imperfect fields as below 0.5 regardless of
how many pixels were right. The macro values are still computed and written
under "macro_contrast", so the size of the correction is on the record.

THRESHOLDS, AND WHY THESE
-------------------------
  0.50  predicted and true paddy areas overlap less than they disagree
  0.70  a commonly quoted "usable" level in segmentation practice
  0.30  near-failure
None is an official criterion, and the output says so. All three are
reported so no conclusion rests on one cut.

Writes results/field_distributions.json and three figures.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.utils import completed_seed_runs  # noqa: E402
from src.evaluation.metrics import own_class_iou_from_confusion  # noqa: E402
OUT = PROJ / "results" / "field_distributions.json"
CONF = PROJ / "results" / "field_confusions"
FIGDIR = PROJ / "results" / "figures_ieee"
THRESHOLDS = (0.30, 0.50, 0.70)
RUNG = "v5"
NBOOT = 10000


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def boot_flat(x, stat, n_boot=NBOOT, seed=0):
    """Resample FIELDS. Valid when each element is one independent field."""
    rng = np.random.default_rng(seed)
    v = np.array([stat(x[rng.integers(0, len(x), len(x))])
                  for _ in range(n_boot)], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_grouped(x, groups, stat, n_boot=NBOOT, seed=0):
    """Resample PLOTS, carrying all of a plot's field-seasons together.

    The regression evaluation set is 41 field-seasons at only 14 plots, and
    in this dataset almost all yield variance is between plots, so
    resampling field-seasons independently would understate every interval.
    """
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        j = np.concatenate([pos[g] for g in pick])
        r = float(stat(x[j]))
        if np.isfinite(r):
            v.append(r)
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def quantiles(x, qs=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    return {str(q): float(np.percentile(x, q)) for q in qs}


def rankdata(v):
    """Ranks with average ranks for ties."""
    v = np.asarray(v, dtype=float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    s = v[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------
# crop type, this study's own ladder rung
# --------------------------------------------------------------------------

def ladder_crop_type(size):
    # An interrupted run leaves best_per_plot_iou.json behind, so globbing
    # the directory is not the same as asking which seeds finished.
    kept, dropped = completed_seed_runs("crop_type", RUNG, PROJ)
    seeds = [d / "best_per_plot_iou.json" for d, _ in kept]
    seeds = [f for f in seeds if f.exists()]
    if not seeds:
        raise RuntimeError("no completed per-plot IoU artefacts for rung "
                           + RUNG)
    fc_path = CONF / f"{RUNG}.json"
    if not fc_path.exists():
        raise RuntimeError(f"missing {fc_path}; run scripts/field_confusions.py")
    fc = json.loads(fc_path.read_text(encoding="utf8"))
    want = sorted(int(f.parent.name.rsplit("seed", 1)[1]) for f in seeds)
    if sorted(int(k) for k in fc["per_seed"]) != want:
        raise RuntimeError(f"field_confusions seeds {sorted(fc['per_seed'])} "
                           f"differ from the completed seeds {want}")
    if dropped:
        print("EXCLUDED, training did not complete:")
        for d, b in dropped:
            ran = (b or {}).get("epochs_run_so_far", "?")
            plan = (b or {}).get("epochs_planned", "?")
            print(f"  {d.name}: {ran} of {plan} epochs")
        print()
    per, per_macro = {}, {}
    for f in seeds:
        for k, v in json.loads(f.read_text(encoding="utf8")).items():
            per_macro.setdefault(float(k), []).append(float(v))
    for _, rows in fc["per_seed"].items():
        for k, r in rows.items():
            per.setdefault(float(k), []).append(
                own_class_iou_from_confusion(np.asarray(r["cm"])))
    plots = sorted(per)
    x = np.array([float(np.mean(per[p])) for p in plots])
    xm = np.array([float(np.mean(per_macro[p])) for p in plots])
    spread = np.array([float(np.ptp(per[p])) for p in plots])
    n_single = sum(
        1 for k, r in next(iter(fc["per_seed"].values())).items()
        if np.count_nonzero(np.asarray(r["cm"]).sum(1)) == 1)

    row = next((r for r in csv.DictReader(
        (PROJ / "results/ladder_crop_type.csv").open(encoding="utf8"))
        if r["rung"] == RUNG), None)
    pooled = float(row["pixel_macro_iou"]) if row else float("nan")
    plotmean = float(row["plot_macro_iou"]) if row else float("nan")

    w = np.array([size[p] for p in plots], dtype=float)
    out = {
        "rung": RUNG,
        "seed_dirs": [f.parent.name for f in seeds],
        "n_seeds": len(seeds),
        "excluded_incomplete": [
            {"dir": d.name,
             "epochs_run": (b or {}).get("epochs_run_so_far"),
             "epochs_planned": (b or {}).get("epochs_planned")}
            for d, b in dropped],
        "n_fields": len(plots),
        "per_field_score": "IoU of the crop(s) the field holds",
        "n_single_crop_fields": int(n_single),
        "pooled_pixel_macro_iou": pooled,
        "per_field_mean": float(x.mean()),
        "per_field_mean_from_ladder_csv": plotmean,
        "per_field_sd": float(x.std(ddof=1)),
        "gap_pooled_minus_per_field": pooled - float(x.mean()),
        "quantiles": quantiles(x),
        "ci_mean": boot_flat(x, np.mean),
        "ci_median": boot_flat(x, np.median),
        "seed_spread_per_field": {
            "median_range_over_seeds": float(np.median(spread)),
            "max_range_over_seeds": float(spread.max()),
            "n_fields_range_above_0_2": int((spread > 0.2).sum()),
        },
        "below": {},
    }
    for t in THRESHOLDS:
        out["below"][str(t)] = {
            "n": int((x < t).sum()), "frac": float((x < t).mean()),
            "ci": boot_flat(x, lambda a, t=t: float((a < t).mean()))}
    out["macro_contrast"] = {
        "what": ("macro IoU over both classes, as reported before the "
                 "2026-10-12 correction; for a single-crop field it is at "
                 "most 0.5 once any pixel is wrong"),
        "per_field_mean": float(xm.mean()),
        "median": float(np.median(xm)),
        "below": {str(t): int((xm < t).sum()) for t in THRESHOLDS},
        "n_fields_macro_below_half_with_field_iou_at_least_0_9": int(
            ((xm < 0.5) & (x >= 0.9)).sum()),
    }

    short = np.maximum(pooled - x, 0.0)
    o = np.argsort(-short)
    for lab, k in (("worst_decile", max(1, len(x) // 10)),
                   ("worst_quartile", max(1, len(x) // 4))):
        out["share_of_shortfall_" + lab] = {
            "n_fields": k,
            "share": (float(short[o[:k]].sum() / short.sum())
                      if short.sum() > 0 else float("nan"))}

    idx = np.arange(len(x))
    out["size"] = {
        "weight_measure": "pixels entering the pooled confusion matrix",
        "spearman_weight_vs_iou": spearman(w, x),
        "spearman_weight_vs_iou_ci": boot_flat(
            idx, lambda i: spearman(w[i], x[i])),
        "iou_mean_small_half": float(x[w <= np.median(w)].mean()),
        "iou_mean_large_half": float(x[w > np.median(w)].mean()),
        "weighted_mean_iou": float((w * x).sum() / w.sum()),
        "unweighted_mean_iou": float(x.mean()),
        "note": ("weighted_mean_iou is a DIAGNOSTIC, not the pooled metric. "
                 "IoU is a ratio of sums, so a size-weighted mean of "
                 "per-field IoUs is not algebraically equal to the pooled "
                 "IoU. The exact statement is the leave-fields-out "
                 "recomputation under released_crop_type."),
    }
    return out, x, w, plots


# --------------------------------------------------------------------------
# crop type, the released checkpoints
# --------------------------------------------------------------------------

def released_crop_type():
    rows, missing = [], []
    for f in sorted(glob.glob(str(PROJ / "results/plot_level/*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        rec = {"run": d["run"],
               "pooled": float(d["pixel_pooled"]["macro_iou"]),
               "per_field_mean": float(d["plot_level"]["plot_macro_iou"]),
               "per_field_sd": float(d["plot_level"]["plot_iou_sd"]),
               "gap": float(d["pixel_minus_plot"]),
               "ci": [float(d["plot_bootstrap_ci"]["lo"]),
                      float(d["plot_bootstrap_ci"]["hi"])]}
        macro = np.array([float(v) for v in d["per_plot_iou"].values()])
        prf = np.array([own_class_iou_from_confusion(np.asarray(cm))
                        for cm in d["per_plot_confusion"].values()])
        rec["per_field_mean"] = float(prf.mean())
        rec["gap"] = rec["pooled"] - float(prf.mean())
        rec["macro_per_field_mean"] = float(macro.mean())
        rec["macro_gap"] = float(d["pixel_minus_plot"])
        rec["macro_frac_below_0.5"] = float((macro < 0.5).mean())
        rec["n_fields"] = int(prf.size)
        rec["per_field_quantiles"] = quantiles(prf, (0, 25, 50, 75, 100))
        for t in THRESHOLDS:
            rec["frac_below_" + str(t)] = float((prf < t).mean())
        loo = d.get("leave_fields_out")
        if loo and "_pooled_recomputed" in loo:
            base = float(loo["_pooled_recomputed"])
            rec["pooled_recomputed"] = base
            rec["recompute_matches_reference"] = bool(
                abs(base - rec["pooled"]) < 1e-9)
            for lab in ("largest_field", "top_decile", "top_quartile"):
                if lab in loo:
                    pw = float(loo[lab]["pooled_macro_iou_without_them"])
                    rec["pooled_without_" + lab] = pw
                    rec["shift_without_" + lab] = pw - base
                    rec["n_dropped_" + lab] = int(loo[lab]["n_fields_dropped"])
                    rec["px_share_" + lab] = float(loo[lab]["px_share_dropped"])
        else:
            missing.append(d["run"])
        rows.append(rec)
    return rows, missing


# --------------------------------------------------------------------------
# regression tasks
# --------------------------------------------------------------------------

def regression():
    audit = [r for r in csv.DictReader(
        (PROJ / "results/audit_summary.csv").open(encoding="utf8"))
        if r["verdict"] == "PASS" and "_debug" not in r["run"]]
    res = {}
    for task, unit in (("crop_yield", "kg/acre"),
                       ("harvesting_date", "days"),
                       ("transplanting_date", "days")):
        runs = [r["run"] for r in audit if r["task"] == task]
        E, keys, used = [], None, []
        for run in runs:
            f = PROJ / "results/regression_gate" / f"{task}_{run}_bs32.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf8"))
            pf = d.get("per_field_abs_error")
            if not pf:
                continue
            k = list(pf.keys())
            if keys is None:
                keys = k
            elif k != keys:
                raise RuntimeError(
                    f"field-season order differs for {task}/{run}; refusing "
                    f"to stack misaligned vectors")
            E.append(np.array([float(v) for v in pf.values()]))
            used.append(run)
        if not E:
            continue
        E = np.vstack(E)
        plots = np.array([float(k.strip("()").split(",")[1]) for k in keys])
        per = E.mean(0)
        o = np.argsort(-per)
        tot = float(per.sum())
        entry = {
            "unit": unit,
            "n_checkpoints": int(E.shape[0]),
            "checkpoints": used,
            "n_field_seasons": int(E.shape[1]),
            "n_plots": int(len(np.unique(plots))),
            "quantiles": quantiles(per, (0, 10, 25, 50, 75, 90, 100)),
            "mean": float(per.mean()),
            "ci_median_plot_grouped": boot_grouped(per, plots, np.median),
            "ci_mean_plot_grouped": boot_grouped(per, plots, np.mean),
            "p90_over_median": float(np.percentile(per, 90)
                                     / max(1e-9, float(np.median(per)))),
            "worst_field_season": {"key": keys[o[0]],
                                   "abs_error": float(per[o[0]])},
        }
        for lab, k in (("decile", max(1, len(per) // 10)),
                       ("quartile", max(1, len(per) // 4))):
            entry["share_of_total_error_worst_" + lab] = {
                "n_field_seasons": k,
                "even_spread_would_be": k / len(per),
                "share": float(per[o[:k]].sum() / tot)}
        byplot = np.array([per[plots == p].mean() for p in np.unique(plots)])
        po = np.argsort(-byplot)
        kk = max(1, len(byplot) // 4)
        entry["share_of_total_error_worst_quarter_of_plots"] = {
            "n_plots": kk, "even_spread_would_be": kk / len(byplot),
            "share": float(byplot[po[:kk]].sum() / byplot.sum())}
        res[task] = entry
    return res


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def figures(lad, x, w, rel, reg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.2, "figure.dpi": 400,
                         "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 1, figsize=(3.35, 5.8), constrained_layout=True)
    ax[0].hist(x, bins=np.arange(0, 1.05, 0.05), color="#4a7fb5",
               edgecolor="white", linewidth=0.5)
    for t, c in zip(THRESHOLDS, ("#8e44ad", "#c0392b", "#e67e22")):
        ax[0].axvline(t, color=c, lw=1.0, ls="--",
                      label=f"IoU<{t:.2f}: {(x < t).mean():.0%} of fields")
    ax[0].axvline(lad["pooled_pixel_macro_iou"], color="#1f3b73", lw=1.6,
                  label=f"pooled {lad['pooled_pixel_macro_iou']:.3f}")
    ax[0].set_xlabel("field IoU (the crop the field holds)")
    ax[0].set_ylabel("fields")
    ax[0].legend(fontsize=5.4, loc="upper left")
    ax[0].set_title(f"(a) {lad['n_fields']} validation fields, {RUNG}, "
                    f"{lad['n_seeds']} seeds", fontsize=7.4)

    s = np.sort(x)
    ax[1].step(np.arange(1, len(s) + 1) / len(s), s, where="post",
               color="#1f3b73", lw=1.4)
    ax[1].axhline(lad["pooled_pixel_macro_iou"], color="#1f3b73", lw=1.0,
                  ls=":", label="pooled")
    ax[1].axhline(float(np.median(x)), color="#e67e22", lw=1.0, ls="--",
                  label="per-field median")
    ax[1].set_xlabel("cumulative share of fields")
    ax[1].set_ylabel("field IoU")
    ax[1].legend(fontsize=5.6, loc="lower right")
    ax[1].set_title("(b) quantile plot (inverse empirical CDF)",
                fontsize=7.4)

    ax[2].scatter(w, x, s=12, color="#4a7fb5", edgecolor="#1f3b73",
                  linewidth=0.4)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("pixels the field contributes to the pooled metric")
    ax[2].set_ylabel("field IoU")
    ax[2].set_title(f"(c) size vs accuracy, Spearman "
                    f"{lad['size']['spearman_weight_vs_iou']:+.2f}",
                    fontsize=7.4)
    fig.savefig(FIGDIR / "fig_field_distribution.png")
    plt.close(fig)

    have = [r for r in rel if "shift_without_largest_field" in r]
    if have:
        fig, ax = plt.subplots(figsize=(3.35, 3.0), constrained_layout=True)
        y = np.arange(len(have))
        ax.barh(y - 0.2, [r["pooled"] for r in have], height=0.38,
                color="#1f3b73", label="pooled, all fields")
        ax.barh(y + 0.2, [r["pooled_without_largest_field"] for r in have],
                height=0.38, color="#c0392b",
                label="largest field removed")
        ax.set_yticks(y)
        ax.set_yticklabels([r["run"] for r in have], fontsize=5.0)
        ax.set_xlabel("pixel-pooled macro IoU")
        ax.legend(fontsize=5.6, loc="lower right")
        ax.set_title("Effect of removing one field of 36", fontsize=7.4)
        fig.savefig(FIGDIR / "fig_leave_field_out.png")
        plt.close(fig)

    if reg:
        fig, ax = plt.subplots(figsize=(3.35, 2.5), constrained_layout=True)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":",
                label="even spread")
        for task, c in zip(reg, ("#1f3b73", "#c0392b", "#e67e22")):
            q = reg[task]
            pts = [(0.0, 0.0)]
            for lab in ("decile", "quartile"):
                d = q["share_of_total_error_worst_" + lab]
                pts.append((d["even_spread_would_be"], d["share"]))
            pts.append((1.0, 1.0))
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, lw=1.2,
                    color=c, label=task.replace("_", " "))
        ax.set_xlabel("share of field-seasons, worst first")
        ax.set_ylabel("share of total absolute error")
        ax.legend(fontsize=5.6, loc="lower right")
        ax.set_title("Who carries the regression error", fontsize=7.4)
        fig.savefig(FIGDIR / "fig_error_concentration.png")
        plt.close(fig)


# --------------------------------------------------------------------------

def main() -> int:
    size = {r["plot"]: r["n_pixels_pooled"]
            for r in json.loads((PROJ / "results/field_size_val.json")
                                .read_text(encoding="utf8"))["plots"]}

    lad, x, w, plots = ladder_crop_type(size)
    rel, missing = released_crop_type()
    reg = regression()
    bar = "=" * 74

    print(bar)
    print(f"CROP TYPE, rung {RUNG}, {lad['n_seeds']} seeds, "
          f"{lad['n_fields']} validation fields")
    print(bar)
    print(f"  pooled pixel macro IoU   {lad['pooled_pixel_macro_iou']:.4f}")
    print(f"  per-field mean           {lad['per_field_mean']:.4f}"
          f"   95% CI [{lad['ci_mean'][0]:.3f}, {lad['ci_mean'][1]:.3f}]")
    print(f"  per-field median         {lad['quantiles']['50']:.4f}"
          f"   95% CI [{lad['ci_median'][0]:.3f}, {lad['ci_median'][1]:.3f}]")
    mc = lad["macro_contrast"]
    print(f"  (macro contrast, superseded: mean {mc['per_field_mean']:.4f}, "
          f"median {mc['median']:.4f}, below 0.50 {mc['below']['0.5']}; "
          f"{mc['n_fields_macro_below_half_with_field_iou_at_least_0_9']} of "
          f"those have field IoU >= 0.9)")
    print(f"  gap                      "
          f"{lad['gap_pooled_minus_per_field']:+.4f}")
    print("  quantiles  " + "  ".join(
        f"p{q}={lad['quantiles'][q]:.3f}"
        for q in ("0", "10", "25", "50", "75", "90", "100")))
    print("  fraction below a stated threshold "
          "(none is an official criterion):")
    for t in THRESHOLDS:
        b = lad["below"][str(t)]
        print(f"    IoU < {t:.2f}   {b['n']:2d}/{lad['n_fields']} = "
              f"{b['frac']:.0%}   95% CI "
              f"[{b['ci'][0]:.0%}, {b['ci'][1]:.0%}]")
    sp = lad["seed_spread_per_field"]
    print(f"  seed-to-seed range at a single field: median "
          f"{sp['median_range_over_seeds']:.3f}, max "
          f"{sp['max_range_over_seeds']:.3f}; "
          f"{sp['n_fields_range_above_0_2']} fields move more than 0.20")
    for lab in ("worst_decile", "worst_quartile"):
        d = lad["share_of_shortfall_" + lab]
        print(f"  the {d['n_fields']:2d} worst fields "
              f"({lab.split('_')[1]}) carry {d['share']:.0%} of the "
              f"shortfall below the pooled figure")

    print()
    z = lad["size"]
    print("FIELD SIZE")
    print(f"  Spearman(size, per-field IoU) = "
          f"{z['spearman_weight_vs_iou']:+.3f}  95% CI "
          f"[{z['spearman_weight_vs_iou_ci'][0]:+.3f}, "
          f"{z['spearman_weight_vs_iou_ci'][1]:+.3f}]")
    print(f"  mean IoU, smaller half of fields {z['iou_mean_small_half']:.4f}")
    print(f"  mean IoU, larger half of fields  {z['iou_mean_large_half']:.4f}")
    print(f"  size-weighted mean of per-field IoU "
          f"{z['weighted_mean_iou']:.4f} vs unweighted "
          f"{z['unweighted_mean_iou']:.4f}  (diagnostic only)")

    print()
    print(bar)
    print("RELEASED CROP-TYPE CHECKPOINTS: is the gap universal, and what "
          "carries it?")
    print(bar)
    print(f"{'run':22s}{'pooled':>8s}{'field':>8s}{'gap':>8s}{'<0.5':>7s}"
          f"{'drop big':>10s}{'shift':>8s}")
    for r in rel:
        sh = r.get("shift_without_largest_field")
        pw = r.get("pooled_without_largest_field")
        tail = (f"{pw:>10.4f}{sh:>+8.4f}" if sh is not None
                else f"{'n/a':>10s}{'n/a':>8s}")
        print(f"{r['run']:22s}{r['pooled']:>8.4f}{r['per_field_mean']:>8.4f}"
              f"{r['gap']:>+8.4f}{r['frac_below_0.5']:>7.0%}{tail}")
    g = np.array([r["gap"] for r in rel])
    print(f"  gap: median {np.median(g):+.4f}, range "
          f"[{g.min():+.4f}, {g.max():+.4f}]; {int((g > 0).sum())}/{len(g)} "
          f"checkpoints score HIGHER pooled than per-field")
    if (g < 0).any():
        worse = ", ".join(r["run"] for r in rel if r["gap"] < 0)
        print(f"  the gap is NOT universal: {worse} scores higher per-field "
              f"than pooled, so the finding is a dispersion effect and not a "
              f"uniform optimism.")
    bad = [r["run"] for r in rel
           if r.get("recompute_matches_reference") is False]
    if bad:
        print(f"  WARNING: pooled recomputation from per-field confusions "
              f"disagrees with the reference for {', '.join(bad)}")
    sh = np.array([r["shift_without_largest_field"] for r in rel
                   if "shift_without_largest_field" in r])
    if sh.size:
        print(f"  removing the single largest field (1 of 36, holding "
              f"{100 * rel[0].get('px_share_largest_field', float('nan')):.0f}% "
              f"of scored pixels) moves the pooled figure by a median of "
              f"{np.median(np.abs(sh)):.4f} and up to "
              f"{np.abs(sh).max():.4f}, over {sh.size} checkpoints")
    if missing:
        print(f"  NOT RUN with the leave-fields-out analysis: "
              f"{', '.join(missing)}")

    print()
    print(bar)
    print("REGRESSION TASKS: per-field-season absolute error")
    print(bar)
    for task, q in reg.items():
        print(f"{task}  ({q['n_checkpoints']} checkpoints, "
              f"{q['n_field_seasons']} field-seasons at {q['n_plots']} "
              f"plots, {q['unit']})")
        print("   quantiles  " + "  ".join(
            f"p{k}={q['quantiles'][k]:.1f}"
            for k in ("0", "10", "25", "50", "75", "90", "100")))
        print(f"   median {q['quantiles']['50']:.1f}  95% CI "
              f"[{q['ci_median_plot_grouped'][0]:.1f}, "
              f"{q['ci_median_plot_grouped'][1]:.1f}]  (plots resampled)")
        print(f"   p90/median = {q['p90_over_median']:.2f}")
        for lab in ("decile", "quartile"):
            d = q["share_of_total_error_worst_" + lab]
            print(f"   worst {lab:<9s} ({d['n_field_seasons']:2d} "
                  f"field-seasons) carries {d['share']:.0%} of the summed "
                  f"error; even spread would be "
                  f"{d['even_spread_would_be']:.0%}")
        d = q["share_of_total_error_worst_quarter_of_plots"]
        print(f"   worst quarter of PLOTS ({d['n_plots']} of "
              f"{q['n_plots']}) carries {d['share']:.0%}; even spread "
              f"{d['even_spread_would_be']:.0%}")
        print(f"   largest single error: field-season "
              f"{q['worst_field_season']['key']} at "
              f"{q['worst_field_season']['abs_error']:.1f} {q['unit']}")

    OUT.write_text(json.dumps(
        {"ladder_crop_type": lad, "released_crop_type": rel,
         "released_missing_leave_fields_out": missing,
         "regression": reg,
         "threshold_note": ("0.30, 0.50 and 0.70 are stated conventions, not "
                            "official criteria. All three are reported so no "
                            "conclusion rests on one cut."),
         "bootstrap_note": ("crop-type intervals resample the 36 validation "
                            "FIELDS; regression intervals resample the 14 "
                            "PLOTS and carry each plot's field-seasons "
                            "together, because field-seasons at one plot are "
                            "not independent."),
         "n_boot": NBOOT},
        indent=2), encoding="utf8")
    figures(lad, x, w, rel, reg)
    print()
    print(f"wrote {OUT}")
    for n in ("fig_field_distribution.png", "fig_leave_field_out.png",
              "fig_error_concentration.png"):
        print(f"wrote {FIGDIR / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
