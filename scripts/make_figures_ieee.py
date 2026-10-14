"""Column-width figures for the two-column IEEE layout.

An IEEE column is about 3.4 inches. A figure drawn for a 6.5-inch page and
then scaled down loses its axis labels. These are drawn at column width with
type sized for it, so nothing is illegible after placement.
"""
from __future__ import annotations
import csv, glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJ))
from src.utils import completed_per_plot_iou  # noqa: E402,F401
# Every per-field figure uses FIELD IoU (the crop the field holds), not the
# trainer's macro IoU; see src/evaluation/metrics.py for why.
from src.utils import completed_field_iou  # noqa: E402
OUT = PROJ / "results" / "figures_ieee"
OUT.mkdir(parents=True, exist_ok=True)
W = 3.35                                    # one IEEE column, inches
plt.rcParams.update({"font.size": 7.2, "figure.dpi": 400,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.titlesize": 7.6, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                     "axes.labelsize": 7.0, "lines.linewidth": 1.4,
                     "savefig.pad_inches": 0.02})


def _pooled(rung):
    for r in csv.DictReader((PROJ/"results/ladder_crop_type.csv")
                            .open(encoding="utf8")):
        if r["rung"] == rung:
            return float(r["pixel_macro_iou"])
    raise KeyError(rung)


def f_perfield():
    seeds, _ = completed_field_iou("v5", PROJ)
    v = np.sort(np.array([np.mean([d[k] for d in seeds]) for k in seeds[0]]))
    pooled = _pooled("v5")
    fig, ax = plt.subplots(figsize=(W, 2.25))
    ax.bar(range(len(v)), v, width=0.84,
           color=["#c0392b" if x < 0.5 else "#2e7d32" for x in v])
    # a field scoring exactly 0 has no visible bar; mark it so the red
    # count in the figure matches the text
    zeros = [i for i, x in enumerate(v) if x == 0]
    if zeros:
        ax.scatter(zeros, [0.025] * len(zeros), marker="x", s=22,
                   color="#c0392b", lw=1.4, zorder=3)
    ax.axhline(pooled, color="#1f3b73", lw=1.5,
               label=f"pixel-pooled IoU {pooled:.4f}")
    # the median, not the mean: it is the statistic the text quotes, and a
    # figure disagreeing with the prose by 0.003 invites a query it cannot
    # answer
    ax.axhline(np.median(v), color="#e67e22", lw=1.5, ls="--",
               label=f"median field {np.median(v):.3f}")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(f"validation fields, sorted (n={len(v)})")
    ax.set_ylabel("field IoU"); ax.set_ylim(0, 1.05)
    # above the axes: most fields score near 1.0, so any legend inside the
    # plot covers bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False)
    fig.savefig(OUT/"fig_perfield.png"); plt.close(fig)
    return int((v < 0.5).sum()), len(v)


def f_ladder():
    rows = [r for r in csv.DictReader(
        (PROJ/"results/ladder_crop_type.csv").open(encoding="utf8"))
        if r.get("pixel_macro_iou")]
    order = [f"v{i}" for i in range(1, 11)]
    rows.sort(key=lambda r: order.index(r["rung"]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.errorbar(x, [float(r["pixel_macro_iou"]) for r in rows],
                yerr=[float(r["pixel_sd"]) for r in rows], marker="o", ms=3,
                capsize=2, color="#1f3b73", label="pixel-pooled")
    # per-field series: mean field IoU per seed, then mean and sd over the
    # same balanced seeds the pooled series uses
    fm, fs = [], []
    for r in rows:
        dicts, sids = completed_field_iou(r["rung"], PROJ)
        keep = {int(x) for x in r["balanced_seeds"].split()}
        per_seed = [float(np.mean(list(d.values())))
                    for d, sid in zip(dicts, sids) if sid in keep]
        assert len(per_seed) == len(keep), r["rung"]
        fm.append(float(np.mean(per_seed)))
        fs.append(float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0)
    ax.errorbar(x, fm, yerr=fs, marker="s", ms=3,
                capsize=2, color="#e67e22", label="mean field IoU")
    ax.axvspan(5.5, 9.5, color="grey", alpha=0.10)
    ax.set_xticks(x); ax.set_xticklabels([r["rung"] for r in rows], rotation=45)
    ax.set_ylabel("IoU"); ax.set_xlabel("ablation rung")
    ax.legend(loc="lower right")
    fig.savefig(OUT/"fig_ladder.png"); plt.close(fig)


def f_tails():
    # Same exclusion as the manuscript: *_debug are 2-epoch smoke tests, not
    # released models. Without this the bars disagree with Table I.
    rows = [r for r in csv.DictReader(
        (PROJ/"results/audit_summary.csv").open(encoding="utf8"))
        if r["verdict"] == "PASS" and "_debug" not in r["run"]]
    tasks = [("crop_yield", "yield (kg)"), ("harvesting_date", "harvest (d)"),
             ("transplanting_date", "transpl. (d)")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95),
                             constrained_layout=True)
    for ax, (t, lab) in zip(axes, tasks):
        rs = [r for r in rows if r["task"] == t]
        g = lambda k: np.median([float(r[k]) for r in rs])
        ax.bar(["med", "p75", "p90"], [g("median"), g("p75"), g("p90")],
               color=["#2e7d32", "#f0ad4e", "#c0392b"], width=0.65)
        ax.axhline(g("agg_mae"), color="#1f3b73", lw=1.3)
        ax.set_title(lab, fontsize=6.8)
        ax.tick_params(labelsize=5.4, pad=1.2)
        ax.margins(y=0.08)
    axes[0].set_ylabel("abs. error", fontsize=6.6)
    fig.savefig(OUT/"fig_tails.png"); plt.close(fig)


def f_batch():
    fig, ax = plt.subplots(figsize=(W*0.72, 1.85))
    bs, iou = [8, 16, 32], [0.6409, 0.6561, 0.7071]
    ax.plot(bs, iou, marker="o", ms=4, color="#c0392b")
    for b, v in zip(bs, iou):
        ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6)
    ax.set_xscale("log", base=2); ax.set_xticks(bs)
    ax.set_xticklabels(bs); ax.set_ylim(0.60, 0.745)
    ax.set_xlabel("evaluation batch size"); ax.set_ylabel("macro IoU")
    fig.savefig(OUT/"fig_batch.png"); plt.close(fig)


def f_seeds():
    per, _ = completed_field_iou("v5", PROJ)
    common = sorted(set.intersection(*[set(d) for d in per]))
    a = np.array([per[0][p] for p in common]); b = np.array([per[1][p] for p in common])
    fig, ax = plt.subplots(figsize=(W*0.72, 1.95))
    ax.scatter(a, b, s=12, color="#1f3b73", alpha=0.8, edgecolor="white", lw=0.3)
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
    ax.axhline(0.5, color="#c0392b", lw=0.8, ls="--")
    ax.axvline(0.5, color="#c0392b", lw=0.8, ls="--")
    ax.set_xlabel("field IoU, seed 0"); ax.set_ylabel("seed 1")
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
    fig.savefig(OUT/"fig_seeds.png"); plt.close(fig)
    return float(np.corrcoef(a, b)[0, 1])


def f_triage():
    d = json.loads((PROJ/"results/triage/triage_v5.json").read_text(encoding="utf8"))
    fig, ax = plt.subplots(figsize=(W, 2.15))
    for k, col, mk, lab in (("seed_disagreement", "#1f3b73", "o", "seed disagreement"),
                            ("low_confidence", "#e67e22", "s", "low confidence")):
        r = sorted(d[k]["screening"], key=lambda z: z["workload"])
        ax.plot([z["workload"] for z in r], [z["recall"] for z in r],
                marker=mk, ms=3.2, color=col,
                label=f"{lab} (r={d[k]['corr_with_iou']:+.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8, label="random")
    ax.set_xlabel("fields referred (workload)"); ax.set_ylabel("unreliable caught")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right")
    fig.savefig(OUT/"fig_triage.png"); plt.close(fig)


if __name__ == "__main__":
    nbad, n = f_perfield(); f_ladder(); f_tails(); f_batch()
    r = f_seeds(); f_triage()
    print(f"  per-field: {nbad}/{n} below 0.5 | seed corr {r:+.3f}")
    print(f"  wrote {len(list(OUT.glob('*.png')))} column-width figures")
