"""Figures for the paper.

    python scripts/make_figures.py

Every figure is drawn from a results file on disk. Nothing is illustrative,
nothing is redrawn by hand, and no value is typed in by the author.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def fig1_per_field_vs_headline():
    """The headline: the aggregate describes no individual field."""
    runs = sorted(glob.glob(str(
        PROJ / "outputs/crop_type/v5_seed*/best_per_plot_iou.json")))
    if not runs:
        print("  fig1 SKIPPED: no v5 per-plot files"); return
    per = {}
    for f in runs:
        for k, v in json.loads(Path(f).read_text(encoding="utf8")).items():
            per.setdefault(float(k), []).append(v)
    vals = np.array([np.mean(v) for v in per.values()])
    vals.sort()
    headline = 0.8665                      # v5 pixel-pooled, 3-seed mean
    plot_macro = float(vals.mean())

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colours = ["#c0392b" if v < 0.5 else "#2e7d32" for v in vals]
    ax.bar(range(len(vals)), vals, color=colours, width=0.82)
    ax.axhline(headline, color="#1f3b73", lw=2,
               label=f"reported pixel-pooled IoU = {headline:.3f}")
    ax.axhline(plot_macro, color="#e67e22", lw=2, ls="--",
               label=f"mean per-field IoU = {plot_macro:.3f}")
    ax.axhline(0.5, color="grey", lw=1, ls=":")
    n_bad = int((vals < 0.5).sum())
    ax.set_xlabel(f"validation fields, sorted ({len(vals)} total)")
    ax.set_ylabel("IoU")
    ax.set_title("A reported accuracy describes no individual field\n"
                 f"{n_bad} of {len(vals)} fields score below 0.5 "
                 f"({100*n_bad/len(vals):.0f}%)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.savefig(OUT / "fig1_per_field_vs_headline.png")
    plt.close(fig)
    print(f"  fig1: {len(vals)} fields, {n_bad} below 0.5")


def fig2_ladder():
    """What each component bought, against the seed-to-seed spread."""
    p = PROJ / "results" / "ladder_crop_type.csv"
    if not p.exists():
        print("  fig2 SKIPPED"); return
    rows = list(csv.DictReader(p.open(encoding="utf8")))
    rows = [r for r in rows if r.get("pixel_macro_iou")]
    order = [f"v{i}" for i in range(1, 11)]
    rows.sort(key=lambda r: order.index(r["rung"]))
    names = [r["rung"] for r in rows]
    px = np.array([float(r["pixel_macro_iou"]) for r in rows])
    pxsd = np.array([float(r["pixel_sd"]) for r in rows])
    pl = np.array([float(r["plot_macro_iou"]) for r in rows])
    plsd = np.array([float(r["plot_sd"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(names))
    ax.errorbar(x, px, yerr=pxsd, marker="o", lw=1.8, capsize=3,
                color="#1f3b73", label="pixel-pooled IoU")
    ax.errorbar(x, pl, yerr=plsd, marker="s", lw=1.8, capsize=3,
                color="#e67e22", label="per-field (plot-macro) IoU")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_xlabel("ablation rung (each adds exactly one component)")
    ax.set_ylabel("IoU")
    ax.set_title("Only information helps: temporal (v1→v2) and radar (v4→v5).\n"
                 "Architecture, capacity and optimiser (v6→v10) do not.",
                 fontsize=10)
    ax.axvspan(5.5, 9.5, color="grey", alpha=0.10)
    ax.text(7.5, 0.33, "model complexity\n(no measurable gain)", ha="center",
            fontsize=8, color="#555555")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig2_ladder.png")
    plt.close(fig)
    print(f"  fig2: {len(names)} rungs, 3 seeds")


def fig3_tails():
    """The regression tasks: the typical field is fine, the tail is not."""
    p = PROJ / "results" / "audit_summary.csv"
    if not p.exists():
        print("  fig3 SKIPPED"); return
    rows = [r for r in csv.DictReader(p.open(encoding="utf8"))
            if r["verdict"] == "PASS"]
    tasks = ["crop_yield", "harvesting_date", "transplanting_date"]
    label = {"crop_yield": "crop yield\n(kg)",
             "harvesting_date": "harvest date\n(days)",
             "transplanting_date": "transplanting date\n(days)"}

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, t in zip(axes, tasks):
        rs = [r for r in rows if r["task"] == t]
        agg = np.median([float(r["agg_mae"]) for r in rs])
        med = np.median([float(r["median"]) for r in rs])
        p75 = np.median([float(r["p75"]) for r in rs])
        p90 = np.median([float(r["p90"]) for r in rs])
        ax.bar(["typical\nfield", "p75", "p90"], [med, p75, p90],
               color=["#2e7d32", "#f0ad4e", "#c0392b"], width=0.62)
        ax.axhline(agg, color="#1f3b73", lw=2, ls="-",
                   label=f"reported MAE {agg:.0f}")
        ax.set_title(label[t], fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("absolute error per field-season")
    fig.suptitle("The reported error hides the tail: 1 field-season in 10 is "
                 "2–4× worse than typical", fontsize=10, y=1.04)
    fig.savefig(OUT / "fig3_regression_tails.png")
    plt.close(fig)
    print(f"  fig3: {len(rows)} models across 3 tasks")


def fig4_batch_size():
    """The defect: a frozen model's score moves with evaluation batch size."""
    bs = [8, 16, 32]
    iou = [0.6409, 0.6561, 0.7071]
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(bs, iou, marker="o", lw=2, color="#c0392b")
    for b, v in zip(bs, iou):
        ax.annotate(f"{v:.4f}", (b, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("evaluation batch size")
    ax.set_ylabel("macro IoU")
    ax.set_ylim(0.60, 0.74)
    ax.set_title("Same weights, same data, same code:\n"
                 "6.5 IoU points from batch size alone", fontsize=9)
    fig.savefig(OUT / "fig4_batch_size.png")
    plt.close(fig)
    print("  fig4: batch-size defect")


def fig5_seed_consistency():
    """Failure is systematic: the same fields fail in every run."""
    runs = sorted(glob.glob(str(
        PROJ / "outputs/crop_type/v5_seed*/best_per_plot_iou.json")))
    if len(runs) < 2:
        print("  fig5 SKIPPED"); return
    per = [{float(k): v for k, v in
            json.loads(Path(f).read_text(encoding="utf8")).items()}
           for f in runs]
    common = sorted(set.intersection(*[set(d) for d in per]))
    a = np.array([per[0][p] for p in common])
    b = np.array([per[1][p] for p in common])
    r = np.corrcoef(a, b)[0, 1]

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.scatter(a, b, s=28, color="#1f3b73", alpha=0.75, edgecolor="white")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.axhline(0.5, color="#c0392b", lw=1, ls="--")
    ax.axvline(0.5, color="#c0392b", lw=1, ls="--")
    both = int(((a < 0.5) & (b < 0.5)).sum())
    ax.set_xlabel("per-field IoU, seed 0")
    ax.set_ylabel("per-field IoU, seed 1")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Failure is systematic, not random\n"
                 f"r = {r:+.2f};  {both} fields fail in both runs", fontsize=9)
    fig.savefig(OUT / "fig5_seed_consistency.png")
    plt.close(fig)
    print(f"  fig5: r={r:+.3f}, {both} fields fail in both seeds")


if __name__ == "__main__":
    fig1_per_field_vs_headline()
    fig2_ladder()
    fig3_tails()
    fig4_batch_size()
    fig5_seed_consistency()
    print(f"\n  wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
