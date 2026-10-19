"""The paper's satellite-image figure: what a field score looks like on the ground.

    python scripts/make_fig_satellite.py

Writes results/figures_ieee/fig_satellite.png, a print-ready (white
background, column-width) companion to the dark gallery panels of
scripts/render_sickle_gallery.py, whose cached v5 seed-0 predictions it
reuses. Nothing is re-run on the GPU; if the cache is missing the gallery
script rebuilds it on the CPU.

Two groups of fields, chosen by rules that do not look at the picture:
  (a) typical fields: the gallery's size rule (largest field, then the
      75th, 50th and 25th percentile fields by labelled pixels);
  (b) the fields the paper counts as failing: mean field IoU over the
      completed v5 seeds below 0.5, read from results/field_confusions/v5.json
      exactly as the manuscript computes it.
For every field the chip shown is the season with the most labelled pixels,
again chosen without looking at the prediction. The number under a tile is
the field's IoU at seed 0 over all of its chips; the seed mean follows
for the failing group, since that is what selected it. The cache is checked
against the checkpoint (best.json pooled score, field_confusions seed 0) and
the checkpoint against docs/CHECKPOINT_INVENTORY.csv by SHA-256.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # the GPU is the sweep's

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

import render_sickle_gallery as G                    # noqa: E402
from src.utils import completed_field_iou            # noqa: E402

OUT = PROJ / "results" / "figures_ieee" / "fig_satellite.png"
CKPT_DIR = PROJ / "outputs" / "crop_type" / "v5_seed0"
INK, MUTED = "#111111", "#555555"
C_OK, C_BAD = "#1a7f37", "#c62828"


def trace_checkpoint() -> dict:
    """The checkpoint behind the tiles, by path and SHA-256, checked against
    docs/CHECKPOINT_INVENTORY.csv so the figure names a file that exists in
    the inventory with the same bytes."""
    import csv
    import hashlib
    f = CKPT_DIR / "checkpoint_best.pth.tar"
    rel = f.relative_to(PROJ).as_posix()
    rows = [r for r in csv.DictReader(
        (PROJ / "docs" / "CHECKPOINT_INVENTORY.csv").open(encoding="utf8"))
        if r["file"] == rel]
    assert len(rows) == 1, f"{rel}: {len(rows)} inventory rows"
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == rows[0]["sha256"], "checkpoint changed since inventory"
    return {"file": rel, "sha256": h.hexdigest(),
            "inventory_group": rows[0]["group"], "run": rows[0]["run"]}


def best_sample(R, plot, most_wrong=False):
    """The chip of this field with the most labelled pixels, or, for a field
    already selected because it fails, the chip with the most wrong pixels
    (ties broken by labelled pixels). A field's score covers every season it
    appears in, so the largest chip of a failing field can be a season the
    model got right, and would hide the failure the panel is there to show."""
    ig = R["ignore_index"]
    best, key_best = None, (-1, -1)
    for s in range(R["truth"].shape[0]):
        m = (R["truth"][s] != ig) & (R["pid"][s] == plot)
        wrong = int((m & (R["pred"][s] != R["truth"][s])).sum())
        key = (wrong, int(m.sum())) if most_wrong else (int(m.sum()), 0)
        if m.any() and key > key_best:
            best, key_best = s, key
    return best


def failing_fields():
    seeds, have = completed_field_iou("v5", PROJ)
    keys = sorted(set.intersection(*[set(d) for d in seeds]))
    mean = {k: float(np.mean([d[k] for d in seeds])) for k in keys}
    bad = sorted((k for k in keys if mean[k] < 0.5), key=lambda k: mean[k])
    return bad, mean, have


def window(R, s, plot, lo=14):
    """A square crop centred on the field: the field plus a margin, at
    least `lo` pixels (140 m) and at most the whole 32-pixel chip. Small
    fields are otherwise two or three pixels in a 320 m chip and cannot be
    seen at column width. Every tile keeps its own 100 m bar."""
    ig = R["ignore_index"]
    H, W = R["truth"][s].shape
    ys, xs = np.nonzero((R["truth"][s] != ig) & (R["pid"][s] == plot))
    w = int(min(H, max(lo, max(np.ptp(ys), np.ptp(xs)) + 1 + 8)))
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    y0 = int(np.clip(cy - w // 2, 0, H - w))
    x0 = int(np.clip(cx - w // 2, 0, W - w))
    return slice(y0, y0 + w), slice(x0, x0 + w)


def tile(ax, R, s, plot, mode, scale=8):
    """White-page version of the gallery tile, cropped to the field."""
    from matplotlib.colors import to_rgb
    ig = R["ignore_index"]
    yy, xx = window(R, s, plot)
    rgb = R["rgb"][s][yy, xx].astype(np.float32)
    truth = R["truth"][s][yy, xx]
    valid = truth != ig
    target = valid & (R["pid"][s][yy, xx] == plot)
    others = valid & ~target
    img = rgb
    if mode == "pred":
        pred = R["pred"][s][yy, xx]
        col = np.zeros_like(rgb)
        col[pred == G.PADDY] = to_rgb(G.C_PADDY)
        col[pred == G.NON_PADDY] = to_rgb(G.C_OTHER)
        col[valid & (pred != truth)] = to_rgb(G.C_WRONG)
        alpha = np.where(valid, 0.85, 0.28)[..., None]
        img = rgb * (1 - alpha) + col * alpha
    up = np.kron(img, np.ones((scale, scale, 1)))
    ax.imshow(up, interpolation="nearest")
    if others.any():
        G._outline(ax, others, scale, color="#bdbdbd", lw=0.5)
    G._outline(ax, target, scale, color="#ffffff", lw=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999"); sp.set_linewidth(0.4)
    h = up.shape[0]
    ax.plot([4, 4 + 10 * scale], [h - 6, h - 6], color="white", lw=1.4)
    if mode == "image":
        ax.text(4 + 5 * scale, h - 10, "100 m", color="white", fontsize=5.6,
                ha="center", va="bottom")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import _vector_figs  # noqa: F401,E402  (vector PDF beside every PNG)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    R = G.run_rung("v5", 0)
    R["own_iou"] = G.own_class_iou(R)
    typical = [f["plot"] for f in G.pick_fields(R)]
    bad, mean, seeds = failing_fields()
    if not bad:
        raise SystemExit("no field below 0.5 at v5; the figure's group (b) "
                         "would be empty")
    # Traceability. The cache must be the v5 seed-0 checkpoint's own output:
    # its pooled score equals the one the trainer saved in best.json, and its
    # field IoUs equal seed 0 of the confusion file the paper counts from.
    # Anything else and the tiles would show a different run from the text.
    best = json.loads((CKPT_DIR / "best.json").read_text(encoding="utf8"))
    assert abs(R["pooled_macro_iou"] - best["macro_iou"]) < 1e-9, \
        (R["pooled_macro_iou"], best["macro_iou"])
    s0 = completed_field_iou("v5", PROJ)[0][0]
    dev = max(abs(s0[float(k)] - v) for k, v in R["own_iou"].items())
    assert dev < 1e-6 and len(R["own_iou"]) == len(s0), \
        f"seed-0 field IoU differs from field_confusions by {dev:.4g}"
    ckpt = trace_checkpoint()

    groups = [("(a) Typical fields, chosen by size", typical, False),
              (f"(b) The {len(bad)} fields scoring below 0.5", bad, True)]
    rows = sum((len(g[1]) + 1) // 2 for g in groups)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    fig = plt.figure(figsize=(3.3, 1.02 * rows + 0.95), facecolor="white")
    gs = fig.add_gridspec(rows + len(groups), 4,
                          height_ratios=sum(([0.09] + [1] * ((len(g[1]) + 1) // 2)
                                             for g in groups), []),
                          left=0.01, right=0.99, top=0.995, bottom=0.135,
                          wspace=0.05, hspace=0.62)
    r = 0
    for title, plots, is_bad in groups:
        ax = fig.add_subplot(gs[r, :]); ax.axis("off")
        ax.text(0, -2.6, title, fontsize=7.0, weight="bold", color=INK,
                va="center")   # into the gap, nearer its tiles
        r += 1
        for i, plot in enumerate(plots):
            rr, cc = r + i // 2, 2 * (i % 2)
            s = best_sample(R, plot, most_wrong=is_bad)
            a1 = fig.add_subplot(gs[rr, cc])
            a2 = fig.add_subplot(gs[rr, cc + 1])
            tile(a1, R, s, plot, "image")
            tile(a2, R, s, plot, "pred")
            cls, iou = G._field_label(R, {"plot": plot, "sample": s})
            # One two-line label centred under the pair, so neither line
            # has to fit under a single 0.8-inch tile.
            mid = 1 + 0.05 / 2          # halfway across the wspace gap
            a1.text(mid, -0.05, f"field {int(plot)} ({cls})", fontsize=6.2,
                    color=INK, ha="center", va="top", transform=a1.transAxes)
            lab = f"IoU {iou:.2f}"
            # (b) was selected on the seed mean, so that decides its colour
            score = mean[float(plot)] if is_bad else iou
            if is_bad:
                lab += f"; {len(seeds)}-seed mean {mean[float(plot)]:.2f}"
            a1.text(mid, -0.20, lab, fontsize=6.2, weight="bold",
                    ha="center", va="top", transform=a1.transAxes,
                    color=C_OK if score >= 0.5 else C_BAD)
        r += (len(plots) + 1) // 2
    leg = [Patch(facecolor=G.C_PADDY, label="predicted paddy"),
           Patch(facecolor=G.C_OTHER, label="predicted non-paddy"),
           Patch(facecolor=G.C_WRONG, label="wrong, inside surveyed field"),
           Patch(facecolor="#9e9e9e", edgecolor="white",
                 label="surveyed field (outline)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=6.2,
               frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=1.2, columnspacing=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    meta = {
        "rung": "v5", "seed_shown": 0, "seeds_for_selection": seeds,
        "checkpoint": ckpt,
        "pooled_macro_iou": R["pooled_macro_iou"],
        "pooled_macro_iou_in_best_json": best["macro_iou"],
        "max_abs_diff_seed0_field_iou_vs_field_confusions": dev,
        "typical_fields_rule": "largest, then 75th/50th/25th percentile "
                               "by labelled pixels (render_sickle_gallery)",
        "typical_fields": [int(p) for p in typical],
        "failing_fields_rule": "mean field IoU over completed seeds < 0.5 "
                               "(as the manuscript)",
        "failing_fields": {str(int(k)): mean[k] for k in bad},
        "seed0_field_iou": {str(int(float(k))): v for k, v in
                            R["own_iou"].items()
                            if float(k) in set(typical) | set(bad)},
        "chip_rule": "(a) the field's chip with the most labelled pixels; "
                     "(b) the chip with the most wrong pixels at seed 0; "
                     "both cropped to the field plus margin (14-32 px)",
    }
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2),
                                        encoding="utf8")
    print(f"wrote {OUT} ({len(typical)} typical, {len(bad)} failing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
