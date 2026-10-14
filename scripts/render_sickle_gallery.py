"""Version-by-version prediction panels for the ablation ladder (v1..v10).

    python scripts/render_sickle_gallery.py            # all ten rungs, seed 0
    python scripts/render_sickle_gallery.py --rungs v1 v10

For each rung's seed-0 checkpoint this runs the validation split on the CPU
(the GPU belongs to the data-regime sweep), at the pinned evaluation batch
size the metric depends on, and draws:

  ladder_<rung>.png     four validation fields, each as a pair: the
                        Sentinel-2 chip with the surveyed plot outlined, and
                        the rung's paddy / non-paddy prediction over it
  ladder_montage.png    the same four fields across all ten rungs
  ladder_field_iou.png  every validation field's IoU at every rung, beside
                        the pooled headline

WHAT A PANEL CAN AND CANNOT SHOW
--------------------------------
A SICKLE sample is a 32 x 32 chip at 10 m (320 m across). Only the surveyed
plot inside it carries a label, about 3 % of the chip, so the model's
prediction elsewhere is drawn faintly and is NOT scored. The fields shown are
chosen by size (the largest field and the 75th, 50th and 25th percentile
fields by labelled pixels), never by how well any rung did on them.

WHICH PER-FIELD SCORE IS SHOWN
------------------------------
35 of the 36 validation fields hold one crop in every season. For such a
field, macro IoU over {paddy, non-paddy} is 1.0 when every pixel is right and
at most 0.5 once a single pixel is predicted as the absent class, whose IoU
then becomes 0. That step, not field failure, drives most of the low macro
values. The panels therefore show the IoU of the field's own crop (for the
one mixed field, the mean over the classes it holds). The macro values are
still recomputed, checked against the best_per_plot_iou.json each run wrote
during training, and recorded in the manifest; any disagreement is printed,
not hidden. Training evaluated on the GPU with mixed
precision and this runs in float32 on the CPU, so tiny differences are
possible.

Writes results/gallery/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # never touch the GPU

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import patches, utils                       # noqa: E402
from src.dataset import build_loader                 # noqa: E402
from src.evaluation.metrics import (crop_type_metrics,  # noqa: E402
                                    plot_macro_iou)
from src.model import build_model                    # noqa: E402
from src.train import to_device                      # noqa: E402

OUT = PROJ / "results" / "gallery"
RUNGS = [f"v{i}" for i in range(1, 11)]
PADDY, NON_PADDY = 0, 1
BG, FG, MUTED = "#0f1110", "#f2f2ee", "#a7aca6"
C_PADDY, C_OTHER, C_WRONG = "#2ecc71", "#f5a524", "#ef4444"


# ------------------------------------------------------------- inference
def run_rung(rung: str, seed: int = 0) -> dict:
    cache = OUT / f"cache_{rung}_seed{seed}.npz"
    meta_p = OUT / f"cache_{rung}_seed{seed}.json"
    if cache.exists() and meta_p.exists():
        z = np.load(cache, allow_pickle=False)
        return {**{k: z[k] for k in z.files},
                **json.loads(meta_p.read_text(encoding="utf8"))}

    patches.install()
    cfg = utils.load_config(rung)
    cfg["training"]["num_workers"] = 0
    utils.set_seed(seed)
    device = "cpu"
    task, ig = cfg["data"]["task"], cfg["data"]["ignore_index"]
    run = utils.run_dir(cfg, seed)
    ckpt = run / "checkpoint_best.pth.tar"
    if not ckpt.exists():
        raise SystemExit(f"{rung}: no checkpoint at {ckpt} -- NOT RUN")

    loader, df = build_loader(cfg, cfg["data"]["eval_split"], "eval")
    model, _ = build_model(cfg, device)
    state = torch.load(ckpt, map_location=torch.device(device),
                       weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    s2_bands = cfg["satellites"]["S2"]["rgb_bands"]

    preds, truths, pids, prob, rgbs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, masks = to_device(batch, device)
            out = nn.Softmax(dim=1)(model(inputs))
            preds.append(out.argmax(1).numpy().astype(np.int8))
            prob.append(out[:, PADDY].numpy().astype(np.float16))
            truths.append(masks[task].long().numpy().astype(np.int16))
            pids.append(masks["plot_mask"].numpy().astype(np.float32))
            imgs = inputs["S2"][0].numpy()            # B, T, C, H, W
            for i in range(imgs.shape[0]):
                rgbs.append(_clear_rgb(imgs[i], s2_bands))
    pred = np.concatenate(preds)
    truth = np.concatenate(truths)
    pid = np.concatenate(pids)
    p_paddy = np.concatenate(prob)
    rgb = np.stack(rgbs).astype(np.float16)

    pooled = crop_type_metrics(truth.ravel(), pred.ravel(), ignore_index=ig)
    summ, per_plot = plot_macro_iou(truth.ravel(), pred.ravel(), pid.ravel(),
                                    ignore_index=ig, return_per_plot=True)
    saved = json.loads((run / "best_per_plot_iou.json").read_text(
        encoding="utf8"))
    best = json.loads((run / "best.json").read_text(encoding="utf8"))
    diffs = [abs(per_plot[float(k)] - v) for k, v in saved.items()
             if float(k) in per_plot]
    meta = {
        "rung": rung, "seed": seed,
        "description": cfg["project"]["description"],
        "ignore_index": ig,
        "pooled_macro_iou": float(pooled.macro_iou),
        "pooled_macro_iou_saved": float(best["macro_iou"]),
        "plot_macro_iou": float(summ["plot_macro_iou"]),
        "per_plot_iou": {str(k): v for k, v in per_plot.items()},
        "max_abs_diff_vs_saved_per_plot": float(max(diffs)) if diffs else None,
        "n_plots_compared": len(diffs),
        "batch_size": int(cfg["training"]["batch_size"]),
        "device": device,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pred=pred, truth=truth, pid=pid,
                        p_paddy=p_paddy, rgb=rgb)
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf8")
    return {"pred": pred, "truth": truth, "pid": pid, "p_paddy": p_paddy,
            "rgb": rgb, **meta}


def _clear_rgb(seq: np.ndarray, rgb_bands) -> np.ndarray:
    """A displayable Sentinel-2 RGB from one sample's sequence.

    Padding frames (all zero) are skipped. Clouds are bright in the blue band,
    so among real frames the one at the 20th percentile of mean blue is used:
    clearer than the median frame, without picking a dark shadowed one.
    """
    real = [t for t in range(seq.shape[0]) if np.any(seq[t] != 0)]
    if not real:
        return np.zeros((seq.shape[-2], seq.shape[-1], 3), np.float32)
    blue = np.array([seq[t][rgb_bands[2]].mean() for t in real])
    t = real[int(np.argsort(blue)[int(0.2 * (len(real) - 1))])]
    img = np.stack([seq[t][b] for b in rgb_bands], axis=-1).astype(np.float32)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img * 0


def own_class_iou(R: dict) -> dict:
    """Per field: IoU of the class(es) the field actually holds."""
    ig = R["ignore_index"]
    t, p, g = R["truth"].ravel(), R["pred"].ravel(), R["pid"].ravel()
    k = t != ig
    t, p, g = t[k], p[k], g[k]
    out = {}
    for f in np.unique(g):
        s = g == f
        tt, pp = t[s], p[s]
        vals = [np.sum((tt == c) & (pp == c)) / np.sum((tt == c) | (pp == c))
                for c in np.unique(tt)]
        out[str(float(f))] = float(np.mean(vals))
    return out


# ------------------------------------------------------------- selection
def pick_fields(R: dict, k: int = 4):
    """Fields by SIZE only: largest, then the 75th, 50th, 25th percentile.

    For each, the sample (season) with the most labelled pixels is shown.
    Independent of every rung's predictions, so no rung is flattered.
    """
    ig = R["ignore_index"]
    n_pix = {}
    best_sample = {}
    for s in range(R["truth"].shape[0]):
        valid = R["truth"][s] != ig
        if not valid.any():
            continue
        p = float(np.median(R["pid"][s][valid]))
        n = int(valid.sum())
        n_pix[p] = n_pix.get(p, 0) + n
        if n > best_sample.get(p, (-1, -1))[0]:
            best_sample[p] = (n, s)
    order = sorted(n_pix, key=lambda p: (-n_pix[p], p))
    qs = [0.0, 0.25, 0.5, 0.75][:k]
    chosen = []
    for q in qs:
        p = order[min(len(order) - 1, int(round(q * (len(order) - 1))))]
        if p not in chosen:
            chosen.append(p)
    total = sum(n_pix.values())
    return [{"plot": p, "sample": best_sample[p][1],
             "px_share": n_pix[p] / total} for p in chosen]


# ------------------------------------------------------------- drawing
def _outline(ax, valid, scale, color=FG, lw=1.6):
    from matplotlib.collections import LineCollection
    H, W = valid.shape
    segs = []
    for y in range(H):
        for x in range(W):
            if not valid[y, x]:
                continue
            if y == 0 or not valid[y - 1, x]:
                segs.append([(x, y), (x + 1, y)])
            if y == H - 1 or not valid[y + 1, x]:
                segs.append([(x, y + 1), (x + 1, y + 1)])
            if x == 0 or not valid[y, x - 1]:
                segs.append([(x, y), (x, y + 1)])
            if x == W - 1 or not valid[y, x + 1]:
                segs.append([(x + 1, y), (x + 1, y + 1)])
    segs = [[(a * scale - 0.5, b * scale - 0.5) for a, b in s] for s in segs]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def _tile(ax, R, s, mode, scale=8, plot=None):
    """mode 'image': the chip with the plot outlined; 'pred': prediction.

    A chip can hold more than one surveyed plot. The field named in the
    title is outlined in white; any other labelled plot in the chip is
    outlined thinly in grey. Both are scored, so both get the strong overlay.
    """
    from matplotlib.colors import to_rgb
    ig = R["ignore_index"]
    rgb = R.get("rgb_display", R["rgb"])[s].astype(np.float32)
    valid = R["truth"][s] != ig
    target = valid if plot is None else valid & (R["pid"][s] == plot)
    others = valid & ~target
    up = np.kron(rgb, np.ones((scale, scale, 1)))
    if mode == "pred":
        pred = R["pred"][s]
        col = np.zeros_like(rgb)
        col[pred == PADDY] = to_rgb(C_PADDY)
        col[pred == NON_PADDY] = to_rgb(C_OTHER)
        wrong = valid & (pred != R["truth"][s])
        col[wrong] = to_rgb(C_WRONG)
        alpha = np.where(valid, 0.78, 0.30)[..., None]
        mix = rgb * (1 - alpha) + col * alpha
        up = np.kron(mix, np.ones((scale, scale, 1)))
    ax.imshow(up, interpolation="nearest")
    if others.any():
        _outline(ax, others, scale, color="#8a8f8a", lw=0.8)
    _outline(ax, target, scale, color=FG if mode == "image" else "#ffffff")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333833")
    # 100 m scale bar: 10 pixels at 10 m
    h = up.shape[0]
    ax.plot([6, 6 + 10 * scale], [h - 8, h - 8], color=FG, lw=2.2)
    ax.text(6 + 5 * scale, h - 13, "100 m", color=FG, fontsize=6.5,
            ha="center", va="bottom")


def _field_label(R, f):
    ig = R["ignore_index"]
    s = f["sample"]
    t = R["truth"][s]
    valid = (t != ig) & (R["pid"][s] == f["plot"])
    cls = "paddy" if np.mean(t[valid] == PADDY) >= 0.5 else "non-paddy"
    iou = R["own_iou"].get(str(float(f["plot"])))
    return cls, iou


def render_rung(R, fields, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.2, 7.4), facecolor=BG)
    fig.suptitle(f"SICKLE — Paddy Mapping ({R['rung']})",
                 color=FG, fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.925, R["description"] + "   ·   seed 0",
             color=MUTED, fontsize=10, ha="center")
    gs = fig.add_gridspec(2, 4, left=0.02, right=0.775, top=0.86,
                          bottom=0.04, wspace=0.05, hspace=0.30)
    for i, f in enumerate(fields):
        r, c = divmod(i, 2)
        ax1 = fig.add_subplot(gs[r, 2 * c])
        ax2 = fig.add_subplot(gs[r, 2 * c + 1])
        _tile(ax1, R, f["sample"], "image", plot=f["plot"])
        _tile(ax2, R, f["sample"], "pred", plot=f["plot"])
        cls, iou = _field_label(R, f)
        ax1.set_title(f"field {int(f['plot'])} · surveyed {cls}",
                      color=FG, fontsize=9.5, loc="left")
        tone = (C_PADDY if iou is not None and iou >= 0.5 else C_WRONG)
        ax2.set_title(f"{R['rung']} · {cls} IoU "
                      f"{iou:.2f}" if iou is not None else "no score",
                      color=tone, fontsize=9.5, loc="left", weight="bold")
    # summary box, in the XRayEarth manner
    vals = np.array(list(R["own_iou"].values()))
    lines = [f"validation fields  {len(vals)}",
             f"headline (pooled)  {R['pooled_macro_iou']:.3f}",
             f"mean field IoU     {vals.mean():.3f}",
             f"median field IoU   {np.median(vals):.3f}",
             f"fields below 0.50  {int((vals < 0.5).sum())}",
             f"worst field        {vals.min():.2f}"]
    fig.text(0.79, 0.80, "\n".join(lines), color=FG, fontsize=9.6,
             family="DejaVu Sans Mono", va="top",
             bbox=dict(facecolor="#1a1d1b", edgecolor="#3a3f3b",
                       boxstyle="round,pad=0.6"))
    leg = [Patch(facecolor=C_PADDY, label="predicted paddy"),
           Patch(facecolor=C_OTHER, label="predicted non-paddy"),
           Patch(facecolor=C_WRONG, label="wrong, inside surveyed plot"),
           Patch(facecolor="none", edgecolor=FG,
                 label="the field named above"),
           Patch(facecolor="none", edgecolor="#8a8f8a",
                 label="other surveyed plot in chip")]
    fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.787, 0.30),
               frameon=True, facecolor="#1a1d1b", edgecolor="#3a3f3b",
               labelcolor=FG, fontsize=9)
    fig.text(0.79, 0.24,
             "Each chip is 320 m across at\n10 m per pixel. Only the\n"
             "outlined plot is labelled;\npredictions outside it are\n"
             "drawn faintly and not scored.\nFields chosen by size, not\n"
             "by result. Field scores are\nthe IoU of the field's own\n"
             "crop.", color=MUTED, fontsize=8.5, va="top")
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


def render_montage(Rs, fields, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rungs = [R["rung"] for R in Rs]
    fig, axes = plt.subplots(len(fields), len(Rs) + 1,
                             figsize=(1.32 * (len(Rs) + 1), 1.55 * len(fields)),
                             facecolor=BG)
    for r, f in enumerate(fields):
        _tile(axes[r, 0], Rs[0], f["sample"], "image", scale=4,
              plot=f["plot"])
        cls, _ = _field_label(Rs[0], f)
        axes[r, 0].set_ylabel(f"field {int(f['plot'])}\n{cls}", color=FG,
                              fontsize=8)
        for c, R in enumerate(Rs, 1):
            _tile(axes[r, c], R, f["sample"], "pred", scale=4,
                  plot=f["plot"])
            _, iou = _field_label(R, f)
            axes[r, c].set_xlabel(f"{iou:.2f}" if iou is not None else "-",
                                  color=C_PADDY if (iou or 0) >= 0.5 else C_WRONG,
                                  fontsize=8, labelpad=2)
            if r == 0:
                axes[r, c].set_title(R["rung"], color=FG, fontsize=9,
                                     weight="bold")
    axes[0, 0].set_title("Sentinel-2", color=FG, fontsize=9)
    fig.suptitle("Four validation fields across the ablation ladder "
                 "(seed 0; number under each tile = IoU of that field's crop)",
                 color=FG, fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


def render_field_strip(Rs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    rng = np.random.default_rng(0)
    for i, R in enumerate(Rs):
        v = np.array(list(R["own_iou"].values()))
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=14,
                   color=np.where(v >= 0.5, C_PADDY, C_WRONG), alpha=0.85,
                   lw=0)
        ax.plot([i - 0.3, i + 0.3], [R["pooled_macro_iou"]] * 2,
                color=FG, lw=2.2)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=MUTED,
                lw=1.4, ls="--")
    ax.axhline(0.5, color="#555", lw=0.8)
    ax.set_xticks(range(len(Rs)))
    ax.set_xticklabels([R["rung"] for R in Rs], color=FG)
    ax.set_ylabel("IoU of each field's own crop", color=FG)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color("#333833")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Every field at every rung (dots), the pooled headline "
                 "(white bar) and the median field (dashed)  ·  seed 0",
                 color=FG, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=RUNGS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    Rs = []
    for rung in args.rungs:
        R = run_rung(rung, args.seed)
        Rs.append(R)
        d = R["max_abs_diff_vs_saved_per_plot"]
        print(f"{rung}: pooled IoU {R['pooled_macro_iou']:.4f} "
              f"(saved {R['pooled_macro_iou_saved']:.4f}); per-field IoU "
              f"matches training's record to within {d:.2e} over "
              f"{R['n_plots_compared']} fields", flush=True)
    # The left-hand image must be the same chip in every panel. v1 sees a
    # single acquisition (often a cloudy one), so the display image comes
    # from a rung that receives the full season; sample order is identical
    # across rungs because the evaluation loader never shuffles.
    ref = next((R for R in Rs if R["rung"] != "v1"), Rs[0])
    for R in Rs:
        if not np.array_equal(R["truth"], ref["truth"]):
            raise SystemExit(f"{R['rung']}: sample order differs from "
                             f"{ref['rung']}; cannot share display images")
        R["rgb_display"] = ref["rgb"]
        R["own_iou"] = own_class_iou(R)
    fields = pick_fields(Rs[0])
    for R in Rs:
        render_rung(R, fields, OUT / f"ladder_{R['rung']}.png")
    if len(Rs) > 1:
        render_montage(Rs, fields, OUT / "ladder_montage.png")
        render_field_strip(Rs, OUT / "ladder_field_iou.png")
    (OUT / "gallery_manifest.json").write_text(json.dumps({
        "fields": [{"plot": f["plot"], "sample_index": int(f["sample"]),
                    "share_of_labelled_pixels": f["px_share"]}
                   for f in fields],
        "selection_rule": "largest field, then 75th/50th/25th percentile "
                          "fields by labelled pixels; independent of results",
        "per_field_score_shown": "IoU of the field's own crop; macro IoU "
                                 "steps to <= 0.5 on any error in a "
                                 "single-crop field",
        "rungs": {R["rung"]: {**{k: R[k] for k in (
            "pooled_macro_iou", "pooled_macro_iou_saved", "plot_macro_iou",
            "max_abs_diff_vs_saved_per_plot", "n_plots_compared",
            "batch_size", "device")},
            "median_field_own_crop_iou": float(np.median(
                list(R["own_iou"].values()))),
            "median_field_macro_iou": float(np.median(
                list(R["per_plot_iou"].values())))} for R in Rs}}, indent=2),
        encoding="utf8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
