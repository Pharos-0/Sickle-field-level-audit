"""Qualitative prediction maps for a trained checkpoint.

Numbers decide the study; pictures are how a reader checks the numbers are
about what they think they are. This renders, per plot: the Sentinel-2 RGB
context, the ground-truth paddy mask, the prediction, and the disagreement.

    python -m src.predict --config v1 --seed 0 --n 6

Everything outside the plot is `ignore_index` and is drawn grey, not black, so
that "not evaluated" is visually distinct from "predicted non-paddy". Conflating
those two is the easiest way to make a model look better than it is: only ~3% of
each 32x32 chip carries a label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import patches, utils                 # noqa: E402
from src.dataset import build_loader           # noqa: E402
from src.model import build_model              # noqa: E402

PADDY, NON_PADDY = 0, 1


def _rgb(img: np.ndarray, rgb_bands) -> np.ndarray:
    """A displayable RGB composite from one acquisition.

    Percentile stretch rather than a fixed divisor: reflectance scaling varies
    between scenes and a fixed /10000 renders most chips nearly black.
    """
    chans = [img[b] for b in rgb_bands]
    out = np.stack(chans, axis=-1).astype(np.float32)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    if hi <= lo:
        return np.zeros_like(out)
    return np.clip((out - lo) / (hi - lo), 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    patches.install()
    cfg = utils.load_config(args.config)
    utils.set_seed(args.seed)
    device = utils.pick_device(args.device or cfg["runtime"]["device"])
    task = cfg["data"]["task"]
    ig = cfg["data"]["ignore_index"]

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        utils.run_dir(cfg, args.seed) / "checkpoint_best.pth.tar")
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} -- NOT RUN")
        return 1

    loader, df = build_loader(cfg, cfg["data"]["eval_split"], "eval")
    model, _ = build_model(cfg, device)
    state = torch.load(ckpt, map_location=torch.device(device),
                       weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    sat = list(cfg["satellites"])[0]
    rgb_bands = cfg["satellites"][sat]["rgb_bands"]

    out_dir = PROJ / "results" / "predictions" / f"{cfg['project']['version']}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(["#2e7d32", "#c62828"])     # paddy green, other red
    done = 0
    with torch.no_grad():
        for batch in loader:
            inp, mask = batch
            dev_in = {k: (v[0].to(device), v[1].to(device))
                      for k, v in inp.items()}
            pred = nn.Softmax(dim=1)(model(dev_in)).argmax(1).cpu().numpy()
            y = mask[task].long().numpy()
            imgs = inp[sat][0].numpy()

            for i in range(pred.shape[0]):
                if done >= args.n:
                    break
                valid = y[i] != ig
                if valid.sum() == 0:
                    continue

                p = np.ma.masked_where(~valid, pred[i])
                t = np.ma.masked_where(~valid, y[i])
                disagree = np.ma.masked_where(~valid, (pred[i] != y[i]).astype(int))

                mid = imgs[i].shape[0] // 2
                fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
                ax[0].imshow(_rgb(imgs[i][mid], rgb_bands))
                ax[0].set_title(f"S2 RGB (t={mid})", fontsize=9)
                ax[1].imshow(t, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax[1].set_title("truth: paddy / non-paddy", fontsize=9)
                ax[2].imshow(p, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax[2].set_title("prediction", fontsize=9)
                ax[3].imshow(disagree, cmap=ListedColormap(["#eeeeee", "#000000"]),
                             vmin=0, vmax=1, interpolation="nearest")
                iou_hint = int((pred[i][valid] == y[i][valid]).sum())
                ax[3].set_title(f"disagreement ({valid.sum() - iou_hint} px)",
                                fontsize=9)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                    a.set_facecolor("#9e9e9e")          # grey = not evaluated
                fig.suptitle(f"{cfg['project']['version']}  "
                             f"{valid.sum()} labelled px of {y[i].size} "
                             f"({100*valid.sum()/y[i].size:.1f}%)", fontsize=9)
                fig.tight_layout()
                fp = out_dir / f"pred_{done:02d}.png"
                fig.savefig(fp, dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {fp}")
                done += 1
            if done >= args.n:
                break

    print(f"\n{done} prediction maps in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
