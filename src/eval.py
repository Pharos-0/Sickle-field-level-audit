"""Evaluate a trained ladder checkpoint, or any released SICKLE run.

    python -m src.eval --config v1 --seed 0
    python -m src.eval --checkpoint outputs/crop_type/v1_seed0/checkpoint_best.pth.tar

Reports both metric families side by side and states the gap between them. On
this dataset the gap is large and not uniform: across the eleven released cells
measured so far its median is +0.18 and it reorders which model wins, so a
single number is not a summary of this benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import utils                      # noqa: E402
from src.dataset import build_loader       # noqa: E402
from src.model import build_model, count_params   # noqa: E402
from src.train import evaluate             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    utils.set_seed(args.seed)
    device = utils.pick_device(args.device or cfg["runtime"]["device"])
    task = cfg["data"]["task"]
    split = args.split or cfg["data"]["eval_split"]

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        utils.run_dir(cfg, args.seed) / "checkpoint_best.pth.tar")
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} -- NOT RUN")
        return 1

    loader, df = build_loader(cfg, split, "eval")
    model, _ = build_model(cfg, device)
    state = torch.load(ckpt, map_location=torch.device(device),
                       weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    m = evaluate(model, loader, cfg, device, task,
                 boot_n=cfg["evaluation"]["bootstrap_n"])
    per_plot = m.pop("_per_plot", None)

    print(f"{cfg['project']['version']}  seed {args.seed}  split {split}  "
          f"device {device}  batch_size {cfg['training']['batch_size']}")
    print(f"  checkpoint {ckpt}")
    print(f"  {len(df)} samples / {df.PLOT_ID.nunique()} plots   "
          f"params {count_params(model):,}\n")

    print("PIXEL-POOLED (benchmark-comparable)")
    for k in ("macro_iou", "paddy_iou", "non_paddy_iou", "macro_f1",
              "paddy_precision", "paddy_recall"):
        if k in m:
            print(f"   {k:24s} {m[k]:.4f}")

    if "plot_macro_iou" in m:
        print("\nPLOT-LEVEL (this study's spatial unit)")
        print(f"   {'plot_macro_iou':24s} {m['plot_macro_iou']:.4f}")
        if "plot_ci_lo" in m:
            print(f"   {'95% CI (plots resampled)':24s} "
                  f"[{m['plot_ci_lo']:.4f}, {m['plot_ci_hi']:.4f}]")
        if "plot_n_plots" in m:
            print(f"   {'n_plots':24s} {m['plot_n_plots']}")
        gap = m["macro_iou"] - m["plot_macro_iou"]
        print(f"\n   pixel-pooled minus plot-macro: {gap:+.4f}")
        if abs(gap) > 0.02:
            print("   -> they disagree materially. Pixel-pooled IoU is carried "
                  "by the larger plots.")

    out = utils.run_dir(cfg, args.seed) / f"eval_{split}.json"
    utils.write_json(out, {"config": args.config, "seed": args.seed,
                           "split": split, "device": device,
                           "batch_size": cfg["training"]["batch_size"],
                           "checkpoint": str(ckpt), "metrics": m,
                           "per_plot_iou": per_plot})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
