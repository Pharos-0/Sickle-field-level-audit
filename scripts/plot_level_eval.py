"""Evaluate a released checkpoint with PLOT-LEVEL metrics.

`scripts/phase5_gate.py` reproduces the authors' pixel-pooled benchmark number.
This adds the metric the study specification asks for in Section 10.2:

    Plot-Macro IoU -- per-plot IoU, every plot weighted equally,
                      with a 95% CI from resampling PLOTS (not pixels)

Why it matters here: the evaluation set is 36 plots whose pixel counts differ
by a large factor, so pixel-pooled IoU is dominated by a handful of large
plots. Plot-Macro IoU is the metric that matches this study's statistical
claim, since the plot is the independent spatial unit.

Band 0 of the SICKLE mask carries PLOT_ID, which is what makes this
computable at all (see docs/IMPLEMENTATION_AUDIT.md section 3).

Batch size is fixed at 32 by default: evaluation on this benchmark is
batch-size dependent (docs/REPRODUCTION_GATE.md section 2).

    python scripts/plot_level_eval.py --run "[S2]_unet3d"
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))
sys.path.insert(0, str(PROJ / "src"))

RUNS = Path("D:/FPD/runs/wacv_2024")
# Resolved from configs/base.yaml so this script and the ladder can never
# read different datasets. Was hard-coded to the now-deleted D:/FPD/sickle_data.
sys.path.insert(0, str(PROJ))
from src.utils import data_root  # noqa: E402
DATA = data_root()
OUT = PROJ / "results" / "plot_level"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="[S2]_unet3d")
    ap.add_argument("--task", default="crop_type")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.task / args.run

    import pandas as pd
    from utils.dataset import SICKLE_Dataset
    from utils import utae_utils, model_utils
    from evaluation.metrics import (bootstrap_plot_ci, crop_type_metrics,
                                    plot_macro_iou)

    cfg = json.loads((run_dir / "conf.json").read_text(encoding="utf8"))
    cfg.update(device=device, data_dir=str(DATA), num_workers=0)
    CFG = Namespace(**cfg)

    df = pd.read_csv(DATA / "sickle_dataset_tabular.csv")
    val_df = df[df.SPLIT == "val"].reset_index(drop=True)
    dt = SICKLE_Dataset(df=val_df, data_dir=str(DATA),
                        satellites=CFG.satellites,
                        ignore_index=CFG.ignore_index,
                        actual_season=getattr(CFG, "actual_season", False))
    loader = data.DataLoader(
        dt, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value),
        num_workers=0)

    model = model_utils.Fusion_model(CFG).to(device)
    ck = torch.load(run_dir / "checkpoint_best.pth.tar",
                    map_location=torch.device(device), weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return [to_dev(c) for c in x]

    preds, trues, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, masks = to_dev(batch)
            out = nn.Softmax(dim=1)(model(inputs))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(masks[args.task].long().cpu().numpy())
            pids.append(masks["plot_mask"].cpu().numpy())

    y_pred = np.concatenate([p.ravel() for p in preds])
    y_true = np.concatenate([t.ravel() for t in trues])
    plot_ids = np.concatenate([g.ravel() for g in pids])

    pooled = crop_type_metrics(y_true, y_pred, ignore_index=CFG.ignore_index)
    summary, per_plot = plot_macro_iou(y_true, y_pred, plot_ids,
                                       ignore_index=CFG.ignore_index,
                                       return_per_plot=True)
    ci = bootstrap_plot_ci(per_plot, n_boot=args.n_boot)

    print(f"run {args.task}/{args.run}   device {device}   "
          f"batch_size {args.batch_size}\n")
    print("PIXEL-POOLED (the SICKLE-comparable benchmark metric)")
    for k, v in pooled.as_dict().items():
        print(f"   {k:24s} {v:.4f}" if isinstance(v, float)
              else f"   {k:24s} {v}")

    print("\nPLOT-LEVEL (this study's spatial-generalisation metric)")
    print(f"   plot_macro_iou           {summary['plot_macro_iou']:.4f}")
    print(f"   plot_iou_sd              {summary['plot_iou_sd']:.4f}")
    print(f"   95% CI (resampling plots) [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"   n_plots                  {summary['n_plots']}")
    print(f"   n_plots_skipped          {summary['n_plots_skipped']}")
    print(f"   undefined class instances {summary['n_undefined_class_instances']}")

    gap = pooled.macro_iou - summary["plot_macro_iou"]
    print(f"\n   pixel-pooled minus plot-macro: {gap:+.4f}")
    if abs(gap) > 0.02:
        print("   -> the two disagree materially. Pixel-pooled IoU is being "
              "carried by the larger plots; the plot-weighted value is the one "
              "that matches this study's statistical unit.")

    vals = np.array(sorted(per_plot.values()))
    if vals.size:
        print(f"\n   per-plot IoU: min {vals.min():.3f}  "
              f"p25 {np.percentile(vals, 25):.3f}  "
              f"median {np.median(vals):.3f}  "
              f"p75 {np.percentile(vals, 75):.3f}  max {vals.max():.3f}")

    (OUT / f"plotlevel_{args.task}_{args.run}.json").write_text(json.dumps({
        "run": args.run, "task": args.task, "batch_size": args.batch_size,
        "device": device,
        "pixel_pooled": pooled.as_dict(),
        "plot_level": summary,
        "plot_bootstrap_ci": ci,
        "pixel_minus_plot": float(gap),
        "per_plot_iou": {str(k): v for k, v in sorted(per_plot.items())},
    }, indent=2), encoding="utf8")
    print(f"\nwrote {OUT / f'plotlevel_{args.task}_{args.run}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
