"""Evaluate a released checkpoint with PLOT-LEVEL metrics.

`scripts/phase5_gate.py` reproduces the authors' pixel-pooled benchmark number.
This adds the metric the study specification asks for in Section 10.2:

    Plot-Macro IoU -- per-plot IoU, every plot weighted equally,
                      with a 95% CI from resampling PLOTS (not pixels)

Why it matters here: the evaluation set is 36 plots whose pixel counts differ
by a large factor -- one field holds 47% of all scored pixels -- so the
pixel-pooled figure is dominated by a handful of large plots. That is a
statement about WEIGHT, not about difficulty: per-field IoU does not
correlate with field size here. Plot-Macro IoU is the metric that matches
this study's statistical claim, since the plot is the independent spatial
unit.

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
# Path order is load-bearing. `reference_impl/utils/` (the authors' package)
# and `src/utils.py` (this study's helpers) both answer to the name `utils`,
# and a module beats a namespace package, so putting `src` on sys.path makes
# `from utils.dataset import SICKLE_Dataset` resolve to src/utils.py and fail.
# `src` is therefore NOT added; this study's own modules are imported through
# the `src.` package instead. scripts/phase5_gate.py and
# scripts/regression_gate.py already follow this rule.
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))

# released checkpoints; SICKLE_RUNS as in .env.example
RUNS = Path(__import__("os").environ.get(
    "SICKLE_RUNS", "D:/FPD/runs/wacv_2024"))
# Resolved from configs/base.yaml so this script and the ladder can never
# read different datasets. Was hard-coded to the now-deleted D:/FPD/sickle_data.
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
    from src.evaluation.metrics import (bootstrap_plot_ci, crop_type_metrics,
                                        iou_from_confusion,
                                        per_plot_confusions, plot_macro_iou)

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
    cms = per_plot_confusions(y_true, y_pred, plot_ids,
                              ignore_index=CFG.ignore_index)

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
        print("   -> the two disagree materially. The pooled figure is "
              "dominated BY WEIGHT by a few large fields (see the "
              "leave-fields-out lines below); whether that raises or lowers "
              "it depends on how those particular fields scored, not on "
              "large fields being easier -- per-field IoU and field size "
              "are uncorrelated here (Spearman -0.04, see "
              "scripts/field_distributions.py). The plot-weighted value is "
              "the one that matches this study's statistical unit.")

    # ------------------------------------------------------------------
    # How much of the pooled figure is one field?
    #
    # The pooled confusion matrix is the plain sum of the per-plot matrices,
    # so dropping a plot and recomputing is exact. This is not a reweighting
    # argument about what the metric "effectively" measures: it is the metric
    # itself, recomputed on 35 fields instead of 36.
    # ------------------------------------------------------------------
    weights = {k: v["n_pixels"] for k, v in cms.items()}
    total_px = sum(weights.values())
    biggest = max(weights, key=weights.get) if weights else None
    loo = {}
    if biggest is not None:
        full = sum(v["cm"] for v in cms.values())
        base = float(np.nanmean(iou_from_confusion(full)))
        order = sorted(weights, key=weights.get, reverse=True)
        for label, drop in (("largest_field", order[:1]),
                            ("top_decile", order[:max(1, len(order) // 10)]),
                            ("top_quartile", order[:max(1, len(order) // 4)])):
            rest = sum(v["cm"] for k, v in cms.items() if k not in set(drop))
            loo[label] = {
                "n_fields_dropped": len(drop),
                "px_share_dropped": sum(weights[k] for k in drop) / total_px,
                "pooled_macro_iou_without_them":
                    float(np.nanmean(iou_from_confusion(rest))),
            }
        print()
        print(f"   pooled macro IoU recomputed from per-field "
              f"confusions: "
              f"{base:.4f}  (reference {pooled.macro_iou:.4f}, "
              f"{'consistent' if abs(base - pooled.macro_iou) < 1e-9 else 'INCONSISTENT'})")
        print(f"   largest field holds {weights[biggest] / total_px:.1%} of "
              f"the {total_px} scored pixels")
        for label, d in loo.items():
            print(f"   drop {label:<14s} ({d['n_fields_dropped']:2d} fields, "
                  f"{d['px_share_dropped']:5.1%} of pixels) -> pooled "
                  f"{d['pooled_macro_iou_without_them']:.4f} "
                  f"({d['pooled_macro_iou_without_them'] - base:+.4f})")
        loo["_pooled_recomputed"] = base

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
        "per_plot_n_pixels": {str(k): int(v["n_pixels"])
                              for k, v in sorted(cms.items())},
        "per_plot_confusion": {str(k): v["cm"].tolist()
                               for k, v in sorted(cms.items())},
        "leave_fields_out": loo,
    }, indent=2), encoding="utf8")
    print(f"\nwrote {OUT / f'plotlevel_{args.task}_{args.run}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
