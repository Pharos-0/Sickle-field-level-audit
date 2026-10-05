"""PHASE 5 — THE REPRODUCTION GATE.

Load a released SICKLE checkpoint, evaluate it on the validation split with the
authors' own dataset, model and metric code, and check that the number comes
back.

Nothing in this study proceeds until this passes. A ladder built on a harness
that cannot recover a published number measures nothing.

TOLERANCE IS DECLARED BEFORE THE RUN AND IS NOT WIDENED AFTERWARDS:

    PASS if |reproduced val IoU - released best val IoU| <= 0.01 absolute

The released number comes from the run's own `trainlog.json` (the best epoch by
val_iou), which is what `checkpoint_best.pth.tar` corresponds to. The paper's
table reports TEST numbers, which cannot be reproduced because the test split
is unlabelled.

Device note: the discrete GPU on this machine intermittently powers down, and
the released checkpoints were saved on CUDA, so tensors are mapped explicitly.

    python scripts/phase5_gate.py
    python scripts/phase5_gate.py --run "[S1]_unet3d"
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
REF = PROJ / "reference_impl"
sys.path.insert(0, str(PROJ / "shims"))      # tifffile-backed rasterio
sys.path.insert(0, str(REF))                 # the authors' package layout

RUNS = Path("D:/FPD/runs/wacv_2024")
# Resolved from configs/base.yaml so this script and the ladder can never
# read different datasets. Was hard-coded to the now-deleted D:/FPD/sickle_data.
sys.path.insert(0, str(PROJ))
from src.utils import data_root  # noqa: E402
DATA = data_root()
OUT = PROJ / "results" / "gate"

TOLERANCE = 0.01          # declared before the run; do not widen


def load_cfg(run_dir: Path, device: str) -> Namespace:
    """Use the authors' own released config, with only the device changed."""
    cfg = json.loads((run_dir / "conf.json").read_text(encoding="utf8"))
    cfg["device"] = device
    cfg["data_dir"] = str(DATA)
    cfg["num_workers"] = 0
    cfg["best_path"] = str(run_dir)
    cfg["run_path"] = str(OUT / run_dir.name)
    return Namespace(**cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="[S2]_unet3d")
    ap.add_argument("--task", default="crop_type")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.task / args.run
    if not (run_dir / "checkpoint_best.pth.tar").exists():
        print(f"no checkpoint at {run_dir}")
        return 1

    import pandas as pd
    from utils.dataset import SICKLE_Dataset
    from utils import utae_utils, model_utils
    from utils.metric import get_metrics

    CFG = load_cfg(run_dir, device)
    Path(CFG.run_path).mkdir(parents=True, exist_ok=True)
    print(f"run      : {args.task}/{args.run}")
    print(f"device   : {device}")
    print(f"satellites: {CFG.satellites}")
    print(f"model    : {CFG.model}   num_classes {CFG.num_classes}")

    # ---- the released number this must recover
    log = json.loads((run_dir / "trainlog.json").read_text(encoding="utf8"))
    best_ep = max(log, key=lambda k: log[k].get("val_iou", -1))
    released = float(log[best_ep]["val_iou"])
    print(f"\nreleased best val_iou  : {released:.4f}  (epoch {best_ep})")
    print(f"declared tolerance     : +/- {TOLERANCE}")

    # ---- data, exactly as the authors build it
    df = pd.read_csv(DATA / "sickle_dataset_tabular.csv")
    if args.task != "crop_type":
        df = df[df.YIELD > 0].reset_index(drop=True)
    val_df = df[df.SPLIT == "val"].reset_index(drop=True)
    print(f"val samples            : {len(val_df)}")
    print(f"val plots              : {val_df.PLOT_ID.nunique()}")

    dt_val = SICKLE_Dataset(df=val_df, data_dir=str(DATA),
                            satellites=CFG.satellites,
                            ignore_index=CFG.ignore_index,
                            actual_season=getattr(CFG, "actual_season", False))
    loader = data.DataLoader(
        dt_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value),
        num_workers=0)

    # ---- model + released weights
    model = model_utils.Fusion_model(CFG).to(device)
    ckpt = torch.load(run_dir / "checkpoint_best.pth.tar",
                      map_location=torch.device(device), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params       : {nparams:,} "
          f"(released conf says {getattr(CFG, 'N_params', '?')})")

    # ---- evaluate
    # Loop structure copied from reference_impl/evaluate.py::iterate so the
    # reproduction uses the authors' own conventions: masks is a dict keyed by
    # task, plot_mask is separate, and softmax is applied before metrics.
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return [to_dev(c) for c in x]

    preds, trues, pids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_dev(batch)
            inputs, masks = batch
            plot_mask = masks["plot_mask"]
            y_true = masks[args.task].long()
            out = model(inputs)
            out = nn.Softmax(dim=1)(out)
            preds.append(out.cpu())
            trues.append(y_true.cpu())
            pids.append(plot_mask.cpu())
            if (i + 1) % 5 == 0:
                print(f"   batch {i + 1}/{len(loader)}", flush=True)

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)
    pid_masks = torch.cat(pids)
    print(f"\npredictions            : {tuple(y_pred.shape)}")

    # reference_impl/utils/metric.py::get_metrics returns, for crop_type:
    #   (f1_macro, acc, iou, paddy_f1, non_paddy_f1,
    #    paddy_acc, non_paddy_acc, paddy_iou, non_paddy_iou, (y_pred, y_true))
    m = get_metrics(y_pred, y_true, pid_masks,
                    ignore_index=CFG.ignore_index, task=args.task)
    names = ["f1_macro", "acc", "iou", "paddy_f1", "non_paddy_f1",
             "paddy_acc", "non_paddy_acc", "paddy_iou", "non_paddy_iou"]
    metrics = {}
    for name, val in zip(names, m):
        metrics[name] = float(val.item() if torch.is_tensor(val) else val)

    print("\nreproduced metrics:")
    for k, v in metrics.items():
        try:
            print(f"   {k:28s} {float(v):.4f}")
        except (TypeError, ValueError):
            print(f"   {k:28s} {v}")

    repro = metrics.get("val_iou", metrics.get("iou"))
    verdict = "UNKNOWN"
    delta = None
    if repro is not None:
        delta = abs(float(repro) - released)
        verdict = "PASS" if delta <= TOLERANCE else "FAIL"

    print("\n" + "=" * 68)
    print(f"released  {released:.4f}")
    if repro is not None:
        print(f"reproduced {float(repro):.4f}      |delta| = {delta:.4f}")
    print(f"tolerance  {TOLERANCE}")
    print(f"GATE: {verdict}")
    if verdict == "FAIL":
        print("\nDo NOT proceed to the ladder. Investigate the discrepancy "
              "first; a harness that cannot recover a released number cannot "
              "be trusted to measure a new one.")

    (OUT / f"gate_{args.task}_{args.run}_bs{args.batch_size}.json").write_text(json.dumps({
        "task": args.task, "run": args.run, "device": device,
        "batch_size": args.batch_size,
        "batch_size_note": ("Evaluation on this benchmark is batch-size "
                            "dependent: pad_collate pads the temporal axis to "
                            "the per-batch maximum. A result without its batch "
                            "size is not reproducible. See "
                            "docs/REPRODUCTION_GATE.md"),
        "released_best_val_iou": released, "released_best_epoch": int(best_ep),
        "reproduced": (float(repro) if repro is not None else None),
        "abs_delta": delta, "tolerance": TOLERANCE, "verdict": verdict,
        "val_samples": int(len(val_df)),
        "val_plots": int(val_df.PLOT_ID.nunique()),
        "trainable_params": int(nparams),
        "all_metrics": {k: (float(v) if not isinstance(v, str) else v)
                        for k, v in metrics.items()},
    }, indent=2), encoding="utf8")
    print(f"\nwrote {OUT / f'gate_{args.task}_{args.run}.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
