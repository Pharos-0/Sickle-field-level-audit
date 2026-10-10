"""Reproduction gate + per-field audit for the REGRESSION tasks.

    python scripts/regression_gate.py --task crop_yield --run "[S2]_unet3d"
    python scripts/regression_gate.py --task harvesting_date --run "[S2]_unet3d"

Two jobs in one pass over the released checkpoint:

  1. GATE. Recover the number in the authors' own trainlog. Nothing built on a
     harness that cannot do that is trustworthy. Tolerance is declared here,
     before any run, and is not widened afterwards.

  2. AUDIT. Report the error per FIELD, not pooled -- the distribution, the
     quantiles, and the share of fields beyond a threshold. This is the part
     the aggregate number hides.

ON THE UNITS, because this is the whole question for yield
----------------------------------------------------------
`val_mape` in the released logs comes from `torchmetrics.MeanAbsolutePercentage
Error`, which returns a FRACTION, not a percent. Verified numerically: truth
100 with prediction 50 returns 0.5000. So a released value of 0.3274 means
**32.74%**, not 0.33%. This script prints both the raw fraction and the
percent so the two can never be confused again.

Mask band 5 is yield DIVIDED BY the field's pixel count, so a field's yield is
the SUM over its pixels. The reference `get_combined_yield` does this, and so
does the audit here. A per-pixel yield error would measure a quantity that
scales inversely with field area and would mean nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tdata

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))

from src import patches                                   # noqa: E402
from src.utils import data_root, write_json               # noqa: E402
from src.evaluation.regression_metrics import (           # noqa: E402
    aggregate_to_field_seasons, bootstrap_field_ci, error_distribution,
    field_errors, regression_metrics)

RUNS = Path("D:/FPD/runs/wacv_2024")
OUT = PROJ / "results" / "regression_gate"

# Declared BEFORE any run. Relative, because the tasks differ in scale:
# yield is kg, dates are days.
TOLERANCE_REL = 0.10          # reproduced within 10% of the released value

# yield sums over a field's pixels; a date is one value repeated across them
HOW = {"crop_yield": "sum", "sowing_date": "mean",
       "transplanting_date": "mean", "harvesting_date": "mean"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="crop_yield", choices=sorted(HOW))
    ap.add_argument("--run", default="[S2]_unet3d")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    patches.install()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.task / args.run
    ckpt = run_dir / "checkpoint_best.pth.tar"
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} -- NOT RUN")
        return 1

    import pandas as pd
    from utils.dataset import SICKLE_Dataset
    from utils import utae_utils, model_utils

    cfg = json.loads((run_dir / "conf.json").read_text(encoding="utf8"))
    cfg.update(device=device, data_dir=str(data_root()), num_workers=0)
    CFG = Namespace(**cfg)

    # the number this must recover
    log = json.loads((run_dir / "trainlog.json").read_text(encoding="utf8"))
    best_ep = min(log, key=lambda k: log[k].get("val_mape", 9e9))
    rel = {k: log[best_ep][k] for k in ("val_mape", "val_rmse", "val_mae")
           if k in log[best_ep]}

    print(f"task {args.task}   run {args.run}   device {device}   "
          f"batch_size {args.batch_size}")
    print(f"released best epoch {best_ep}: "
          + "  ".join(f"{k} {v:.4f}" for k, v in rel.items()))
    print(f"  released val_mape as a PERCENT: {rel['val_mape']*100:.2f}%  "
          f"(torchmetrics returns a fraction)")
    print(f"declared tolerance: +/-{TOLERANCE_REL*100:.0f}% relative\n")

    df = pd.read_csv(data_root() / "sickle_dataset_tabular.csv")
    df = df[df.YIELD > 0].reset_index(drop=True)
    val_df = df[df.SPLIT == "val"].reset_index(drop=True)

    sats = {k: dict(v) for k, v in CFG.satellites.items()}
    for v in sats.values():
        v["img_size"] = tuple(v["img_size"])
    ds = SICKLE_Dataset(df=val_df, data_dir=str(data_root()), satellites=sats,
                        ignore_index=CFG.ignore_index,
                        actual_season=getattr(CFG, "actual_season", False))
    loader = tdata.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value))

    model = model_utils.Fusion_model(CFG).to(device)
    state = torch.load(ckpt, map_location=torch.device(device),
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return [to_dev(c) for c in x]

    # Keep the SAMPLE axis. A plot appears in several season-samples and each
    # is a separate observation; flattening first destroys that boundary and
    # changes the number (0.2874 vs 0.3269 against a released 0.3274).
    preds, trues, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, masks = to_dev(batch)
            out = model(inputs)
            preds.append(out.squeeze().float().cpu().numpy())
            trues.append(masks[args.task].float().cpu().numpy())
            pids.append(masks["plot_mask"].cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    pid = np.concatenate(pids, axis=0)

    t_f, p_f = aggregate_to_field_seasons(
        y_true, y_pred, pid, ignore_index=CFG.ignore_index,
        how=HOW[args.task], require_positive=(args.task == "crop_yield"))

    # PIXEL-LEVEL, the unit the reference uses for the DATE tasks.
    #
    # get_regression_metrics only calls get_combined_yield for crop_yield. For
    # the date tasks it falls through to MAPE(), which is
    #     mean(|pred - true|) / 183.
    # -- per pixel, with a CONSTANT denominator of 183 days. That is not a
    # percentage error despite the column name, and it is verified: across 9
    # released date runs the reported val_mape equals val_mae/183 exactly.
    # So the date gate must reproduce val_mae at pixel level; the field-level
    # numbers below are this study's contribution, not a reproduction target.
    flat_t, flat_p = y_true.ravel(), y_pred.ravel()
    keep = (flat_t != CFG.ignore_index) & np.isfinite(flat_t) & np.isfinite(flat_p)
    px_mae = float(np.mean(np.abs(flat_p[keep] - flat_t[keep])))
    px_rmse = float(np.sqrt(np.mean((flat_p[keep] - flat_t[keep]) ** 2)))
    m = regression_metrics(t_f, p_f)
    errs = field_errors(t_f, p_f)
    dist = error_distribution(errs)
    ci = bootstrap_field_ci(t_f, p_f, metric="mape", n_boot=args.n_boot)

    if args.task == "crop_yield":
        gate_on = "MAPE over field-seasons"
        released_val, repro_val = rel["val_mape"], (m.mape or 0) / 100.0
    else:
        gate_on = "MAE over PIXELS (the reference's unit for date tasks)"
        released_val, repro_val = rel["val_mae"], px_mae
    delta_rel = abs(repro_val - released_val) / max(abs(released_val), 1e-9)
    verdict = "PASS" if delta_rel <= TOLERANCE_REL else "FAIL"

    print(f"REPRODUCTION  -- gated on {gate_on}")
    print(f"   released    {released_val:.4f}")
    print(f"   reproduced  {repro_val:.4f}")
    print(f"   relative delta  {delta_rel*100:.1f}%     GATE: {verdict}")
    if args.task != "crop_yield":
        print(f"   NOTE: the released val_mape for date tasks is MAE/183, a "
              f"CONSTANT denominator, not a percentage error "
              f"({rel['val_mae']:.3f}/183 = {rel['val_mae']/183:.4f} = "
              f"{rel['val_mape']:.4f}). Verified on 9 released runs.")
        print(f"   pixel-level: MAE {px_mae:.3f}  RMSE {px_rmse:.3f}")
    print(f"   field-seasons   {m.n_fields}   excluded from MAPE: "
          f"{m.n_excluded_from_mape}")

    print("\nPER-FIELD AUDIT (the number an aggregate hides)")
    print(f"   MAE  {m.mae:.2f}      RMSE {m.rmse:.2f}")
    print(f"   absolute error per field-season: median {dist['median']:.2f}   "
          f"p75 {dist['p75']:.2f}   p90 {dist['p90']:.2f}   max {dist['max']:.2f}")
    print(f"   95% CI on MAPE (resampling fields): "
          f"[{ci['lo']:.2f}%, {ci['hi']:.2f}%]")
    if m.rmse > 1.5 * m.mae:
        print("   -> RMSE far exceeds MAE: the error is driven by a few badly "
              "wrong fields, not spread evenly.")

    write_json(OUT / f"{args.task}_{args.run}_bs{args.batch_size}.json", {
        "task": args.task, "run": args.run, "device": device,
        "batch_size": args.batch_size,
        "released": rel, "released_best_epoch": int(best_ep),
        "mape_note": ("torchmetrics MeanAbsolutePercentageError returns a "
                      "FRACTION; 0.3274 means 32.74%. Verified numerically."),
        "gated_on": gate_on,
        "pixel_level": {"mae": px_mae, "rmse": px_rmse},
        "date_mape_note": ("For date tasks the released val_mape is MAE/183 "
                           "(constant denominator), not a percentage error. "
                           "Verified: val_mape == val_mae/183 on 9/9 runs."),
        "reproduced": m.as_dict(), "relative_delta": delta_rel,
        "tolerance_rel": TOLERANCE_REL, "verdict": verdict,
        "per_field_error": dist, "mape_ci_fields": ci,
        "per_field_abs_error": {str(k): v for k, v in sorted(errs.items())},
    })
    print(f"\nwrote {OUT / f'{args.task}_{args.run}_bs{args.batch_size}.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
