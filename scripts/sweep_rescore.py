"""Re-score every data-regime sweep run with field IoU.

    python scripts/sweep_rescore.py              # GPU if free, else CPU
    python scripts/sweep_rescore.py --device cpu
    python scripts/sweep_rescore.py --redo       # ignore existing outputs

Protocol: configs/sweep_analysis_protocol.yaml (frozen before this ran).

The trainer recorded each run's pixel-pooled macro IoU and a per-plot macro
IoU that scores a class absent from a plot's truth as 0. The protocol's
secondary metric is field IoU, which leaves such a class out (the master
prompt's section 10.2 rule). A ratio cannot be re-scored, so each run's best
checkpoint is re-run on the validation split in float32 at the pinned batch
size, and per-field confusion matrices are kept.

Validity check, per run: the recomputed pooled macro IoU must lie within
2e-3 of best.json (training evaluated under reduced precision). A run that
fails is recorded as failing, not dropped, and the analysis refuses to
report while any run fails.

Writes
  results/sweep/field_confusions/<subset>__<run>.json   per-field matrices
      (label-derived, so git-ignored)
  results/sweep/rescore.csv                             one row per run,
      aggregates only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

SWEEP = PROJ / "outputs" / "crop_type" / "sweep"
OUT = PROJ / "results" / "sweep"
TOL_POOLED = 2e-3
FIELDS = ["subset", "plot_count", "seed", "config", "run", "best_epoch",
          "pooled_macro_iou_training", "pooled_macro_iou_recomputed",
          "abs_diff", "pooled_check", "n_fields", "n_single_crop",
          "field_iou_mean", "field_iou_median", "n_fields_below_0_5",
          "trainer_plot_macro_iou", "device"]


def protocol_configs():
    import yaml
    return list(yaml.safe_load((PROJ / "configs" / "sweep_analysis_protocol.yaml")
                               .read_text(encoding="utf8"))["design"]["configs"])


def runs():
    """The protocol's runs only. Five v5 runs (seed 1, 50-250 plots) predate
    commit a8aa0dc (64f48cb before the 2026-10-23 history rewrite), which replaced v5 with v2_radar on 2026-10-10 because v5
    also carries augmentation and class weighting; they are not re-scored."""
    keep = set(protocol_configs())
    out = []
    for d in sorted(SWEEP.glob("plots_*_seed*/*_seed*")):
        m = re.match(r"plots_(\d+)_seed0*(\d+)$", d.parent.name)
        c = re.match(r"(.+)_seed(\d+)$", d.name)
        if m and c and c.group(1) in keep:
            out.append({"dir": d, "subset": d.parent.name,
                        "plot_count": int(m.group(1)), "seed": int(m.group(2)),
                        "config": c.group(1), "run": d.name})
    return out


def infer(run_dir: Path, device: str):
    """Best checkpoint on the validation split, float32, pinned batch size."""
    import torch
    import torch.nn as nn
    from src import patches
    from src.dataset import build_loader
    from src.model import build_model
    from src.train import to_device

    patches.install()
    cfg = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf8"))
    cfg["training"]["num_workers"] = 0
    task, ig = cfg["data"]["task"], cfg["data"]["ignore_index"]
    loader, _ = build_loader(cfg, cfg["data"]["eval_split"], "eval")
    model, _ = build_model(cfg, device)
    state = torch.load(run_dir / "checkpoint_best.pth.tar",
                       map_location=torch.device(device), weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    preds, truths, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, masks = to_device(batch, device)
            out = nn.Softmax(dim=1)(model(inputs).float())
            preds.append(out.argmax(1).cpu().numpy().astype(np.int8))
            truths.append(masks[task].long().cpu().numpy().astype(np.int16))
            pids.append(masks["plot_mask"].cpu().numpy().astype(np.float32))
    return (np.concatenate(truths).ravel(), np.concatenate(preds).ravel(),
            np.concatenate(pids).ravel(), ig, int(cfg["training"]["batch_size"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    import torch
    from src.evaluation.metrics import (crop_type_metrics,
                                        own_class_iou_from_confusion,
                                        per_plot_confusions)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rs = runs()
    print(f"{len(rs)} sweep runs found; device {device}", flush=True)
    (OUT / "field_confusions").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(rs, 1):
        best = json.loads((r["dir"] / "best.json").read_text(encoding="utf8"))
        if not best.get("complete"):
            print(f"[{i}/{len(rs)}] {r['subset']}/{r['run']}: NOT COMPLETE, "
                  f"skipped", flush=True)
            continue
        fc = OUT / "field_confusions" / f"{r['subset']}__{r['run']}.json"
        if fc.exists() and not args.redo:
            rec = json.loads(fc.read_text(encoding="utf8"))
        else:
            t, p, g, ig, bs = infer(r["dir"], device)
            pooled = float(crop_type_metrics(t, p, ignore_index=ig).macro_iou)
            cms = per_plot_confusions(t, p, g, ignore_index=ig)
            rec = {"subset": r["subset"], "run": r["run"], "device": device,
                   "batch_size": bs, "pooled_macro_iou": pooled,
                   "fields": {str(k): {"cm": v["cm"].tolist(),
                                       "n_pixels": v["n_pixels"]}
                              for k, v in cms.items()}}
            fc.write_text(json.dumps(rec), encoding="utf8")
        fi = {k: own_class_iou_from_confusion(np.asarray(v["cm"]))
              for k, v in rec["fields"].items()}
        vals = np.array(list(fi.values()))
        single = sum(1 for v in rec["fields"].values()
                     if np.count_nonzero(np.asarray(v["cm"]).sum(1)) == 1)
        diff = abs(rec["pooled_macro_iou"] - float(best["macro_iou"]))
        rows.append({
            "subset": r["subset"], "plot_count": r["plot_count"],
            "seed": r["seed"], "config": r["config"], "run": r["run"],
            "best_epoch": best.get("best_epoch"),
            "pooled_macro_iou_training": float(best["macro_iou"]),
            "pooled_macro_iou_recomputed": rec["pooled_macro_iou"],
            "abs_diff": diff, "pooled_check": diff <= TOL_POOLED,
            "n_fields": len(vals), "n_single_crop": single,
            "field_iou_mean": float(vals.mean()),
            "field_iou_median": float(np.median(vals)),
            "n_fields_below_0_5": int((vals < 0.5).sum()),
            "trainer_plot_macro_iou": best.get("plot_macro_iou"),
            "device": rec["device"],
        })
        print(f"[{i}/{len(rs)}] {r['subset']}/{r['run']}: pooled "
              f"{rec['pooled_macro_iou']:.4f} (training "
              f"{float(best['macro_iou']):.4f}, diff {diff:.1e}) field IoU "
              f"mean {vals.mean():.3f}", flush=True)

    with open(OUT / "rescore.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    bad = [f"{x['subset']}/{x['run']}" for x in rows if not x["pooled_check"]]
    print(f"wrote {OUT / 'rescore.csv'}: {len(rows)} runs; pooled check "
          f"failed for {len(bad)}" + (f": {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
