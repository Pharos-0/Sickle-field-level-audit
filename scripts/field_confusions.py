"""Per-field confusion matrices for a ladder rung, from CPU inference.

    python scripts/field_confusions.py              # rung v5, its completed seeds

The trainer records each field's macro IoU (best_per_plot_iou.json) but not
its confusion matrix, and a ratio cannot be re-scored. Every per-field
quantity the study reports is therefore derived from the matrices written
here, so the macro score and the field score (src/evaluation/metrics.py)
come from the same pixels.

Inference runs on the CPU at the pinned evaluation batch size, reusing the
prediction caches of scripts/render_sickle_gallery.py. Training evaluated on
the GPU under reduced-precision autocast; float32 on the CPU can flip a
pixel that sits on the decision boundary, and in a 20-pixel field one pixel
moves IoU by a few thousandths. So two checks, with different strictness:
each seed's POOLED macro IoU must match training's best.json within 2e-3
(that stops the script), and every field's macro IoU is compared with
training's record and the largest difference is written out.

Writes results/field_confusions/<rung>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.evaluation.metrics import (crop_type_metrics,  # noqa: E402
                                    iou_from_confusion,
                                    own_class_iou_from_confusion,
                                    per_plot_confusions)
from src.utils import completed_seed_runs               # noqa: E402

OUT = PROJ / "results" / "field_confusions"
TOL_POOLED = 2e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="v5")
    args = ap.parse_args()
    import render_sickle_gallery as g

    kept, dropped = completed_seed_runs("crop_type", args.rung, PROJ)
    seeds = sorted(int(d.name.rsplit("seed", 1)[1]) for d, _ in kept)
    if not seeds:
        raise SystemExit(f"no completed seeds for {args.rung}")
    out = {"rung": args.rung, "seeds": seeds, "device": "cpu",
           "excluded_incomplete": [d.name for d, _ in dropped],
           "per_seed": {}}
    worst = 0.0
    pooled_check = {}
    for s in seeds:
        R = g.run_rung(args.rung, s)
        ig = R["ignore_index"]
        pooled = float(crop_type_metrics(R["truth"].ravel(), R["pred"].ravel(),
                                         ignore_index=ig).macro_iou)
        rec = json.loads((PROJ / "outputs" / "crop_type" /
                          f"{args.rung}_seed{s}" / "best.json")
                         .read_text(encoding="utf8"))["macro_iou"]
        pooled_check[str(s)] = {"cpu": pooled, "training": float(rec)}
        if abs(pooled - float(rec)) > TOL_POOLED:
            raise SystemExit(f"{args.rung} seed {s}: pooled macro IoU {pooled:.4f} "
                             f"vs training's {float(rec):.4f}; refusing to write")
        cms = per_plot_confusions(R["truth"].ravel(), R["pred"].ravel(),
                                  R["pid"].ravel(), ignore_index=ig)
        saved = json.loads((PROJ / "outputs" / "crop_type" /
                            f"{args.rung}_seed{s}" /
                            "best_per_plot_iou.json").read_text(encoding="utf8"))
        rows = {}
        for k, v in cms.items():
            macro = float(np.nanmean(iou_from_confusion(v["cm"])))
            field = own_class_iou_from_confusion(v["cm"])
            ref = saved.get(str(k))
            if ref is not None:
                worst = max(worst, abs(macro - float(ref)))
            rows[str(k)] = {"cm": v["cm"].tolist(), "n_pixels": v["n_pixels"],
                            "macro_iou": macro, "field_iou": field}
        out["per_seed"][str(s)] = rows
        n1 = sum(1 for r in rows.values()
                 if np.count_nonzero(np.asarray(r["cm"]).sum(1)) == 1)
        print(f"{args.rung} seed {s}: {len(rows)} fields, {n1} hold one crop")
    out["max_abs_diff_macro_vs_training_record"] = worst
    out["pooled_macro_iou_check"] = pooled_check
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{args.rung}.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"pooled IoU matches training's record for every seed; largest "
          f"per-field macro difference {worst:.2e}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
