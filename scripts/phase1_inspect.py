"""PHASE 1 — Arm B viability decision, and the real on-disk data format.

Two jobs, both pre-requisites for writing any dataloader:

  A. Decide Arm B (reduced K-class) viability against the PRE-SPECIFIED
     criterion, and write artifacts/arm_b_viability.json. The criterion is
     fixed before the decision and is not lowered to manufacture a result.

  B. Extract ONE plot's files from the zip and report the actual array shapes,
     dtypes and value ranges. Nothing about the data format may be assumed.

    python scripts/phase1_inspect.py
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

FPD = Path("D:/FPD")
ZIP = FPD / "sickle_dataset-002.zip"
TAB = Path("C:/ds/sickle/sickle_dataset_tabular.csv")
ART = Path(__file__).resolve().parents[1] / "artifacts"

#: Pre-specified feasibility criterion for THIS study. Not a general
#: statistical sufficiency claim. Fixed before the decision is computed.
ARM_B_MIN_CLASSES = 3
ARM_B_MIN_VAL_PLOTS_PER_CLASS = 5
ARM_B_CANDIDATE_THRESHOLDS = [3, 5, 10, 15, 20]


def arm_b_viability() -> dict:
    rows = list(csv.DictReader(TAB.open(newline="", encoding="utf8")))
    split_plots = defaultdict(set)
    crop_plots = defaultdict(set)
    for r in rows:
        if r["PLOT_ID"]:
            split_plots[r["SPLIT"]].add(r["PLOT_ID"])
            crop_plots[r["CROP"]].add(r["PLOT_ID"])
    train, val = split_plots["train"], split_plots["val"]

    per_class = []
    for crop, ps in sorted(crop_plots.items(), key=lambda kv: -len(kv[1])):
        per_class.append({
            "crop": crop, "plots_total": len(ps),
            "plots_train": len(ps & train), "plots_val": len(ps & val),
        })

    evaluated = {}
    for m in ARM_B_CANDIDATE_THRESHOLDS:
        kept = [c for c in per_class if c["plots_train"] >= m]
        min_val = min((c["plots_val"] for c in kept), default=0)
        evaluated[str(m)] = {
            "K": len(kept),
            "classes": [c["crop"] for c in kept],
            "min_val_plots_in_a_retained_class": min_val,
            "meets_criterion": (len(kept) >= ARM_B_MIN_CLASSES
                                and min_val >= ARM_B_MIN_VAL_PLOTS_PER_CLASS),
        }

    viable = any(v["meets_criterion"] for v in evaluated.values())
    return {
        "criterion": {
            "note": ("Pre-specified feasibility criterion for this study; not a "
                     "general statistical sufficiency claim. Fixed before the "
                     "decision was computed and not lowered afterwards."),
            "min_classes": ARM_B_MIN_CLASSES,
            "min_val_plots_per_class": ARM_B_MIN_VAL_PLOTS_PER_CLASS,
        },
        "per_class_plot_counts": per_class,
        "thresholds_evaluated": evaluated,
        "arm_b_status": "VIABLE" if viable else "NOT_VIABLE",
        "reason": ("At least one threshold yields >=3 classes each with >=5 "
                   "validation plots." if viable else
                   "No threshold yields 3 classes with >=5 validation plots "
                   "each. Beyond Paddy, only Banana and Coconut have any "
                   "validation support (4 and 3 plots). Arm B is skipped; the "
                   "difficulty axis must be replaced, not dropped."),
    }


def inspect_format() -> dict:
    """Pull one plot's actual files out of the zip and report what they are."""
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    out: dict = {"samples": {}}

    for sat in ("S2", "S1", "L8"):
        cand = [n for n in names
                if n.startswith(f"images/{sat}/") and n.endswith(".npz")]
        if not cand:
            out["samples"][sat] = {"status": "NO FILES FOUND"}
            continue
        key = sorted(cand)[0]
        with z.open(key) as fh:
            data = np.load(io.BytesIO(fh.read()), allow_pickle=False)
            arrays = {}
            for k in data.files:
                a = data[k]
                arrays[k] = {
                    "shape": list(a.shape), "dtype": str(a.dtype),
                    "min": float(np.nanmin(a)) if a.size else None,
                    "max": float(np.nanmax(a)) if a.size else None,
                    "has_nan": bool(np.isnan(a).any()) if
                    np.issubdtype(a.dtype, np.floating) else False,
                }
        out["samples"][sat] = {"path": key, "keys": list(arrays),
                               "arrays": arrays}

    for res in ("10m", "30m", "3m"):
        cand = [n for n in names
                if n.startswith(f"masks/{res}/") and n.endswith(".tif")]
        if not cand:
            out[f"mask_{res}"] = {"status": "NO FILES FOUND"}
            continue
        key = sorted(cand)[0]
        with z.open(key) as fh:
            im = Image.open(io.BytesIO(fh.read()))
            a = np.array(im)
        vals, counts = np.unique(a, return_counts=True)
        out[f"mask_{res}"] = {
            "path": key, "shape": list(a.shape), "dtype": str(a.dtype),
            "unique_values": vals.tolist()[:20],
            "value_counts": {str(v): int(c) for v, c in
                             list(zip(vals, counts))[:20]},
        }

    # how are files named / keyed back to the tabular rows?
    s2 = [n for n in names if n.startswith("images/S2/") and n.endswith(".npz")]
    out["naming"] = {
        "example_image_paths": sorted(s2)[:5],
        "example_mask_paths": sorted(
            n for n in names if n.startswith("masks/10m/")
            and n.endswith(".tif"))[:5],
        "distinct_dirs_under_images_S2": len({n.split("/")[2] for n in s2
                                              if n.count("/") > 2}),
    }
    return out


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)

    print("A. ARM B VIABILITY\n")
    ab = arm_b_viability()
    print(f"{'crop':22s}{'total':>7s}{'train':>7s}{'val':>6s}")
    for c in ab["per_class_plot_counts"][:12]:
        print(f"{c['crop'][:22]:22s}{c['plots_total']:7d}"
              f"{c['plots_train']:7d}{c['plots_val']:6d}")
    print()
    for m, v in ab["thresholds_evaluated"].items():
        print(f"  MIN_PLOTS_PER_CLASS={m:>3s} -> K={v['K']:2d}  "
              f"min val plots in a retained class={v['min_val_plots_in_a_retained_class']:3d}"
              f"  {'MEETS' if v['meets_criterion'] else 'fails'}")
    print(f"\n  >>> ARM B STATUS: {ab['arm_b_status']}")
    print(f"      {ab['reason']}")
    (ART / "arm_b_viability.json").write_text(json.dumps(ab, indent=2),
                                              encoding="utf8")

    print("\n\nB. ON-DISK DATA FORMAT (one sample per modality)\n")
    fmt = inspect_format()
    for sat, d in fmt["samples"].items():
        if "arrays" not in d:
            print(f"  {sat}: {d.get('status')}")
            continue
        print(f"  images/{sat}  {d['path'].split('/')[-1]}")
        for k, a in d["arrays"].items():
            print(f"      key '{k}'  shape {a['shape']}  dtype {a['dtype']}  "
                  f"range [{a['min']}, {a['max']}]"
                  + ("  HAS NaN" if a["has_nan"] else ""))
    for res in ("3m", "10m", "30m"):
        d = fmt.get(f"mask_{res}", {})
        if "shape" in d:
            print(f"  masks/{res}   shape {d['shape']}  dtype {d['dtype']}  "
                  f"values {d['unique_values']}")
    print("\n  example image paths:")
    for p in fmt["naming"]["example_image_paths"]:
        print(f"      {p}")
    print("  example mask paths:")
    for p in fmt["naming"]["example_mask_paths"]:
        print(f"      {p}")

    (ART / "dataset_format.json").write_text(json.dumps(fmt, indent=2),
                                             encoding="utf8")
    print(f"\nwrote {ART / 'arm_b_viability.json'}")
    print(f"wrote {ART / 'dataset_format.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
