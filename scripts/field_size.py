"""Per-field size for the validation plots, exactly as the metric weights them.

    python scripts/field_size.py

Needed for the sensitivity analysis in scripts/field_distributions.py.

WHY THIS IS NOT A ROUGH COUNT
-----------------------------
The quantity that matters is a plot's WEIGHT in the pixel-pooled metric:
the number of its pixels that survive the loader and enter the pooled
confusion matrix. That is not the raw mask footprint, because the reference
loader does two things to a mask before it is scored:

  1. `SICKLE_Dataset.__getitem__` resizes the 6-band mask from the raster
     grid (33x33 at 10 m) to `img_size` (32x32) with `albumentations.Resize`
     at `cv2.INTER_NEAREST`. Nearest-neighbour matters: it is what keeps plot
     IDs and class codes exact instead of interpolating them into values
     that match no class, and a plot one raster row wide can be dropped
     entirely or doubled by it.
  2. It then sets every pixel whose plot ID is not in this split's ID set to
     `ignore_index`, which removes neighbouring plots that happen to fall in
     the same patch.

(The sibling class `UNET3d_Dataset` resizes with the albumentations default,
which is bilinear; this study's configs instantiate `SICKLE_Dataset`, so
nearest-neighbour is the behaviour being reproduced here.)

This script reproduces both steps on the masks alone. No model and no
imagery are touched, so the counts are model-independent, and that is the
point: field size is a property of the data, not of a checkpoint. The total
is cross-checked against the `n_pixels` recorded by the released-checkpoint
plot-level evaluations, which is an independent path to the same number.

Two size measures are written, because they answer different questions:

  n_pixels_pooled  the plot's weight in the pixel-pooled metric
  area_acres       AREA from the tabular record -- reported physical size,
                   independent of the raster
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))

import albumentations as A          # noqa: E402
import cv2                          # noqa: E402
import rasterio                     # noqa: E402

ROOT = Path("D:/FPD/sickledataset extracted")
OUT = PROJ / "results" / "field_size_val.json"
RES, IMG, IGNORE, SPLIT = 10, 32, -999, "val"


def read_mask(uid: int) -> np.ndarray:
    """`SICKLE_Dataset._read_masks` followed by its resize, verbatim."""
    with rasterio.open(ROOT / "masks" / f"{RES}m" / f"{uid}.tif") as fp:
        m = fp.read()
    m[0][m[0] == 0] = -1
    m[1] -= 1
    m[1][m[1] > 1] = 1
    m[2] -= 1
    m[3] -= 1
    m[4] -= 1
    m[5][m[5] == 0] = -1
    m[m < -1] = 0
    m[m < 0] = IGNORE
    m = A.Resize(height=IMG, width=IMG,
                 interpolation=cv2.INTER_NEAREST)(
        image=m.transpose(1, 2, 0))["image"].transpose(2, 0, 1)
    return m


def main() -> int:
    df = pd.read_csv(ROOT / "sickle_dataset_tabular.csv")
    sub = df[df.SPLIT == SPLIT].reset_index(drop=True)
    # `self.plot_ids = set(df.PLOT_ID)` -- the split frame's own IDs
    plot_ids = set(sub.PLOT_ID)
    print(f"{len(sub)} {SPLIT} samples, "
          f"{len({int(p) for p in plot_ids if pd.notna(p)})} distinct plot ids")

    px, nsamp, missing = defaultdict(int), defaultdict(int), []
    n_scored = 0          # pixels entering the pooled confusion matrix
    n_unassigned = 0      # of those, pixels belonging to no retained plot
    for uid in sub.UNIQUE_ID.astype(int):
        f = ROOT / "masks" / f"{RES}m" / f"{uid}.tif"
        if not f.exists():
            missing.append(int(uid))
            continue
        m = read_mask(int(uid))
        pm, ct = m[0].copy(), m[1].copy()
        # step 2, verbatim: `mask[:, plot_mask == unmatched] = ignore_index`
        for u in set(np.unique(pm)[1:]) - plot_ids:
            ct[pm == u] = IGNORE
            pm[pm == u] = IGNORE
        # `confusion` keeps exactly `t != ignore_index`, t being crop_type
        scored = ct != IGNORE
        n_scored += int(scored.sum())
        # `plot_macro_iou` then skips plot ids that are ignore or negative
        assigned = scored & (pm != IGNORE) & (pm >= 0)
        n_unassigned += int((scored & ~assigned).sum())
        ids, counts = np.unique(pm[assigned], return_counts=True)
        for pid, c in zip(ids, counts):
            px[float(pid)] += int(c)
            nsamp[float(pid)] += 1
    if missing:
        print(f"  {len(missing)} masks not on disk: {missing[:6]} "
              f"-- recorded, not silently dropped")

    area = {}
    for pid, g in sub.dropna(subset=["PLOT_ID"]).groupby("PLOT_ID"):
        a = g["AREA"].dropna()
        area[float(pid)] = float(a.mean()) if len(a) else None

    rows = [{"plot": float(p), "n_pixels_pooled": px[p],
             "n_samples": nsamp[p], "area_acres": area.get(float(p))}
            for p in sorted(px)]
    v = np.array([r["n_pixels_pooled"] for r in rows], float)
    order = np.argsort(-v)
    tot = v.sum()

    print()
    print(f"{n_scored} pixels enter the pooled confusion matrix")
    print(f"  {int(tot)} of them ({tot / max(1, n_scored):.1%}) belong "
          f"to a retained plot, over {len(rows)} plots")
    print(f"  {n_unassigned} ({n_unassigned / max(1, n_scored):.1%}) are "
          f"scored by the pooled metric but assigned to no plot, so "
          f"the per-field metric never sees them")
    print(f"  per plot: min {v.min():.0f}  p25 {np.percentile(v,25):.0f}"
          f"  median {np.median(v):.0f}  p75 {np.percentile(v,75):.0f}"
          f"  max {v.max():.0f}   ratio max:min = {v.max()/max(1,v.min()):.0f}x")
    for lab, k in (("largest plot", 1),
                   ("top decile", max(1, len(v) // 10)),
                   ("top quartile", max(1, len(v) // 4))):
        # the count is printed, not asserted in the label: a hand-written
        # "4 largest" beside k = 36 // 10 = 3 is exactly the kind of silent
        # mismatch this audit exists to catch
        print(f"  {k:2d} largest ({lab:<12s}) carry "
              f"{v[order[:k]].sum() / tot:>6.1%} of all scored pixels")
    print(f"  an equal-weight field would carry {1/len(v):.1%}")

    # cross-check against an independent path to the same total
    xs = []
    for f in glob.glob(str(PROJ / "results/plot_level/*.json")):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        xs.append(int(d["pixel_pooled"]["n_pixels"]))
    if xs:
        u = sorted(set(xs))
        agree = len(u) == 1 and u[0] == int(tot)
        print(f"\n  CROSS-CHECK: released plot-level evaluations recorded "
              f"n_pixels = {u}")
        print(f"  this script, from masks alone: {int(tot)}  -> "
              f"{'EXACT MATCH' if agree else 'MISMATCH, see note'}")
        if not agree:
            print("  (a mismatch means this reconstruction of the loader is "
                  "wrong; the released figure is authoritative)")

    OUT.write_text(json.dumps(
        {"split": SPLIT, "mask_res_m": RES, "img_size": IMG,
         "n_plots": len(rows), "n_pixels_in_pooled_metric": n_scored,
         "n_pixels_assigned_to_a_plot": int(tot),
         "n_pixels_scored_but_unassigned": n_unassigned,
         "missing_masks": missing,
         "share_largest": float(v[order[0]] / tot),
         "share_top_decile": float(v[order[:max(1, len(v)//10)]].sum() / tot),
         "share_top_quartile": float(v[order[:max(1, len(v)//4)]].sum() / tot),
         "equal_weight_share": float(1 / len(v)),
         "cross_check_released_n_pixels": sorted(set(xs)) if xs else None,
         "cross_check_exact": (bool(len(set(xs)) == 1
                                    and list(set(xs))[0] == n_scored)
                               if xs else None),
         "plots": rows,
         "note": ("n_pixels_pooled is the plot's weight in the pixel-pooled "
                  "metric: its pixels surviving the reference loader's "
                  "bilinear mask resize and unmatched-plot removal, summed "
                  "over all of its validation samples.")},
        indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
