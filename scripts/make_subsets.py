"""Build the plot-level training subsets for the data-regime sweep.

The sweep's independent unit is the PLOT. Subsets are therefore drawn over
plots, never over samples, pixels or satellite observations.

Two properties are enforced:

  * **Nested.** For a given seed, the 50-plot set is a subset of the 100-plot
    set, which is a subset of the 150-plot set, and so on. Disjoint draws would
    confound the data-regime curve with sampling variation between points.
  * **Recorded composition.** Plots rotate crops between seasons (verified:
    24 plots grow more than one crop, 16 carry both a paddy and a non-paddy
    label), so a given plot count yields a variable number of season-samples
    AND a variable class mix. Both are written into every manifest, because a
    curve drawn against plot count while class balance drifts underneath it
    would be uninterpretable.

Validation is NOT subsampled: all 36 validation plots are used at every point,
so the evaluation set is identical across the sweep.

    python scripts/make_subsets.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("D:/FPD/sickle_data")
ART = Path(__file__).resolve().parents[1] / "artifacts" / "subsets"

PLOT_COUNTS = [50, 100, 150, 200, 250, 315]
SEEDS = [1, 2, 3, 4, 5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=",".join(map(str, PLOT_COUNTS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    counts = [int(c) for c in args.counts.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "sickle_dataset_tabular.csv")

    train = df[df.SPLIT == "train"]
    val = df[df.SPLIT == "val"]
    train_plots = np.array(sorted(train.PLOT_ID.dropna().unique()))
    val_plots = sorted(val.PLOT_ID.dropna().unique())

    print(f"train plots available : {len(train_plots)}")
    print(f"val plots (fixed)     : {len(val_plots)}")
    if max(counts) > len(train_plots):
        print(f"ERROR: requested {max(counts)} plots but only "
              f"{len(train_plots)} exist in train. 388 does not exist.")
        return 1

    index = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(train_plots)       # one ordering per seed
        for n in sorted(counts):
            sel = sorted(order[:n].tolist())       # nested by construction
            sub = train[train.PLOT_ID.isin(sel)]

            paddy = int((sub.PADDY_BIN == 1.0).sum())
            non_paddy = int((sub.PADDY_BIN == 2.0).sum())
            crops = Counter(sub.CROP.dropna())

            rec = {
                "seed": seed,
                "n_plots_requested": n,
                "n_plots": len(sel),
                "plot_ids": sel,
                "n_samples": int(len(sub)),
                "samples_paddy": paddy,
                "samples_non_paddy": non_paddy,
                "paddy_fraction": round(paddy / max(1, paddy + non_paddy), 4),
                "n_crops": len(crops),
                "crop_counts": dict(crops.most_common()),
                "years": sorted(int(y) for y in sub.YEAR.dropna().unique()),
                "river_parts": sorted(
                    str(r) for r in sub.RIVER_PART.dropna().unique() if str(r)),
                "val_plots_fixed": len(val_plots),
                "nested": True,
            }
            path = ART / f"plots_{n:03d}_seed{seed:03d}.json"
            path.write_text(json.dumps(rec, indent=2), encoding="utf8")
            index.append({k: rec[k] for k in
                          ("seed", "n_plots", "n_samples", "samples_paddy",
                           "samples_non_paddy", "paddy_fraction", "n_crops")})

    # ---- verify nesting actually holds
    print("\nverifying nesting ...")
    bad = 0
    for seed in seeds:
        prev: set | None = None
        for n in sorted(counts):
            ids = set(json.loads(
                (ART / f"plots_{n:03d}_seed{seed:03d}.json").read_text(
                    encoding="utf8"))["plot_ids"])
            if prev is not None and not prev.issubset(ids):
                print(f"  NESTING VIOLATION seed {seed} at n={n}")
                bad += 1
            prev = ids
    print(f"  {'OK - all nested' if not bad else f'{bad} violations'}")

    idx = pd.DataFrame(index)
    idx.to_csv(ART / "index.csv", index=False)

    print(f"\n{'plots':>6s}{'samples':>9s}{'paddy':>8s}{'non-pad':>9s}"
          f"{'paddy frac':>12s}{'crops':>7s}")
    for n in sorted(counts):
        g = idx[idx.n_plots == n]
        print(f"{n:6d}{g.n_samples.mean():9.1f}{g.samples_paddy.mean():8.1f}"
              f"{g.samples_non_paddy.mean():9.1f}"
              f"{g.paddy_fraction.mean():12.3f}{g.n_crops.mean():7.1f}"
              f"   (mean over {len(g)} seeds)")

    print(f"\nwrote {len(index)} manifests to {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
