"""PHASE 1b — determine what the six mask bands actually mean.

The masks are 6-band float64 GeoTIFFs. The tabular file carries exactly six
per-sample quantities (CROP, PADDY_BIN, SOWING_DAY, TRANSPLANTING_DAY,
HARVESTING_DAY, YIELD), which suggests a mapping -- but suggestion is not
evidence. This tests the correspondence empirically against many samples and
reports the result. Nothing is assumed.

Method: for each sampled mask, take the set of non-zero values in each band and
check whether the sample's tabular value for a candidate field appears in it.
A band that matches a field on (almost) every sample is that field.

Note: a chip may contain neighbouring plots, so a band can legitimately hold
several values; the test asks for CONTAINMENT, not equality.

    python scripts/phase1_mask_semantics.py --n 300
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

ZIP = Path("D:/FPD/sickle_dataset-002.zip")
# Was hard-coded to C:/ds/sickle/, a path outside this project left over
# from an earlier session. Resolved from configs/base.yaml instead so
# every script reads the same dataset as the ladder.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import data_root as _data_root  # noqa: E402
TAB = _data_root() / "sickle_dataset_tabular.csv"
ART = Path(__file__).resolve().parents[1] / "artifacts"

CANDIDATES = ["PADDY_BIN", "SOWING_DAY", "TRANSPLANTING_DAY",
              "HARVESTING_DAY", "YIELD"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--res", default="10m")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)

    rows = {r["UNIQUE_ID"]: r
            for r in csv.DictReader(TAB.open(newline="", encoding="utf8"))}
    z = zipfile.ZipFile(ZIP)
    masks = [n for n in z.namelist()
             if n.startswith(f"masks/{args.res}/") and n.endswith(".tif")]
    ids = sorted((n.split("/")[-1][:-4] for n in masks), key=lambda s: int(s))
    print(f"masks under masks/{args.res}: {len(ids)}   tabular rows: {len(rows)}")
    missing = [i for i in ids if i not in rows]
    print(f"mask ids with no tabular row: {len(missing)}")
    no_mask = [i for i in rows if i not in set(ids)]
    print(f"tabular rows with no mask   : {len(no_mask)}")
    print("  -> masks are indexed by UNIQUE_ID"
          if not missing else "  -> indexing is NOT UNIQUE_ID; investigate")

    use = [i for i in ids if i in rows][: args.n]
    print(f"\ntesting {len(use)} samples\n")

    hits = defaultdict(lambda: defaultdict(int))
    seen = defaultdict(int)
    nband = None
    crop_vals = defaultdict(set)

    for uid in use:
        a = tifffile.imread(io.BytesIO(z.read(f"masks/{args.res}/{uid}.tif")))
        if nband is None:
            nband = a.shape[0]
        r = rows[uid]
        for b in range(a.shape[0]):
            vals = set(np.unique(a[b]).tolist()) - {0.0}
            for f in CANDIDATES:
                raw = (r.get(f) or "").strip()
                if not raw:
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if v == 0.0:
                    continue          # 0 is indistinguishable from background
                seen[(b, f)] += 1
                if any(abs(v - x) < 1e-6 for x in vals):
                    hits[b][f] += 1
        # band 0 vs crop name, collected for a separate look
        crop_vals[r["CROP"]] |= set(np.unique(a[0]).tolist()) - {0.0}

    print(f"bands per mask: {nband}\n")
    print(f"{'band':>5s}  " + "".join(f"{f:>20s}" for f in CANDIDATES))
    print("-" * (7 + 20 * len(CANDIDATES)))
    mapping = {}
    for b in range(nband or 0):
        line = f"{b:5d}  "
        best, best_rate = None, 0.0
        for f in CANDIDATES:
            n = seen.get((b, f), 0)
            h = hits[b].get(f, 0)
            rate = (h / n) if n else float("nan")
            line += f"{(f'{100*rate:.0f}% ({h}/{n})' if n else '-'):>20s}"
            if n >= 10 and rate > best_rate:
                best, best_rate = f, rate
        print(line)
        if best and best_rate >= 0.95:
            mapping[b] = {"field": best, "match_rate": round(best_rate, 4)}

    print("\nband 0 value sets by crop (first 10 crops):")
    for crop, vals in list(sorted(crop_vals.items(),
                                  key=lambda kv: -len(kv[1])))[:10]:
        print(f"  {crop[:22]:22s} -> {sorted(vals)[:8]}")

    out = {
        "resolution": args.res,
        "n_samples_tested": len(use),
        "bands": nband,
        "mask_index": ("UNIQUE_ID" if not missing else "UNKNOWN"),
        "masks_without_tabular_row": len(missing),
        "tabular_rows_without_mask": len(no_mask),
        "confident_band_mapping": {str(k): v for k, v in mapping.items()},
        "band0_values_by_crop": {k: sorted(v) for k, v in crop_vals.items()},
        "caveat": ("A chip may contain neighbouring plots, so a band can hold "
                   "several values. The test asks whether the sample's own "
                   "value is CONTAINED in the band, not whether it is the only "
                   "value. Band 0 appears to be a crop/class code and is "
                   "resolved separately below."),
    }
    (ART / "mask_semantics.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf8")
    print(f"\nwrote {ART / 'mask_semantics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
