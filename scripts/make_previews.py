"""Render human-viewable previews of the SICKLE data.

Reads straight from the archive, so it does not wait for extraction.

Produces, under results/previews/:

  overview_<uid>.png     one sample: RGB over its season, the six mask bands
                         decoded, and the plot outline
  timeseries_<uid>.png   the full S2 sequence for one sample as an RGB strip
  modalities_<uid>.png   S2 RGB vs S1 VV/VH vs L8 for the same plot and date
  contact_sheet.png      many plots at once, paddy and non-paddy

    python scripts/make_previews.py
    python scripts/make_previews.py --uids 5,91,2191
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap

ZIP = Path("D:/FPD/sickle_dataset-002.zip")
# Was hard-coded to C:/ds/sickle/, a path outside this project left over
# from an earlier session. Resolved from configs/base.yaml instead so
# every script reads the same dataset as the ladder.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import data_root as _data_root  # noqa: E402
TAB = _data_root() / "sickle_dataset_tabular.csv"
OUT = Path(__file__).resolve().parents[1] / "results" / "previews"

#: verified in docs/IMPLEMENTATION_AUDIT.md section 3
BANDS = ["PLOT_ID", "crop code", "sowing day", "transplanting day",
         "harvesting day", "yield / pixel"]
CROP_CODE = {1: "Paddy", 2: "Banana", 3: "Coconut", 4: "Banyan", 5: "Sugarcane",
             6: "Palm tree", 7: "Groundnut", 8: "Eucalyptus",
             9: "Tamarind tree", 10: "Mixed plantation", 11: "Cotton",
             15: "Betelwine"}


def load_tab() -> dict[str, dict]:
    return {r["UNIQUE_ID"]: r
            for r in csv.DictReader(TAB.open(newline="", encoding="utf8"))}


def s2_rgb(z: zipfile.ZipFile, path: str) -> np.ndarray | None:
    """True-colour composite from B4/B3/B2, percentile-stretched."""
    try:
        d = np.load(io.BytesIO(z.read(path)), allow_pickle=False)
    except KeyError:
        return None
    try:
        r, g, b = d["B4"], d["B3"], d["B2"]
    except KeyError:
        return None
    img = np.dstack([r, g, b]).astype(np.float32)
    if not np.isfinite(img).any() or img.max() == 0:
        return None
    lo, hi = np.percentile(img[img > 0], [2, 98]) if (img > 0).any() else (0, 1)
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)


def seq_paths(z: zipfile.ZipFile, sat: str, uid: str) -> list[str]:
    pre = f"images/{sat}/npy/{uid}/"
    return sorted(n for n in z.namelist()
                  if n.startswith(pre) and n.endswith(".npz"))


def read_mask(z: zipfile.ZipFile, uid: str, res: str = "10m") -> np.ndarray | None:
    try:
        return tifffile.imread(io.BytesIO(z.read(f"masks/{res}/{uid}.tif")))
    except KeyError:
        return None


def overview(z, uid: str, row: dict) -> bool:
    m = read_mask(z, uid)
    paths = seq_paths(z, "S2", uid)
    if m is None or not paths:
        return False

    picks = [paths[i] for i in
             np.linspace(0, len(paths) - 1, min(6, len(paths))).astype(int)]
    fig, axes = plt.subplots(2, 6, figsize=(17, 7.4))

    for ax, p in zip(axes[0], picks):
        img = s2_rgb(z, p)
        if img is None:
            ax.text(0.5, 0.5, "no RGB", ha="center", va="center")
        else:
            ax.imshow(img)
        ax.set_title(Path(p).stem.split("_")[0][:8], fontsize=8)
        ax.axis("off")
    for ax in axes[0][len(picks):]:
        ax.axis("off")
    axes[0][0].set_ylabel("S2 RGB")

    pid = float(row["PLOT_ID"]) if row["PLOT_ID"] else None
    for i, ax in enumerate(axes[1]):
        band = m[i]
        if i == 0 and pid is not None:
            ax.imshow(band, cmap="tab20")
            ax.contour(band == pid, levels=[0.5], colors="red", linewidths=1.6)
        elif i == 1:
            ax.imshow(band, cmap=ListedColormap(
                plt.cm.tab20(np.linspace(0, 1, 20))), vmin=0, vmax=16)
        else:
            ax.imshow(band, cmap="viridis")
        nz = band[band != 0]
        rng = f"[{nz.min():.0f}, {nz.max():.0f}]" if nz.size else "all zero"
        ax.set_title(f"band {i}\n{BANDS[i]}\n{rng}", fontsize=7.5)
        ax.axis("off")

    crop = row["CROP"] or "?"
    title = (f"UID {uid} · plot {row['PLOT_ID']} · {crop} · "
             f"{row['STANDARD_SEASON']} {row['YEAR'][:4]} · "
             f"{row['RIVER_PART'] or 'region n/a'} · split={row['SPLIT']} · "
             f"{len(paths)} S2 acquisitions")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.945, "red outline = this sample's own plot (band 0 == PLOT_ID)",
             ha="center", fontsize=8.5, color="#c0392b")
    fig.subplots_adjust(hspace=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.2)
    fig.savefig(OUT / f"overview_{uid}.png", dpi=150)
    plt.close(fig)
    return True


def timeseries(z, uid: str, row: dict, maxn: int = 24) -> bool:
    paths = seq_paths(z, "S2", uid)
    if not paths:
        return False
    picks = paths[:maxn]
    n = len(picks)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            img = s2_rgb(z, picks[k])
            if img is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img)
            ax.set_title(Path(picks[k]).stem.split("_")[0][:8], fontsize=7)
        ax.axis("off")
    fig.suptitle(f"S2 time series · UID {uid} · plot {row['PLOT_ID']} · "
                 f"{row['CROP']} · {len(paths)} acquisitions "
                 f"(showing {n})", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"timeseries_{uid}.png", dpi=140)
    plt.close(fig)
    return True


def modalities(z, uid: str, row: dict) -> bool:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    s2 = seq_paths(z, "S2", uid)
    img = s2_rgb(z, s2[len(s2) // 2]) if s2 else None
    if img is not None:
        axes[0].imshow(img)
    axes[0].set_title(f"Sentinel-2 RGB\n10 m, {len(s2)} dates", fontsize=9)

    s1 = seq_paths(z, "S1", uid)
    if s1:
        d = np.load(io.BytesIO(z.read(s1[len(s1) // 2])), allow_pickle=False)
        for ax, k in zip(axes[1:3], ("VV", "VH")):
            if k in d.files:
                a = d[k]
                ax.imshow(a, cmap="gray")
                ax.set_title(f"Sentinel-1 {k}\nradar, {len(s1)} dates\n"
                             f"[{a.min():.1f}, {a.max():.1f}] dB", fontsize=9)
    l8 = seq_paths(z, "L8", uid)
    if l8:
        d = np.load(io.BytesIO(z.read(l8[len(l8) // 2])), allow_pickle=False)
        if "SR_B4" in d.files:
            try:
                rgb = np.dstack([d["SR_B4"], d["SR_B3"], d["SR_B2"]]).astype(np.float32)
                lo, hi = np.percentile(rgb, [2, 98])
                axes[3].imshow(np.clip((rgb - lo) / max(hi - lo, 1), 0, 1))
            except KeyError:
                pass
        axes[3].set_title(f"Landsat-8 RGB\n30 m, {len(l8)} dates", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Three sensors, same plot · UID {uid} · plot {row['PLOT_ID']} "
                 f"· {row['CROP']}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"modalities_{uid}.png", dpi=150)
    plt.close(fig)
    return True


def contact_sheet(z, rows: dict, uids: list[str]) -> None:
    n = len(uids)
    cols = 6
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(2.3 * cols, 2.7 * r))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        uid = uids[k]
        s2 = seq_paths(z, "S2", uid)
        img = s2_rgb(z, s2[len(s2) // 2]) if s2 else None
        if img is not None:
            ax.imshow(img)
        row = rows[uid]
        is_paddy = (row["PADDY_BIN"] == "1.0")
        ax.set_title(f"{row['CROP'][:14]}\nuid {uid}", fontsize=7.5,
                     color="#1e7e34" if is_paddy else "#c0392b")
        ax.axis("off")
    fig.suptitle("SICKLE plots · green = paddy, red = non-paddy "
                 "(Sentinel-2 RGB, mid-season)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "contact_sheet.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", default=None)
    ap.add_argument("--n-sheet", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_tab()
    z = zipfile.ZipFile(ZIP)
    have = {n.split("/")[3] for n in z.namelist()
            if n.startswith("images/S2/npy/") and n.count("/") >= 4}

    if args.uids:
        picks = [u.strip() for u in args.uids.split(",")]
    else:
        paddy = [u for u, r in rows.items()
                 if r["PADDY_BIN"] == "1.0" and u in have]
        other = [u for u, r in rows.items()
                 if r["PADDY_BIN"] == "2.0" and u in have]
        picks = [paddy[0], other[0]]
        yld = [u for u, r in rows.items()
               if (r["YIELD"] or "0") not in ("", "0.0", "0") and u in have]
        if yld:
            picks.append(yld[0])

    print(f"samples with S2 imagery: {len(have):,}")
    made = []
    for uid in picks:
        if uid not in rows:
            print(f"  uid {uid}: not in tabular, skipped")
            continue
        r = rows[uid]
        ok1 = overview(z, uid, r)
        ok2 = timeseries(z, uid, r)
        ok3 = modalities(z, uid, r)
        print(f"  uid {uid:>5s}  {r['CROP'][:16]:16s} "
              f"overview={ok1} timeseries={ok2} modalities={ok3}")
        made.append(uid)

    paddy = [u for u, r in rows.items()
             if r["PADDY_BIN"] == "1.0" and u in have][: args.n_sheet // 2]
    other = [u for u, r in rows.items()
             if r["PADDY_BIN"] == "2.0" and u in have][: args.n_sheet // 2]
    sheet = paddy + other
    if sheet:
        contact_sheet(z, rows, sheet)
        print(f"  contact sheet: {len(sheet)} plots")

    print(f"\nwrote {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
