"""Extract the SICKLE archive to disk, with progress and resume.

3.4M small files on NTFS is slow, so this reports progress, skips files that
already exist at the right size, and can be re-run after an interruption.

Target layout (matches the reference implementation's expectations):

    <out>/images/{L8,S1,S2}/npy/<UNIQUE_ID>/<acquisition>.npz
    <out>/masks/{3m,10m,30m}/<UNIQUE_ID>.tif
    <out>/sickle_dataset_tabular.csv

    python scripts/extract_dataset.py --out D:/FPD/sickle_data
    python scripts/extract_dataset.py --out D:/FPD/sickle_data --only masks,S2
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

ZIP = Path("D:/FPD/sickle_dataset-002.zip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/FPD/sickle_data")
    ap.add_argument("--only", default=None,
                    help="comma list from masks,S2,S1,L8 (default: everything)")
    ap.add_argument("--uids", default=None,
                    help="file with one UNIQUE_ID per line; restricts extraction")
    ap.add_argument("--raw-tif", action="store_true",
                    help="also extract images/<SAT>/tif/ -- 2.99M raw per-band "
                         "GeoTIFFs that the reference dataloader never reads. "
                         "Off by default; they are a redundant second copy of "
                         "the npz arrays and cost ~15x the file count.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(ZIP)
    infos = z.infolist()

    want = None
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}

    uids = None
    if args.uids:
        uids = {l.strip() for l in Path(args.uids).read_text().split() if l.strip()}

    def keep(name: str) -> bool:
        # the reference dataloader globs images/<SAT>/npy/<uid>/*.npz only
        if not args.raw_tif and "/tif/" in name:
            return False
        if want is not None:
            if name.startswith("masks/"):
                if "masks" not in want:
                    return False
            elif name.startswith("images/"):
                sat = name.split("/")[1] if name.count("/") > 1 else ""
                if sat not in want:
                    return False
        if uids is not None:
            if name.startswith("masks/"):
                return Path(name).stem in uids
            if name.startswith("images/") and name.count("/") >= 4:
                return name.split("/")[3] in uids
        return True

    todo = [i for i in infos if not i.is_dir() and keep(i.filename)]
    total = len(todo)
    total_bytes = sum(i.file_size for i in todo)
    print(f"archive  : {ZIP}")
    print(f"target   : {out}")
    print(f"selected : {total:,} files, {total_bytes / 1e9:.2f} GB uncompressed")
    sys.stdout.flush()

    t0 = time.time()
    done = skipped = 0
    for i, info in enumerate(todo, 1):
        dest = out / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as dst:
                dst.write(src.read())
            done += 1
        if i % 20000 == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (total - i) / rate if rate else 0
            print(f"  {i:>9,}/{total:,}  ({100 * i / total:5.1f}%)  "
                  f"written {done:,} skipped {skipped:,}  "
                  f"{rate:,.0f} files/s  ETA {eta / 60:.1f} min", flush=True)

    el = time.time() - t0
    print(f"\nDONE  {done:,} written, {skipped:,} already present, "
          f"{el / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
