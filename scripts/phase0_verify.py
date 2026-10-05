"""PHASE 0 — verify every claimed fact against the actual artefacts.

The master prompt (docs/SICKLE_MASTER_PROMPT.md Section 1) asserts a set of
"verified facts". This script re-derives each one from the raw files and reports
CONFIRMED / REFUTED / UNVERIFIABLE. Nothing downstream may rely on a fact this
script has not confirmed.

    python scripts/phase0_verify.py
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

FPD = Path("D:/FPD")
ZIP = FPD / "sickle_dataset-002.zip"
RUNS = FPD / "runs" / "wacv_2024"
TAB = Path("C:/ds/sickle/sickle_dataset_tabular.csv")
OUT = Path(__file__).resolve().parents[1] / "artifacts"

CHECKS: list[dict] = []


def check(name: str, claimed, actual, ok: bool | None = None) -> None:
    if ok is None:
        ok = (claimed == actual)
    status = "CONFIRMED" if ok else "REFUTED"
    CHECKS.append({"check": name, "claimed": str(claimed),
                   "actual": str(actual), "status": status})
    mark = "OK " if ok else "XX "
    print(f"  [{mark}] {name:46s} claimed={claimed!s:<22s} actual={actual}")


def note(name: str, value) -> None:
    CHECKS.append({"check": name, "claimed": "-", "actual": str(value),
                   "status": "RECORDED"})
    print(f"  [rec] {name:46s} {value}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- files
    print("\n1. ARTEFACT PRESENCE")
    for label, p in [("dataset zip", ZIP), ("released runs", RUNS),
                     ("tabular csv", TAB), ("paper", FPD / "2312.00069v1.pdf"),
                     ("toy dataset", FPD / "WACV 2024" / "sickle_toy_dataset.zip")]:
        check(f"exists: {label}", True, p.exists())
    if ZIP.exists():
        note("zip size (GB)", round(ZIP.stat().st_size / 1e9, 2))

    # ---------------------------------------------------------------- tabular
    if not TAB.exists():
        print("\nTABULAR CSV MISSING — cannot verify counts")
        return 1

    print("\n2. TABULAR COUNTS")
    rows = list(csv.DictReader(TAB.open(newline="", encoding="utf8")))
    check("samples total", 2370, len(rows))

    cols = list(rows[0].keys())
    expected_cols = ["UNIQUE_ID", "PLOT_ID", "RIVER_PART", "STANDARD_SEASON",
                     "YEAR", "AREA", "CROP", "VARIETY", "PADDY_BIN",
                     "SOWING_DATE", "TRANSPLANTING_DATE", "HARVESTING_DATE",
                     "SOWING_DAY", "TRANSPLANTING_DAY", "HARVESTING_DAY",
                     "YIELD", "SPLIT"]
    check("column set", expected_cols, cols)

    plots = {r["PLOT_ID"] for r in rows if r["PLOT_ID"]}
    check("unique PLOT_ID", 351, len(plots))

    split_samples = Counter(r["SPLIT"] for r in rows)
    check("train samples", 1937, split_samples.get("train"))
    check("val samples", 227, split_samples.get("val"))
    check("test samples", 206, split_samples.get("test"))

    split_plots = defaultdict(set)
    for r in rows:
        if r["PLOT_ID"]:
            split_plots[r["SPLIT"]].add(r["PLOT_ID"])
    check("train plots", 315, len(split_plots.get("train", set())))
    check("val plots", 36, len(split_plots.get("val", set())))
    check("test plots", 0, len(split_plots.get("test", set())))

    # ------------------------------------------------- test split unlabelled
    print("\n3. TEST SPLIT LABELS")
    test_rows = [r for r in rows if r["SPLIT"] == "test"]
    for col in ("PLOT_ID", "CROP", "PADDY_BIN"):
        blank = sum(1 for r in test_rows if not r[col].strip())
        check(f"test rows with blank {col}", len(test_rows), blank)

    # ---------------------------------------------------- split independence
    print("\n4. SPLIT INTEGRITY")
    ps = defaultdict(set)
    for r in rows:
        if r["PLOT_ID"]:
            ps[r["PLOT_ID"]].add(r["SPLIT"])
    straddle = [p for p, s in ps.items() if len(s) > 1]
    check("plots straddling splits", 0, len(straddle))

    # ------------------------------------------------------ crop + rotation
    print("\n5. CROP DISTRIBUTION AND ROTATION")
    crop_plots = defaultdict(set)
    for r in rows:
        if r["PLOT_ID"]:
            crop_plots[r["CROP"]].add(r["PLOT_ID"])
    for crop, n in [("Paddy", 262), ("Banana", 37), ("Coconut", 19)]:
        check(f"plots growing {crop}", n, len(crop_plots.get(crop, set())))

    val_plots = split_plots.get("val", set())
    for crop, n in [("Paddy", 29), ("Banana", 4), ("Coconut", 3)]:
        check(f"val plots growing {crop}", n,
              len(crop_plots.get(crop, set()) & val_plots))

    pb_plots = defaultdict(set)
    for r in rows:
        if r["PLOT_ID"]:
            pb_plots[r["PADDY_BIN"]].add(r["PLOT_ID"])
    n_paddy = len(pb_plots.get("1.0", set()))
    n_non = len(pb_plots.get("2.0", set()))
    check("plots with a paddy label", 262, n_paddy)
    check("plots with a non-paddy label", 105, n_non)

    both = [p for p, s in
            {p: {r["PADDY_BIN"] for r in rows if r["PLOT_ID"] == p}
             for p in plots}.items() if len(s) > 1]
    check("plots with BOTH labels (rotation)", 16, len(both))
    check("reconciliation 262+105 == 351+16", 367, len(plots) + len(both))

    multi_crop = sum(1 for p in plots
                     if len({r["CROP"] for r in rows if r["PLOT_ID"] == p}) > 1)
    check("plots growing >1 crop", 24, multi_crop)

    # ---------------------------------------------------------------- yield
    print("\n6. YIELD")
    y = [r for r in rows if r["YIELD"] not in ("", "0.0", "0")]
    check("yield samples", 319, len(y))
    check("yield plots", 131, len({r["PLOT_ID"] for r in y if r["PLOT_ID"]}))

    # ------------------------------------------------------- released runs
    print("\n7. RELEASED RUNS")
    if RUNS.exists():
        per_task = {}
        total = 0
        for t in sorted(RUNS.iterdir()):
            if t.is_dir():
                n = sum(1 for d in t.iterdir() if d.is_dir())
                per_task[t.name] = n
                total += n
        check("released run directories", 78, total)
        for t, n in per_task.items():
            expect = {"crop_type": 13, "crop_yield": 26, "sowing_date": 13,
                      "transplanting_date": 13, "harvesting_date": 13}
            check(f"  runs in {t}", expect.get(t, "?"), n)

        ct = RUNS / "crop_type" / "[S2]_unet3d"
        if (ct / "conf.json").exists():
            c = json.loads((ct / "conf.json").read_text(encoding="utf8"))
            check("conf num_classes", 2, c.get("num_classes"))
            check("conf img_size", [32, 32], c.get("img_size"))
            check("conf epochs", 100, c.get("epochs"))
            check("conf batch_size", 32, c.get("batch_size"))
            check("conf lr", 0.1, c.get("lr"))
            check("conf seed", 0, c.get("seed"))
            check("conf ignore_index", -999, c.get("ignore_index"))
            note("conf N_params", c.get("N_params"))
        if (ct / "trainlog.json").exists():
            log = json.loads((ct / "trainlog.json").read_text(encoding="utf8"))
            check("trainlog epochs", 100, len(log))
            ep = sorted(log, key=int)
            best = max(ep, key=lambda k: log[k].get("val_iou", -1))
            note("[S2]_unet3d best val_iou", round(log[best]["val_iou"], 4))
            note("[S2]_unet3d final val_iou", round(log[ep[-1]]["val_iou"], 4))
            note("[S2]_unet3d final train_iou", round(log[ep[-1]]["train_iou"], 4))
            note("[S2]_unet3d final gap", round(
                log[ep[-1]]["train_iou"] - log[ep[-1]]["val_iou"], 4))
    else:
        check("released runs directory", True, False)

    # ------------------------------------------------------------- zip tree
    print("\n8. ZIP STRUCTURE (reads the central directory only)")
    if ZIP.exists():
        z = zipfile.ZipFile(ZIP)
        names = z.namelist()
        check("zip entries", 3428250, len(names))
        ext = Counter(("." + n.rsplit(".", 1)[-1]) if "." in n.split("/")[-1]
                      else "DIR" for n in names)
        check("npz files", 209039, ext.get(".npz"))
        check("tif files", 2995937, ext.get(".tif"))
        tops = sorted({n.split("/")[0] for n in names})
        check("top-level entries", ["images", "masks",
                                    "sickle_dataset_tabular.csv"], tops)
        sats = sorted({n.split("/")[1] for n in names
                       if n.startswith("images/") and n.count("/") > 1})
        check("image modalities", ["L8", "S1", "S2"], sats)
        res = sorted({n.split("/")[1] for n in names
                      if n.startswith("masks/") and n.count("/") > 1})
        check("mask resolutions", ["10m", "30m", "3m"], res)
        for s in sats:
            note(f"  npz under images/{s}",
                 sum(1 for n in names if n.startswith(f"images/{s}/")
                     and n.endswith(".npz")))

    # ---------------------------------------------------------------- report
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "phase0_verification.csv").open("w", newline="",
                                                encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "claimed", "actual", "status"])
        w.writeheader()
        w.writerows(CHECKS)

    conf = sum(1 for c in CHECKS if c["status"] == "CONFIRMED")
    ref = sum(1 for c in CHECKS if c["status"] == "REFUTED")
    rec = sum(1 for c in CHECKS if c["status"] == "RECORDED")
    print("\n" + "=" * 72)
    print(f"CONFIRMED {conf}   REFUTED {ref}   RECORDED {rec}")
    if ref:
        print("\nREFUTED CHECKS — these must be resolved before implementation:")
        for c in CHECKS:
            if c["status"] == "REFUTED":
                print(f"  - {c['check']}: claimed {c['claimed']}, "
                      f"actual {c['actual']}")
    print(f"\nwrote {OUT / 'phase0_verification.csv'}")
    return 1 if ref else 0


if __name__ == "__main__":
    raise SystemExit(main())
