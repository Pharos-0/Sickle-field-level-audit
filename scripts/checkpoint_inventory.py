"""Inventory every model checkpoint the study used or produced.

    python scripts/checkpoint_inventory.py

Writes docs/CHECKPOINT_INVENTORY.csv (one row per checkpoint file) and
docs/CHECKPOINT_INVENTORY.md (a summary and how to obtain each group).

WHY
---
Neither the dataset nor the authors' released weights may be redistributed:
both are obtained through the SICKLE authors' access form, and the reference
code ships no licence. Our own checkpoints were trained on that data. So the
weights are never committed; instead each file is identified by size and
SHA-256, so a researcher who obtains them can confirm they hold the same
files, and can see which ones a result depended on.

GROUPS
------
  released     the authors' weights.zip, unpacked under RUNS_RELEASED
               (default D:/FPD/runs/wacv_2024; override with the
               SICKLE_RUNS environment variable)
  ladder       the ten-rung ablation, outputs/crop_type/v{k}_seed{s}/
  sweep        the data-regime sweep, outputs/crop_type/sweep/
  interrupted  runs killed by out-of-memory, outputs/_interrupted_runs/

A run is "complete" when its best.json says so. An incomplete run is listed
but not hashed, because a run still training rewrites its checkpoint.
Released runs suffixed _debug are the authors' two-epoch smoke tests and are
excluded from every count in the papers; they are listed and flagged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RUNS_RELEASED = Path(os.environ.get("SICKLE_RUNS",
                                    "D:/FPD/runs/wacv_2024"))
OUT_CSV = PROJ / "docs" / "CHECKPOINT_INVENTORY.csv"
OUT_MD = PROJ / "docs" / "CHECKPOINT_INVENTORY.md"
FIELDS = ["group", "task", "run", "sensors", "architecture", "variant",
          "seed", "file", "complete", "debug", "size_bytes", "sha256",
          "source", "redistributable"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(run_dir: Path) -> bool | None:
    b = run_dir / "best.json"
    if not b.exists():
        return None
    try:
        return bool(json.loads(b.read_text(encoding="utf8")).get("complete"))
    except (OSError, ValueError):
        return None


def parse_released(name: str):
    """'[S2,L8,S1]_utae_season' -> ('S2+L8+S1', 'utae', 'season')."""
    m = re.match(r"\[([^\]]+)\]_([a-z0-9]+)(?:_(season))?(?:_(debug))?$", name)
    if not m:
        return "", "", "", False
    return (m.group(1).replace(",", "+"), m.group(2), m.group(3) or "",
            bool(m.group(4)))


def from_config(run_dir: Path):
    c = run_dir / "config_resolved.json"
    if not c.exists():
        return "", "", ""
    cfg = json.loads(c.read_text(encoding="utf8"))
    return ("+".join(cfg.get("satellites", {})),
            cfg.get("model", {}).get("name", ""),
            cfg.get("data", {}).get("task", ""))


def rows():
    out = []
    # ---- released
    if RUNS_RELEASED.exists():
        for f in sorted(RUNS_RELEASED.glob("*/*/*.pth*")):
            run_dir, task = f.parent, f.parent.parent.name
            sens, arch, var, dbg = parse_released(run_dir.name)
            out.append({"group": "released", "task": task,
                        "run": run_dir.name, "sensors": sens,
                        "architecture": arch, "variant": var or "per-sample",
                        "seed": "", "file": f, "complete": True,
                        "debug": dbg,
                        "source": "SICKLE authors, weights.zip via their "
                                  "access form",
                        "redistributable": "no"})
    else:
        print(f"  released weights not found at {RUNS_RELEASED}; "
              f"set SICKLE_RUNS")
    # ---- ours
    for f in sorted((PROJ / "outputs").rglob("*.pth*")):
        rel = f.relative_to(PROJ / "outputs").as_posix()
        run_dir = f.parent
        sens, arch, task = from_config(run_dir)
        if rel.startswith("_interrupted_runs/"):
            group = "interrupted"
        elif "/sweep/" in rel:
            group = "sweep"
        else:
            group = "ladder"
        m = re.search(r"seed0*(\d+)$", run_dir.name)
        out.append({"group": group, "task": task or "crop_type",
                    "run": run_dir.relative_to(PROJ / "outputs").as_posix(),
                    "sensors": sens, "architecture": arch, "variant": "",
                    "seed": m.group(1) if m else "", "file": f,
                    "complete": complete(run_dir), "debug": False,
                    "source": "trained in this study (scripts/train.py) on "
                              "SICKLE data",
                    "redistributable": "no (derived from access-controlled "
                                       "data)"})
    return out


def main() -> int:
    rs = rows()
    for i, r in enumerate(rs, 1):
        f = r["file"]
        r["size_bytes"] = f.stat().st_size
        # never hash a run that may still be writing its checkpoint
        r["sha256"] = sha256(f) if r["complete"] else ""
        r["file"] = (f.relative_to(PROJ).as_posix() if PROJ in f.parents
                     else f"<SICKLE_RUNS>/"
                          f"{f.relative_to(RUNS_RELEASED).as_posix()}")
        if i % 50 == 0:
            print(f"  hashed {i}/{len(rs)}", flush=True)
    with open(OUT_CSV, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rs:
            w.writerow({k: r[k] for k in FIELDS})

    by = Counter((r["group"], r["task"]) for r in rs)
    size = Counter()
    for r in rs:
        size[r["group"]] += r["size_bytes"]
    n_dbg = sum(r["debug"] for r in rs)
    n_inc = sum(1 for r in rs if not r["complete"])
    lines = [
        "# Checkpoint inventory",
        "",
        "Generated by `scripts/checkpoint_inventory.py`; one row per file in "
        "`CHECKPOINT_INVENTORY.csv` with size and SHA-256. **No checkpoint is "
        "committed.** Neither the SICKLE data nor the released weights may "
        "be redistributed, and our checkpoints are derived from that data.",
        "",
        "| group | task | files | size |",
        "|---|---|---:|---:|",
    ]
    for (g, t), k in sorted(by.items()):
        lines.append(f"| {g} | {t} | {k} | "
                     f"{sum(r['size_bytes'] for r in rs if r['group'] == g and r['task'] == t) / 2**20:,.0f} MB |")
    lines += [
        "",
        f"{n_dbg} released files are the authors' `_debug` smoke tests "
        "(two epochs) and are excluded from every count in the papers. "
        f"{n_inc} file(s) belong to runs not marked complete and are listed "
        "without a checksum.",
        "",
        "## How to obtain each group",
        "",
        "- **Dataset and released weights.** Request access through the form "
        "linked from the SICKLE repository README "
        "(github.com/Depanshu-Sani/SICKLE), then unzip `weights.zip` and point "
        "`SICKLE_RUNS` at its `runs/wacv_2024` folder. Compare "
        "checksums with the CSV.",
        "- **Ladder, sweep and interrupted runs.** Retrain with the committed "
        "configs (`configs/`) and `scripts/train.py` (the full sets: "
        "`scripts/run_ladder_py.py`, `scripts/run_sweep.py`); each run's "
        "`manifest.json`, `config_resolved.json`, `trainlog.json` and "
        "`best.json` are committed and record the seed, commit and settings. "
        "Retrained weights will not be bit-identical (GPU nondeterminism), so "
        "their checksums will differ; the committed per-run metrics are the "
        "reference to compare against.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"wrote {OUT_CSV.relative_to(PROJ)} ({len(rs)} files) and "
          f"{OUT_MD.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
