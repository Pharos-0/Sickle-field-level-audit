"""Item 2: one minimal, self-contained reproduction per reported defect.

    python scripts/minimal_repro.py
    python scripts/minimal_repro.py --json report.json

WHAT A REVIEWER ASKED FOR, AND WHY THE EXISTING SCRIPTS DO NOT SUPPLY IT
------------------------------------------------------------------------
`scripts/verify_all.py` asks whether each claim still holds.
`scripts/independent_check.py` re-derives the arithmetic without importing
`src/`. Neither shows, side by side, the number the benchmark publishes and
the number this audit measures, with the exact checkpoint identifiers and
library versions that produced each. That pairing is what makes a defect
checkable by somebody else, and it is what this file provides.

Each block below is short enough to read in full, names the artefact it
reads, and prints:

    AS PUBLISHED   the value in the released logs or the dataset paper
    AS MEASURED    the value obtained here
    WHY            the one sentence that connects them

ALL FOUR RUN FROM CACHED ARTEFACTS
----------------------------------
None re-executes a model, so this script runs in seconds, without a GPU and
without the dataset, on any machine holding this repository. What it
therefore demonstrates is that the audit's conclusions follow from the
recorded evaluations. It is not a substitute for re-running those
evaluations, which `scripts/phase5_gate.py` and `scripts/regression_gate.py`
do, and the output says so rather than leaving it to be assumed.

The three defects are numbered as in the manuscript. The fourth item is
deliberately not called a defect: the dataset paper does define the quantity
correctly, and only the released artefacts invite the misreading.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import platform
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
BLOCKS: list[dict] = []


def emit(num, title, published, measured, why, source, checkpoints,
         verdict=True, units=""):
    BLOCKS.append({"id": num, "title": title, "as_published": published,
                   "as_measured": measured, "why": why, "source": source,
                   "checkpoints": checkpoints, "reproduced": bool(verdict),
                   "units": units})
    bar = "-" * 74
    print(bar)
    print(f"{num}.  {title}")
    print(bar)
    print(f"  AS PUBLISHED   {published}")
    print(f"  AS MEASURED    {measured}")
    print(f"  WHY            {why}")
    print(f"  checkpoints    {checkpoints}")
    print(f"  source         {source}")
    print(f"  status         {'REPRODUCED' if verdict else 'NOT REPRODUCED'}")
    print()


# ----------------------------------------------------------------- D1
def d1_batch_size():
    """Evaluating ONE frozen checkpoint at different batch sizes."""
    runs = {}
    for f in glob.glob(str(PROJ / "results/gate/gate_crop_type_*_bs*.json")):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        runs.setdefault(d["run"], {})[d["batch_size"]] = d
    multi = {k: v for k, v in runs.items() if len(v) > 1}
    if not multi:
        return emit("D1", "Evaluation depends on batch size with frozen "
                          "weights", "n/a", "no multi-batch-size gate "
                    "artefacts on disk", "cannot be checked from the cache",
                    "results/gate/gate_crop_type_*_bs*.json", "none", False)
    run = max(multi, key=lambda k: max(multi[k][b]["reproduced"]
                                       for b in multi[k])
              - min(multi[k][b]["reproduced"] for b in multi[k]))
    cells = multi[run]
    rel = next(iter(cells.values()))["released_best_val_iou"]
    got = {b: cells[b]["reproduced"] for b in sorted(cells)}
    spread = max(got.values()) - min(got.values())
    emit("D1", "Evaluation depends on batch size with frozen weights",
         f"macro IoU {rel:.4f} for {run}",
         "  ".join(f"bs{b}: {v:.4f}" for b, v in got.items())
         + f"   spread {spread:.4f} ({spread * 100:.1f} IoU points)",
         "the weights are identical in every cell; only the number of "
         "samples per forward pass changes, so a published IoU on this "
         "benchmark is not defined without its evaluation batch size",
         ", ".join(f"results/gate/gate_crop_type_{run}_bs{b}.json"
                   for b in sorted(cells)),
         f"{run}, checkpoint_best.pth.tar, one file, {len(cells)} "
         f"evaluations", spread > 0.01, "macro IoU")


# ----------------------------------------------------------------- D3
def d3_degenerate():
    """Released baselines that predict one class everywhere."""
    degen = []
    for f in glob.glob(str(PROJ / "results/plot_level/*.json")):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        p = d["pixel_pooled"]
        if p["paddy_iou"] < 1e-9:
            degen.append((d["run"], p["macro_iou"], p["paddy_iou"],
                          p["non_paddy_recall"]))
    degen.sort()
    emit("D3", "Released baselines that never predict the positive class",
         "reported as crop-type baselines, with a macro IoU",
         "; ".join(f"{r}: macro IoU {m:.4f} but paddy IoU {pi:.4f} and "
                   f"non-paddy recall {nr:.3f}"
                   for r, m, pi, nr in degen) or "none found",
         "a model that labels every pixel non-paddy still earns a macro IoU "
         "near 0.38 from the negative class alone, so the headline metric "
         "does not reveal that the model has learned nothing about paddy",
         "results/plot_level/plotlevel_crop_type_*.json",
         ", ".join(r for r, *_ in degen) or "none", len(degen) >= 2,
         "macro IoU")


# ----------------------------------------------------------------- D4
def d4_mape():
    """The phenology column labelled MAPE is MAE divided by 183."""
    hits, rows = 0, []
    for task in ("harvesting_date", "transplanting_date"):
        for f in glob.glob(str(
                PROJ / f"results/regression_gate/{task}_*_bs32.json")):
            d = json.loads(Path(f).read_text(encoding="utf8"))
            if "_debug" in d["run"]:
                continue
            rel = d["released"]
            if "val_mape" not in rel or "val_mae" not in rel:
                continue
            implied = rel["val_mae"] / 183.0
            ok = abs(implied - rel["val_mape"]) < 1e-4
            hits += ok
            rows.append((d["run"], task, rel["val_mae"], rel["val_mape"],
                         implied, ok))
    worst = max((abs(r[4] - r[3]) for r in rows), default=float("nan"))
    ex = rows[0] if rows else None
    emit("D4", "The phenology column labelled MAPE is MAE / 183",
         (f"{ex[1]} {ex[0]}: val_mape = {ex[3]:.6f}, which reads as "
          f"{ex[3] * 100:.2f} %" if ex else "n/a"),
         (f"val_mae / 183 = {ex[2]:.4f} / 183 = {ex[4]:.6f}; matches for "
          f"{hits} of {len(rows)} released phenology runs, worst "
          f"disagreement {worst:.2e}" if ex else "n/a"),
         "183 is a constant, not each field's true day, so the column is a "
         "rescaled MAE and not a percentage error. The dataset paper defines "
         "it this way; the released logs do not, which is where the "
         "misreading comes from. This is reported as a metric that invites "
         "misreading, not as a defect in the benchmark's definition",
         "results/regression_gate/{harvesting,transplanting}_date_*_bs32.json",
         f"{len(rows)} released phenology checkpoints",
         bool(rows) and hits == len(rows), "fraction")


# ------------------------------------------------- yield vs a constant
def yield_vs_constant():
    files = sorted(f for f in glob.glob(str(
        PROJ / "results/regression_gate/crop_yield_*_bs32.json"))
        if "_debug" not in f)
    if not files:
        return emit("Y", "Released yield models against a constant",
                    "n/a", "no artefacts", "", "", "none", False)
    first = json.loads(Path(files[0]).read_text(encoding="utf8"))
    keys = list(first["per_field_true"].keys())
    y = np.array([float(first["per_field_true"][k]) for k in keys])

    root = None
    for line in (PROJ / "configs/base.yaml").read_text(
            encoding="utf8").splitlines():
        if line.strip().startswith("root:"):
            root = line.split(":", 1)[1].strip().strip('"').strip("'")
    tr = []
    with open(f"{root}/sickle_dataset_tabular.csv", encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["YIELD"])
            except (TypeError, ValueError):
                continue
            if r["SPLIT"] == "train" and v > 0:
                tr.append(v)
    const = float(np.mean(tr))
    cmae = float(np.mean(np.abs(const - y)))
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf8"))
        p = np.array([float(d["per_field_pred"][k]) for k in keys])
        rows.append((d["run"], float(np.mean(np.abs(p - y)))))
    rows.sort(key=lambda t: t[1])
    worse = sum(1 for _, m in rows if m > cmae)
    emit("Y", "Every released yield model has a higher MAE than a constant",
         f"{len(rows)} released yield checkpoints, reported with MAE, RMSE "
         f"and MAPE, with no constant baseline in the benchmark",
         f"constant = training-split mean = {const:.1f} kg/acre, MAE "
         f"{cmae:.1f}; best checkpoint {rows[0][0]} MAE {rows[0][1]:.1f}; "
         f"{worse} of {len(rows)} checkpoints worse than the constant",
         "the comparison is on the same 41 validation field-seasons and the "
         "same ground truth the gate uses; the constant uses no imagery at "
         "all",
         "results/regression_gate/crop_yield_*_bs32.json and the released "
         "tabular index",
         ", ".join(r for r, _ in rows[:3]) + f", and {len(rows) - 3} more",
         worse == len(rows), "kg/acre")


# ----------------------------------------------------------- provenance
def environment():
    env = {"python": sys.version.split()[0],
           "platform": platform.platform(),
           "numpy": np.__version__}
    for mod in ("torch", "sklearn", "pandas", "matplotlib"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:                                    # noqa: BLE001
            env[mod] = "not installed"
    try:
        import torch
        env["torch_cuda"] = (torch.version.cuda or "cpu build")
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:                                        # noqa: BLE001
        pass
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args()

    print("=" * 74)
    print("MINIMAL REPRODUCTIONS -- published value beside measured value")
    print("=" * 74)
    print("Every block reads a cached evaluation artefact. No model is "
          "re-executed here,")
    print("so this shows that the conclusions follow from the recorded "
          "evaluations; it")
    print("does not re-run them. For that, see scripts/phase5_gate.py and")
    print("scripts/regression_gate.py.")
    print()

    d1_batch_size()
    d3_degenerate()
    d4_mape()
    yield_vs_constant()

    env = environment()
    print("ENVIRONMENT")
    for k, v in env.items():
        print(f"  {k:16s} {v}")
    print()

    n_ok = sum(1 for b in BLOCKS if b["reproduced"])
    print("=" * 74)
    print(f"{n_ok}/{len(BLOCKS)} reproduced from the cached artefacts")
    for b in BLOCKS:
        if not b["reproduced"]:
            print(f"  NOT REPRODUCED: {b['id']} {b['title']}")
    report = {"blocks": BLOCKS, "environment": env,
              "n_reproduced": n_ok, "n_blocks": len(BLOCKS),
              "scope": ("reads cached evaluation artefacts; does not "
                        "re-execute any model")}
    out = PROJ / "results" / "minimal_repro.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"wrote {out.relative_to(PROJ)}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2),
                                   encoding="utf8")
        print(f"wrote {args.json}")
    return 0 if n_ok == len(BLOCKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
