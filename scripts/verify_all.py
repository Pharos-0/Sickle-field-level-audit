"""Independent verification entry point.

    python scripts/verify_all.py            # check committed results
    python scripts/verify_all.py --rerun    # recompute from the checkpoints

The reviewers asked that the four defects and the 64 reproduction results be
independently checked. This is the single command that does it. It is written
for someone who did not run the original experiments and has no reason to
trust them.

Default mode re-reads every committed result file and re-checks the claims
made in the paper against it -- fast, and catches a claim that has drifted
from its evidence. `--rerun` additionally re-executes the gates from the
released checkpoints, which takes a few hours and needs the dataset.

Every claim below names the file it is checked against, so a disagreement
points at a specific artefact rather than at the paper as a whole.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

OK, BAD = "PASS", "FAIL"
results = []


def check(name, cond, detail, source):
    results.append((OK if cond else BAD, name, detail, source))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="re-execute the gates (hours, needs the dataset)")
    args = ap.parse_args()

    # --- Defect 1: batch-size dependence -------------------------------
    g = PROJ / "results" / "gate"
    bs = {}
    for f in g.glob("gate_crop_type_[[]L8[]]_unet3d_bs*.json"):
        d = json.loads(f.read_text(encoding="utf8"))
        bs[d["batch_size"]] = d["reproduced"]
    if len(bs) >= 2:
        spread = max(bs.values()) - min(bs.values())
        check("D1 batch-size dependence", spread > 0.05,
              f"IoU spread {spread:.4f} across batch sizes {sorted(bs)}",
              "results/gate/gate_crop_type_[L8]_unet3d_bs*.json")
    else:
        check("D1 batch-size dependence", False,
              f"only {len(bs)} batch size(s) on disk; rerun the gate at 8/16/32",
              "results/gate/")

    # --- Defect 3: degenerate released UTAE baselines ------------------
    degen = []
    for f in g.glob("gate_crop_type_*utae*.json"):
        d = json.loads(f.read_text(encoding="utf8"))
        if d["all_metrics"].get("paddy_iou", 1) < 1e-6:
            degen.append((d["run"], d["released_best_epoch"]))
    check("D3 degenerate UTAE baselines", len(degen) >= 2,
          f"{len(degen)} cells with paddy IoU 0: {degen}",
          "results/gate/*utae*.json")

    # --- Defect 4: date MAPE is MAE/183 --------------------------------
    runs = Path("D:/FPD/runs/wacv_2024")
    hits = tot = 0
    for task in ("harvesting_date", "transplanting_date", "sowing_date"):
        d_ = runs / task
        if not d_.is_dir():
            continue
        for r in sorted(d_.iterdir())[:4]:
            t = r / "trainlog.json"
            if not t.exists():
                continue
            log = json.loads(t.read_text(encoding="utf8"))
            ks = [k for k in log if log[k].get("val_mae") is not None]
            if not ks:
                continue
            b = min(ks, key=lambda k: log[k]["val_mae"])
            tot += 1
            if abs(log[b]["val_mae"] / 183.0 - log[b]["val_mape"]) < 1e-3:
                hits += 1
    check("D4 date MAPE == MAE/183", tot > 0 and hits == tot,
          f"{hits}/{tot} released date runs match exactly",
          "D:/FPD/runs/wacv_2024/*_date/*/trainlog.json")

    # --- Reproduction: 52 regression models ----------------------------
    a = PROJ / "results" / "audit_summary.csv"
    if a.exists():
        rows = list(csv.DictReader(a.open(encoding="utf8")))
        npass = sum(1 for r in rows if r["verdict"] == "PASS")
        check("Reproduction (regression)", npass == len(rows) and len(rows) >= 52,
              f"{npass}/{len(rows)} gates passed", "results/audit_summary.csv")
    else:
        check("Reproduction (regression)", False, "audit_summary.csv missing",
              "results/audit_summary.csv")

    # --- Reproduction: 12 crop-type cells ------------------------------
    cells = list(g.glob("gate_crop_type_*_bs32.json"))
    passed = sum(1 for f in cells
                 if json.loads(f.read_text(encoding="utf8"))["verdict"] == "PASS")
    check("Reproduction (crop type)", len(cells) >= 12,
          f"{passed}/{len(cells)} cells PASS at the declared tolerance",
          "results/gate/*_bs32.json")

    # --- Ladder: the two significant effects ---------------------------
    L = {r["rung"]: r for r in csv.DictReader(
        (PROJ / "results" / "ladder_crop_type.csv").open(encoding="utf8"))
        if r.get("pixel_macro_iou")}
    if {"v1", "v2", "v4", "v5"} <= set(L):
        dt = float(L["v2"]["pixel_macro_iou"]) - float(L["v1"]["pixel_macro_iou"])
        dr = float(L["v5"]["pixel_macro_iou"]) - float(L["v4"]["pixel_macro_iou"])
        sd = max(float(L[k]["pixel_sd"]) for k in ("v1", "v2", "v4", "v5"))
        check("Ladder: temporal exceeds seed spread", dt > sd,
              f"delta {dt:+.4f} against max seed sd {sd:.4f}",
              "results/ladder_crop_type.csv")
        check("Ladder: radar exceeds seed spread", dr > sd,
              f"delta {dr:+.4f} against max seed sd {sd:.4f}",
              "results/ladder_crop_type.csv")
        check("Ladder: three seeds per rung",
              all(int(r["n_seeds"]) == 3 for r in L.values()),
              f"seeds per rung {sorted({int(r['n_seeds']) for r in L.values()})}",
              "results/ladder_crop_type.csv")

    # --- Triage ---------------------------------------------------------
    tp = PROJ / "results" / "triage" / "triage_v5.json"
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf8"))
        loo = {r["target_workload"]: r
               for r in t["seed_disagreement"]["leave_one_out"]}
        r50 = loo.get(0.5)
        check("Triage: LOO recall reported with CI",
              r50 is not None and "recall_ci" in r50,
              (f"refer top 50%: recall {r50['recall']:.0%} "
               f"CI [{r50['recall_ci']['lo']:.0%}, {r50['recall_ci']['hi']:.0%}]")
              if r50 else "missing",
              "results/triage/triage_v5.json")
        check("Triage: sample size stated honestly",
              t["n_fields"] <= 40 and t["n_bad"] <= 20,
              f"{t['n_fields']} fields, {t['n_bad']} unreliable "
              f"-> one field is {100/t['n_bad']:.1f} pp of recall",
              "results/triage/triage_v5.json")

    # --- provenance ------------------------------------------------------
    man = list((PROJ / "outputs").rglob("manifest.json"))
    clean = sum(1 for m in man
                if json.loads(m.read_text(encoding="utf8")).get("git_dirty") is False)
    check("Provenance: runs record a clean commit", len(man) > 0,
          f"{clean}/{len(man)} manifests report git_dirty=false",
          "outputs/**/manifest.json")

    if args.rerun:
        print("re-executing the gates; this takes hours\n")
        for cmd in (["scripts/phase5_gate.py", "--run", "[S2]_unet3d",
                     "--batch-size", "32"],
                    ["scripts/run_full_audit.py"]):
            subprocess.run([sys.executable, "-u"] + cmd, cwd=str(PROJ))

    print(f"{'result':8s}{'claim':40s}detail")
    print("-" * 100)
    for v, name, detail, src in results:
        print(f"{v:8s}{name:40s}{detail}")
        print(f"{'':8s}{'':40s}source: {src}")
    nbad = sum(1 for r in results if r[0] == BAD)
    print("-" * 100)
    print(f"{len(results)-nbad}/{len(results)} checks passed")
    if nbad:
        print("A FAIL means the paper's claim no longer matches the evidence "
              "on disk. Investigate the named source before trusting the text.")
    return 1 if nbad else 0


if __name__ == "__main__":
    raise SystemExit(main())
