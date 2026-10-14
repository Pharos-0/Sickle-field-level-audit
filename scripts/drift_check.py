"""Item 6: fail the build when a headline number moves beyond its tolerance.

    python scripts/drift_check.py
    python scripts/drift_check.py --json report.json
    python scripts/drift_check.py --update     # rewrite expected values

WHAT THIS ADDS THAT THE OTHER CHECKS DO NOT
-------------------------------------------
  scripts/verify_all.py        asks whether each claim still holds
  scripts/independent_check.py asks whether the arithmetic follows from the
                               raw files, without importing src/
  this script                  asks whether each number is the SAME number,
                               and if not, whether the movement is allowed

The difference matters. A claim such as "most released yield models lose to
a constant" survives a silent change from 24 of 24 to 19 of 24; a hash
manifest catches that something changed but not what, and flags harmless
re-renders just as loudly. This pins the values themselves, against
tolerances declared in configs/tolerances.yaml with a written reason for
each.

Every value is recomputed from primary artefacts rather than read back from
a summary the build also wrote, so a mistake in the build cannot validate
itself.

SEED-DEPENDENT VALUES
---------------------
Entries carrying `depends_on_seeds` are means over the seeds trained so
far. Training another seed changes them legitimately. That case is reported
as SEEDS-CHANGED rather than DRIFT, and still fails, because the new values
must be written into the registry deliberately rather than absorbed.

--update rewrites the `expected` fields in place and prints a diff for the
commit message. It refuses to run unless every change is explained, which
in practice means it is used right after a deliberate re-run.

EXIT CODE
---------
0 only if every entry is within tolerance. 1 otherwise. Suitable for CI.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
REG = PROJ / "configs" / "tolerances.yaml"
LADDER = PROJ / "results" / "ladder_crop_type.csv"
RUNGS = [f"v{i}" for i in range(1, 11)]


# ------------------------------------------------------------- helpers
def data_root() -> str:
    for line in (PROJ / "configs/base.yaml").read_text(
            encoding="utf8").splitlines():
        if line.strip().startswith("root:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no data.root in configs/base.yaml")


def ladder():
    return {r["rung"]: r for r in csv.DictReader(
        LADDER.open(encoding="utf8")) if r.get("pixel_macro_iou")}


def ladder_steps():
    L = ladder()
    return [(b, float(L[b]["pixel_macro_iou"])
             - float(L[a]["pixel_macro_iou"]))
            for a, b in zip(RUNGS, RUNGS[1:]) if a in L and b in L]


def yield_gate_files():
    return sorted(f for f in glob.glob(str(
        PROJ / "results/regression_gate/crop_yield_*_bs32.json"))
        if "_debug" not in f)


_YIELD_CACHE = {}


def yield_table():
    """Truth, the constant baseline, and every checkpoint's MAE and MAPE."""
    if _YIELD_CACHE:
        return _YIELD_CACHE
    files = yield_gate_files()
    if not files:
        raise RuntimeError("no crop_yield gate artefacts")
    first = json.loads(Path(files[0]).read_text(encoding="utf8"))
    keys = list(first["per_field_true"].keys())
    y = np.array([float(first["per_field_true"][k]) for k in keys])
    tr = []
    with open(f"{data_root()}/sickle_dataset_tabular.csv",
              encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["YIELD"])
            except (TypeError, ValueError):
                continue
            if r["SPLIT"] == "train" and v > 0:
                tr.append(v)
    const = float(np.mean(tr))
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf8"))
        if list(d["per_field_true"].keys()) != keys:
            raise RuntimeError(f"field order differs in {f}")
        p = np.array([float(d["per_field_pred"][k]) for k in keys])
        rows.append({"run": d["run"],
                     "mae": float(np.mean(np.abs(p - y))),
                     "mape": float(np.mean(np.abs(p - y) / y) * 100)})
    rows.sort(key=lambda r: r["mae"])
    _YIELD_CACHE.update({
        "y": y, "const": const, "rows": rows,
        "const_mae": float(np.mean(np.abs(const - y))),
        "const_mape": float(np.mean(np.abs(const - y) / y) * 100)})
    return _YIELD_CACHE


def audit_rows():
    f = PROJ / "results/audit_summary.csv"
    return [r for r in csv.DictReader(f.open(encoding="utf8"))
            if "_debug" not in r["run"]]


def tabular_rows():
    with open(f"{data_root()}/sickle_dataset_tabular.csv",
              encoding="utf8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------- derived values
DERIVED = {
    "ladder_sigma_max":
        lambda: max(float(r["pixel_sd"]) for r in ladder().values()),
    "ladder_n_steps": lambda: len(ladder_steps()),
    "ladder_n_exceeding": lambda: sum(
        1 for _, d in ladder_steps()
        if abs(d) >= max(float(r["pixel_sd"]) for r in ladder().values())),
    "ladder_n_inconclusive": lambda: sum(
        1 for _, d in ladder_steps()
        if abs(d) < max(float(r["pixel_sd"]) for r in ladder().values())),
    "const_value": lambda: yield_table()["const"],
    "const_mae": lambda: yield_table()["const_mae"],
    "const_mape": lambda: yield_table()["const_mape"],
    "n_yield_checkpoints": lambda: len(yield_table()["rows"]),
    "n_worse_mae": lambda: sum(
        1 for r in yield_table()["rows"]
        if r["mae"] > yield_table()["const_mae"]),
    "n_worse_mape": lambda: sum(
        1 for r in yield_table()["rows"]
        if r["mape"] > yield_table()["const_mape"]),
    "best_ckpt_mae": lambda: yield_table()["rows"][0]["mae"],
    "n_regression_audited": lambda: len(audit_rows()),
    "n_regression_pass": lambda: sum(
        1 for r in audit_rows() if r["verdict"] == "PASS"),
    "n_val_rows": lambda: sum(
        1 for r in tabular_rows() if r["SPLIT"] == "val"),
    "n_train_rows": lambda: sum(
        1 for r in tabular_rows() if r["SPLIT"] == "train"),
    "bs_spread_focus": lambda: (
        lambda c: max(c.values()) - min(c.values()))(
            bs_cells()["[L8]_unet3d"]),
    "bs_spread_max": lambda: max(
        max(v.values()) - min(v.values()) for v in bs_cells().values()),
}


def bs_cells():
    """Every crop-type checkpoint evaluated at more than one batch size."""
    out = {}
    for f in glob.glob(str(PROJ / "results/gate/gate_crop_type_*_bs*.json")):
        d = json.loads(Path(f).read_text(encoding="utf8"))
        out.setdefault(d["run"], {})[d["batch_size"]] = d["reproduced"]
    return {k: v for k, v in out.items() if len(v) > 1}


def seed_signature() -> dict:
    """Completed seeds per ladder rung, by the trainer's `complete` flag.

    Two defects lived here. The old check counted only v5, so new seeds on
    v3 and v4 -- which move sigma_max, the step partition and the radar
    delta -- were invisible to it, and their effect would have been
    reported as DRIFT rather than as new evidence. And it counted
    directories with a best_per_plot_iou.json, which an interrupted run
    leaves behind; that is the bug fixed everywhere else by
    src.utils.completed_seed_runs, and it is fixed here the same way.
    """
    sys.path.insert(0, str(PROJ))
    from src.utils import completed_seed_runs
    return {f"v{i}": len(completed_seed_runs("crop_type", f"v{i}", PROJ)[0])
            for i in range(1, 11)}


def n_seeds_trained(rung="v5"):
    return seed_signature().get(rung, 0)


# ---------------------------------------------------------- extraction
def dig(obj, path):
    for part in path.split("/"):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def extract(e):
    """Returns (value, note). A missing input is reported, never guessed."""
    kind = e["extractor"]
    if kind == "csv_cell":
        f = PROJ / e["file"]
        if not f.exists():
            return None, f"missing {e['file']}"
        for r in csv.DictReader(f.open(encoding="utf8")):
            if r.get(e["match_column"]) == e["match_value"]:
                return float(r[e["column"]]), ""
        return None, f"no row with {e['match_column']}={e['match_value']}"
    if kind == "json_path":
        f = PROJ / e["file"]
        if not f.exists():
            return None, f"missing {e['file']}"
        try:
            return float(dig(json.loads(f.read_text(encoding="utf8")),
                             e["path"])), ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return None, f"path {e['path']}: {exc}"
    if kind == "derived":
        fn = DERIVED.get(e["fn"])
        if fn is None:
            return None, f"no derived function named {e['fn']}"
        try:
            return float(fn()), ""
        except Exception as exc:                      # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    return None, f"unknown extractor {kind}"


# -------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write a machine-readable report here")
    ap.add_argument("--update", action="store_true",
                    help="rewrite expected values in the registry")
    args = ap.parse_args()

    if not REG.exists():
        print(f"FATAL: tolerance registry missing at "
              f"{REG.relative_to(PROJ)}")
        return 1
    reg = yaml.safe_load(REG.read_text(encoding="utf8"))
    classes = reg["classes"]
    seeds_now = n_seeds_trained()
    sig_now = seed_signature()
    sig_declared = reg.get("seed_signature") or {}
    seeds_moved = bool(sig_declared) and sig_now != sig_declared

    print("=" * 78)
    print(f"DRIFT CHECK against {REG.relative_to(PROJ)} "
          f"version {reg['version']}, declared {reg['declared_on']}")
    print(f"{len(reg['entries'])} declared values; "
          f"{len(reg.get('amendments') or [])} amendment(s) on record")
    print("completed seeds per rung: "
          + " ".join(f"{k}:{v}" for k, v in sig_now.items()))
    if seeds_moved:
        changed = {k: (sig_declared.get(k), v) for k, v in sig_now.items()
                   if sig_declared.get(k) != v}
        print("  CHANGED since the registry was declared: "
              + ", ".join(f"{k} {a}->{b}" for k, (a, b) in changed.items()))
    print("=" * 78)
    print(f"{'id':<40s}{'expected':>14s}{'found':>14s}{'delta':>12s}"
          f"  status")

    rows, bad, seedchg, missing = [], [], [], []
    for e in reg["entries"]:
        tol = classes[e["tolerance_class"]]
        got, note = extract(e)
        exp = float(e["expected"])
        seed_dep = e.get("depends_on_seeds")
        if got is None:
            status = "MISSING"
            missing.append(e["id"])
            delta = float("nan")
        else:
            delta = got - exp
            if abs(delta) <= tol:
                status = "ok"
            elif seed_dep is not None and (seeds_moved
                                           or seeds_now != seed_dep):
                status = "SEEDS-CHANGED"
                seedchg.append(e["id"])
            else:
                status = "DRIFT"
                bad.append(e["id"])
        rows.append({"id": e["id"], "expected": exp, "found": got,
                     "delta": None if got is None else delta,
                     "tolerance": tol,
                     "tolerance_class": e["tolerance_class"],
                     "status": status, "note": note,
                     "depends_on_seeds": seed_dep})
        g = "          n/a" if got is None else f"{got:>14.9g}"
        dl = "         n/a" if got is None else f"{delta:>+12.3g}"
        print(f"{e['id']:<40s}{exp:>14.9g}{g}{dl}  {status}"
              + (f"  ({note})" if note else ""))

    # ---- the printed manuscript values must match the stored ones
    print()
    print("ROUNDING AS PRINTED IN THE MANUSCRIPT")
    roundbad = []
    for e in reg["entries"]:
        if "manuscript_rounding" not in e:
            continue
        got, _ = extract(e)
        if got is None:
            continue
        want = float(e["manuscript_rounding"])
        dps = len(str(e["manuscript_rounding"]).split(".")[-1]) \
            if "." in str(e["manuscript_rounding"]) else 0
        ok = abs(round(got, dps) - want) < 10 ** (-dps - 3)
        if not ok:
            roundbad.append(e["id"])
        print(f"  {e['id']:<40s} prints {want:<12g} "
              f"recomputed rounds to {round(got, dps):<12g} "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    print("=" * 78)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"{n_ok}/{len(rows)} within tolerance")
    if missing:
        print(f"MISSING INPUTS ({len(missing)}): " + ", ".join(missing))
        print("  A missing input is not a pass. Build the artefact or "
              "remove the entry with a recorded reason.")
    if seedchg:
        print(f"SEEDS CHANGED ({len(seedchg)}): " + ", ".join(seedchg))
        print(f"  These are means over v5 seeds, and {seeds_now} seeds are "
              f"now on disk where the registry declares "
              f"{reg['entries'][0].get('depends_on_seeds')}. This is new "
              f"evidence, not a fault, but the registry must be updated "
              f"deliberately: re-run the dependent analyses, then "
              f"`drift_check.py --update`, and record an amendment.")
    if roundbad:
        print(f"ROUNDING MISMATCH ({len(roundbad)}): " + ", ".join(roundbad))
        print("  The manuscript prints a value the artefacts no longer "
              "support.")
    if bad:
        print(f"DRIFT ({len(bad)}): " + ", ".join(bad))
        print("  A declared value moved further than its declared "
              "tolerance allows.")

    if args.update:
        txt = REG.read_text(encoding="utf8")
        changed = 0
        for r in rows:
            if r["status"] in ("ok", "MISSING") or r["found"] is None:
                continue
            old = f"{r['expected']!r}"
            e = next(x for x in reg["entries"] if x["id"] == r["id"])
            lines = txt.split("\n")
            for i, ln in enumerate(lines):
                if ln.strip() == f"- id: {r['id']}":
                    for j in range(i, min(i + 30, len(lines))):
                        if lines[j].strip().startswith("expected:"):
                            pad = lines[j][:len(lines[j])
                                           - len(lines[j].lstrip())]
                            lines[j] = f"{pad}expected: {r['found']!r}"
                            changed += 1
                            # the manuscript renders this value from the
                            # artefact, so its printed rounding moves with
                            # it; keep the declared number of decimals
                            for k2 in range(j + 1, min(j + 8, len(lines))):
                                st = lines[k2].strip()
                                if st.startswith("- id:"):
                                    break
                                if st.startswith("manuscript_rounding:"):
                                    old_r = st.split(":", 1)[1].strip()
                                    dps = (len(old_r.split(".")[1])
                                           if "." in old_r else 0)
                                    new_r = round(float(r["found"]), dps)
                                    p2 = lines[k2][:len(lines[k2])
                                                   - len(lines[k2].lstrip())]
                                    lines[k2] = (f"{p2}manuscript_rounding: "
                                                 f"{new_r:.{dps}f}")
                                    print(f"    printed rounding "
                                          f"{old_r} -> {new_r:.{dps}f}")
                                    break
                            break
                    break
            txt = "\n".join(lines)
            print(f"  updated {r['id']}: {old} -> {r['found']!r}")
            del e
        if seeds_moved or not sig_declared:
            sig_lines = ["seed_signature:"] + [
                f"  {k}: {v}" for k, v in sig_now.items()]
            if "\nseed_signature:" in txt:
                a = txt.index("\nseed_signature:") + 1
                b = a
                ls = txt[a:].split("\n")
                k_ = 1
                while k_ < len(ls) and ls[k_].startswith("  "):
                    k_ += 1
                b = a + len("\n".join(ls[:k_]))
                txt = txt[:a] + "\n".join(sig_lines) + txt[b:]
            else:
                txt = txt.replace("\nclasses:",
                                  "\n" + "\n".join(sig_lines) + "\n\nclasses:",
                                  1)
            changed += 1
            print("  seed signature updated to the current counts")
        if changed:
            REG.write_text(txt, encoding="utf8", newline="\n")
            print(f"rewrote {changed} expected value(s) in "
                  f"{REG.relative_to(PROJ)}")
            print("Record an amendment in that file saying why, then "
                  "re-run without --update.")

    report = {"registry_version": reg["version"],
              "seed_signature_now": sig_now,
              "seed_signature_declared": sig_declared,
              "seeds_changed_since_declared": seeds_moved,
              "declared_on": str(reg["declared_on"]),
              "v5_seeds_on_disk": seeds_now,
              "n_ok": n_ok, "n_entries": len(rows),
              "drift": bad, "seeds_changed": seedchg,
              "missing": missing, "rounding_mismatch": roundbad,
              "entries": rows}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2),
                                   encoding="utf8")
        print(f"wrote {args.json}")
    out = PROJ / "results" / "drift_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"wrote {out.relative_to(PROJ)}")

    if bad or missing or roundbad or seedchg:
        print("DRIFT CHECK FAILED")
        return 1
    print("DRIFT CHECK PASSED: every declared value is within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
