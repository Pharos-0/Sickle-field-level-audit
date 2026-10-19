"""One command that rebuilds every deliverable and re-checks all of them.

    python scripts/build_all.py
    python scripts/build_all.py --skip-pdf     # no Word on this machine

The manuscript, the reference table, the figures and the two reports must
describe the same build. Running the steps by hand in the wrong order is how
they drift: regenerating the paper without the figures, or editing a report
without re-running the checks that back it, produces documents that disagree
about their own contents.

This runs the whole chain in dependency order, stops at the first failure,
and prints a manifest with a SHA-256 prefix for every artefact so the set can
be identified later as having come from one build.

ORDER, AND WHY
--------------
  1 yield screen   the pre-registered negative result the paper reports
  2 field size     per-field pooling weights, read from the masks alone
  3 distributions  the per-field distributions and leave-fields-out result
  4 screen check   the crop-type screen against its label-free baselines
  5 power          what three seeds can and cannot resolve
 5b conditional    does the crop-type screen add anything beyond field size
 5c yield vs const plot-clustered intervals on each yield model's gap to a constant
  6 figures        read results/, must precede the paper that embeds them
 6b satellite fig Sentinel-2 chips and v5 predictions, from the gallery cache
  7 yield figure   the screening figure, same input as step 1
  8 manuscript     assigns citation numbers, writes references_final.json
 8b thesis paper   all three clauses of the objective, both projects
  9 reference xlsx reads references_final.json, so must follow the paper
 10 PDF            renders the .docx produced in step 8
 11 verify_all     claim checks against the artefacts
 12 independent    the same claims re-derived without importing src/
 13 drift_check    every headline number against its declared tolerance
 14 minimal_repro  published value beside measured value, per defect
 15 pytest         unit tests
    manifest       hashes of everything above

Steps 11-15 are checks, not builds: a failure there means the deliverables
are wrong, not that the build broke, and the exit code reflects that.

NOT IN THIS CHAIN, and deliberately so
--------------------------------------
scripts/plot_level_eval.py re-evaluates the twelve released crop-type
checkpoints on CPU and takes about ten minutes. It needs the released
weights, which are not part of this repository, so it cannot be a step in a
build that must run anywhere. Step 3 reports any checkpoint whose
leave-fields-out analysis is absent as NOT RUN rather than quietly
averaging over the ones it has.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
PY = sys.executable

ARTEFACTS = [
    "docs/SICKLE_Field_Level_Audit_IEEE.docx",
    "docs/SICKLE_Field_Level_Audit_IEEE.pdf",
    "docs/Thesis_Paper.docx",
    "docs/Thesis_Paper.pdf",
    "docs/Policy_Brief.docx",
    "docs/Policy_Brief.pdf",
    "docs/SICKLE_Guard_IEEE.docx",
    "docs/SICKLE_Guard_IEEE.pdf",
    "docs/SICKLE_Audit_IEEE_conf.docx",
    "docs/SICKLE_Audit_IEEE_conf.pdf",
    "docs/rse/SICKLE_Audit_RSE.docx",
    "docs/rse/SICKLE_Audit_RSE.pdf",
    "docs/rse/SICKLE_Audit_RSE_highlights.docx",
    "docs/references_rse.json",
    "docs/references_thesis.json",
    "docs/REFERENCE_VERIFICATION.xlsx",
    "docs/references.json",
    "docs/references_final.json",
    "docs/MANUSCRIPT_AUDIT.md",
    "docs/FINAL_VALIDATION_REPORT.md",
    "docs/REPRODUCE.md",
    "docs/MORNING_HANDOFF.md",
    "results/gate/diagnosis_misses.json",
    "configs/yield_screen_protocol.yaml",
    "results/triage/yield_screen.json",
    "results/triage/yield_per_field.csv",
    "results/triage/per_field_signals_v5.csv",
    "results/triage/screen_validate.json",
    "results/field_size_val.json",
    "results/field_distributions.json",
    "results/drift_check.json",
    "results/minimal_repro.json",
    "results/ladder_power.json",
    "results/triage/conditional_screen.json",
    "results/yield_vs_constant.json",
    "configs/conditional_screen_protocol.yaml",
    "configs/tolerances.yaml",
]


def run(label: str, args: list[str], required: bool = True) -> tuple[bool, str]:
    print(f"\n=== {label} " + "=" * max(0, 60 - len(label)))
    t0 = time.time()
    p = subprocess.run([PY, *args], cwd=PROJ, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.rstrip().split("\n")[-12:])
    print(tail)
    ok = p.returncode == 0
    print(f"--- {'ok' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    if not ok and required:
        print(f"\nBUILD STOPPED: {label} exited {p.returncode}.")
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pdf", action="store_true",
                    help="skip the Word export (needs MS Word via COM)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    steps = [
        # yield_screen must precede the figure and the manuscript: both read
        # results/triage/yield_screen.json, and the manuscript now refuses to
        # build without it.
        ("1/15 yield screening", ["scripts/yield_screen.py"], True),
        ("2/15 field size", ["scripts/field_size.py"], True),
        ("3/15 field distributions",
         ["scripts/field_distributions.py"], True),
        ("4/15 crop-type screen validation",
         ["scripts/screen_validate.py"], True),
        ("5/15 ablation power", ["scripts/ladder_power.py"], True),
        ("5b/15 conditional screen test",
         ["scripts/conditional_screen.py"], True),
        # plot-clustered intervals on yield checkpoints vs a constant; the
        # manuscript and thesis refuse to build without them
        ("5c/15 yield vs constant", ["scripts/yield_vs_constant.py"], True),
        ("6/15 figures", ["scripts/make_figures_ieee.py"], True),
        # Sentinel-2 chips with the v5 prediction (Fig. 3); reads the
        # gallery's cached CPU predictions, never the GPU.
        ("6b/15 satellite figure", ["scripts/make_fig_satellite.py"], True),
        ("7/15 yield figure",
         ["scripts/make_figure_yield_screen.py"], True),
        ("8/15 manuscript", ["scripts/make_paper_ieee.py"], True),
        # The thesis paper answers all three clauses of the objective, so it
        # reads artefacts from BOTH projects, and refuses to save if its own
        # text drops the canonical objective or claims too much.
        ("8b/15 thesis paper", ["scripts/make_thesis_paper.py"], True),
        # Two pages for officials and the panel, from the same loader as the
        # thesis; refuses to save if a finding lacks its evidence label.
        ("8c/15 policy brief", ["scripts/make_policy_brief.py"], True),
        # The SICKLE-Guard paper (IEEE): reads the thesis loader, so its
        # numbers cannot drift from the thesis; refuses to save on an
        # overstatement or a missing simulated/scope label.
        ("8d/15 SICKLE-Guard paper",
         ["scripts/make_guard_paper_ieee.py"], True),
        # The audit paper at conference length; same loader as the full
        # version, so the two cannot disagree on a number.
        ("8e/15 audit paper, conference version",
         ["scripts/make_paper_ieee_conf.py"], True),
        # The same audit as a Remote Sensing of Environment submission: the
        # journal build converted to the journal's requirements, with its own
        # checks (abstract length, highlights, name-date citations).
        ("8f/15 audit paper, RSE submission",
         ["scripts/make_paper_rse.py"], True),
        ("9/15 reference table",
         ["scripts/make_reference_table.py"], True),
    ]
    if not args.skip_pdf:
        steps.append(("10/15 PDF export",
                      ["scripts/export_pdf.py",
                       "docs/SICKLE_Field_Level_Audit_IEEE.docx"], True))
        steps.append(("10b/15 thesis PDF export",
                      ["scripts/export_pdf.py", "docs/Thesis_Paper.docx"],
                      True))
        steps.append(("10c/15 policy brief PDF export",
                      ["scripts/export_pdf.py", "docs/Policy_Brief.docx"],
                      True))
        steps.append(("10d/15 SICKLE-Guard paper PDF export",
                      ["scripts/export_pdf.py", "docs/SICKLE_Guard_IEEE.docx"],
                      True))
        steps.append(("10e/15 conference audit paper PDF export",
                      ["scripts/export_pdf.py",
                       "docs/SICKLE_Audit_IEEE_conf.docx"], True))
        steps.append(("10f/15 RSE submission PDF export",
                      ["scripts/export_pdf.py",
                       "docs/rse/SICKLE_Audit_RSE.docx"], True))

    for label, cmd, required in steps:
        ok, _ = run(label, cmd, required)
        if not ok and required:
            return 1

    # --- the build is reproducible only if a second run changes nothing
    before = (PROJ / "docs/references_final.json").read_bytes()
    run("rebuild (idempotence probe)", ["scripts/make_paper_ieee.py"], True)
    after = (PROJ / "docs/references_final.json").read_bytes()
    idem = before == after
    print(f"\nidempotent: {idem}")

    # --- checks
    failures = []
    ok, out = run("11/15 verify_all", ["scripts/verify_all.py"], False)
    if not ok:
        failures.append("verify_all")
    ok, out_ind = run("12/15 independent_check",
                      ["scripts/independent_check.py"], False)
    if not ok:
        failures.append("independent_check")
    skipped = out_ind.count("SKIP  ")

    # Drift is the check the others cannot make: verify_all asks whether a
    # claim still holds, this asks whether the number is still the same one,
    # against a tolerance declared in configs/tolerances.yaml with a reason.
    ok, _ = run("13/15 drift_check", ["scripts/drift_check.py"], False)
    if not ok:
        failures.append("drift_check")

    # Published value beside measured value, per defect, with the
    # checkpoint identifiers and library versions that produced each.
    ok, _ = run("14/15 minimal_repro", ["scripts/minimal_repro.py"], False)
    if not ok:
        failures.append("minimal_repro")

    if not args.skip_tests:
        ok, _ = run("15/15 pytest", ["-m", "pytest", "-q"], False)
        if not ok:
            failures.append("pytest")

    # --- manifest
    print("\n=== 15/15 manifest " + "=" * 42)
    manifest = {}
    missing = []
    for rel in ARTEFACTS:
        f = PROJ / rel
        if not f.exists():
            missing.append(rel)
            continue
        manifest[rel] = {
            "sha256_16": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
            "bytes": f.stat().st_size,
        }
    for f in sorted((PROJ / "results/figures_ieee").glob("*.png")):
        rel = str(f.relative_to(PROJ)).replace("\\", "/")
        manifest[rel] = {
            "sha256_16": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
            "bytes": f.stat().st_size,
        }
    out_path = PROJ / "docs/deliverable_hashes.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    for rel, v in manifest.items():
        print(f"  {v['sha256_16']}  {v['bytes']:>9,}  {rel}")
    if missing:
        print("\n  MISSING: " + ", ".join(missing))

    print("\n" + "=" * 62)
    if failures:
        print("CHECKS FAILED: " + ", ".join(failures))
        print("The deliverables were rebuilt but do not pass their own "
              "checks. Do not circulate this build.")
        return 1
    if not idem:
        print("NOT REPRODUCIBLE: a second run changed the output.")
        return 1
    if missing:
        print("BUILD INCOMPLETE: artefacts listed above are missing.")
        return 1
    print("All deliverables rebuilt from one run and all checks passed.")
    if skipped:
        print(f"NOTE: independent_check reported {skipped} SKIP(s). "
              f"A skip is not a pass -- see its output above.")
    print("Independent reproduction by another person is NOT part of this "
          "build and remains outstanding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
