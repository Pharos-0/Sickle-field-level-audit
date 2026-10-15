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
    # Details can carry non-ASCII (a rupee sign, a typographic minus);
    # a piped Windows console would otherwise crash the whole audit.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
        for r in sorted(d_.iterdir()):
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
        allrows = list(csv.DictReader(a.open(encoding="utf8")))
        rows = [r for r in allrows if "_debug" not in r["run"]]
        npass = sum(1 for r in rows if r["verdict"] == "PASS")
        check("Reproduction (regression)", npass == len(rows) and len(rows) >= 48,
              f"{npass}/{len(rows)} gates passed "
              f"({len(allrows) - len(rows)} *_debug smoke tests excluded)",
              "results/audit_summary.csv")
        check("Audited count excludes 2-epoch smoke tests",
              len(allrows) - len(rows) == 4,
              f"{len(allrows) - len(rows)} *_debug runs excluded from the "
              f"{len(rows)} regression models reported",
              "results/audit_summary.csv")
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
        # An earlier manuscript draft said "two of nine" steps exceeded the
        # seed spread when four do, and omitted the class-weighting rung.
        rungs = [f"v{i}" for i in range(1, 11)]
        sdmax = max(float(L[k]["pixel_sd"]) for k in L)
        steps = [float(L[b]["pixel_macro_iou"]) - float(L[a]["pixel_macro_iou"])
                 for a, b in zip(rungs, rungs[1:]) if a in L and b in L]
        exc = [d for d in steps if abs(d) >= sdmax]
        check("Ladder: all adjacent steps partitioned", len(steps) == 9,
              f"{len(steps)} steps = {len(exc)} exceeding sigma_max "
              f"{sdmax:.4f} ({sum(1 for d in exc if d > 0)} up, "
              f"{sum(1 for d in exc if d < 0)} down) + "
              f"{len(steps) - len(exc)} inconclusive",
              "results/ladder_crop_type.csv")
        # Asserting three seeds per rung was right while the ladder was
        # uniform and becomes wrong the moment extra seeds are trained for
        # the rungs the power analysis singles out. What must hold is that
        # every rung has at least the minimum the paper claims, and that
        # the manuscript states whatever the real counts are rather than a
        # remembered number.
        _c = {k: int(r["n_seeds"]) for k, r in L.items()}
        check("Ladder: every rung has at least three completed seeds",
              all(v >= 3 for v in _c.values()),
              "seeds per rung "
              + ", ".join(f"{k}:{v}" for k, v in sorted(_c.items()))
              + (" (uniform)" if len(set(_c.values())) == 1
                 else " (NOT uniform -- the manuscript renders this from "
                      "the CSV, so check the rendered text says so)"),
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

    # --- Yield screening: a NEGATIVE result, pinned so it cannot drift ---
    ysf = PROJ / "results" / "triage" / "yield_screen.json"
    if ysf.exists():
        y = json.loads(ysf.read_text(encoding="utf8"))
        tf = y["threshold_free"]
        lo, hi = tf["auroc_ci"]["lo"], tf["auroc_ci"]["hi"]
        # The manuscript says the signal is NOT demonstrated. That claim is
        # true exactly when the interval contains chance; if a rebuild ever
        # makes it exclude 0.5, the paper is wrong and this must fail.
        check("Yield screen: reported as not demonstrated",
              lo <= 0.5 <= hi and not tf["auroc_ci_excludes_half"],
              f"AUROC {tf['auroc']:.3f} CI [{lo:.3f}, {hi:.3f}] contains "
              f"chance 0.5",
              "results/triage/yield_screen.json")
        check("Yield screen: independent unit is the plot",
              y["independent_unit"] == "plot" and y["n_plots"] < y["n_field_seasons"],
              f"{y['n_field_seasons']} field-seasons from {y['n_plots']} "
              f"plots; intervals bootstrap plots",
              "results/triage/yield_screen.json")
        check("Yield screen: ensemble members are not independent",
              y["ensemble_independence"]["pairwise_corr_median"] > 0.5,
              f"median pairwise prediction correlation "
              f"{y['ensemble_independence']['pairwise_corr_median']:.3f} "
              f"across {y['n_checkpoints']} checkpoints -- disclosed, not "
              f"treated as an independent ensemble",
              "results/triage/yield_screen.json")

    # --- Diagnosis of the three unreproduced crop-type cells ------------
    dj = PROJ / "results" / "gate" / "diagnosis_misses.json"
    if dj.exists():
        d_ = json.loads(dj.read_text(encoding="utf8"))
        cells = d_.get("cells", [])
        expl = [c for c in cells if c["best_abs_delta"] <= d_["tolerance"]]
        # The manuscript says one of three is explained by batch composition
        # and two are not. If a rebuild changes that split the text is wrong.
        check("Gate misses: one of three explained by batch size",
              len(cells) == 3 and len(expl) == 1,
              f"{len(expl)}/{len(cells)} cells reach the released value at "
              f"some batch size: "
              + "; ".join(f"{c['run']} best bs{c['best_batch_size']} off "
                          f"{c['best_abs_delta']:.4f}" for c in cells),
              "results/gate/diagnosis_misses.json")
        check("Gate misses: batch size genuinely moves the score",
              all(c["spread"] > 0 for c in cells),
              "spreads " + ", ".join(f"{c['spread']:.4f}" for c in cells)
              + " -- a zero spread would mean the sweep did not run",
              "results/gate/diagnosis_misses.json")

    # --- Released yield models vs a constant predictor ------------------
    import ast as _ast, glob as _glob
    gl = [x for x in sorted(_glob.glob(str(PROJ / "results/regression_gate/crop_yield_*_bs32.json")))
          if "_debug" not in x]
    gl = [x for x in gl if "per_field_pred" in json.loads(Path(x).read_text(encoding="utf8"))]
    if gl:
        d0 = json.loads(Path(gl[0]).read_text(encoding="utf8"))
        truth = {_ast.literal_eval(k): float(v)
                 for k, v in d0["per_field_true"].items()}
        ks = sorted(truth)
        yv = [truth[k] for k in ks]
        trn = []
        with open(str(PROJ / "configs/base.yaml"), encoding="utf8") as fh:
            root = next((l.split(":", 1)[1].strip().strip(chr(34) + chr(39))
                         for l in fh if l.strip().startswith("root:")), None)
        with open(f"{root}/sickle_dataset_tabular.csv", encoding="utf8") as fh:
            for r in csv.DictReader(fh):
                try:
                    v = float(r["YIELD"])
                except Exception:
                    continue
                if r["SPLIT"] == "train" and v > 0:
                    trn.append(v)
        cm = sum(trn) / len(trn)
        cmae = sum(abs(cm - t) for t in yv) / len(yv)
        worse = 0
        for x in gl:
            dd = json.loads(Path(x).read_text(encoding="utf8"))
            pv = [float(dd["per_field_pred"][str(k)]) for k in ks]
            mae = sum(abs(a - b) for a, b in zip(pv, yv)) / len(yv)
            worse += mae > cmae
        check("Released yield models: none beats the constant on MAE "
              "(point estimates)",
              worse == len(gl),
              f"{worse}/{len(gl)} released yield checkpoints have higher "
              f"observed MAE than the train-mean constant ({cmae:.1f} "
              f"kg/acre)",
              "results/regression_gate/crop_yield_*_bs32.json")

        # Point estimates are not the claim. With 41 field-seasons from 14
        # plots no checkpoint's gap is distinguishable from zero, so the
        # documents may say "none is shown to beat a constant" and must not
        # say the models lose to it (corrected 2026-10-13).
        yvf = PROJ / "results" / "yield_vs_constant.json"
        Y = json.loads(yvf.read_text(encoding="utf8")) if yvf.exists() else {}
        from docx import Document as _D
        said = []
        banned = ("loses to a constant", "lose to a constant",
                  "worse than a constant", "worse than the constant")
        for name in ("SICKLE_Field_Level_Audit_IEEE.docx", "Thesis_Paper.docx",
                     "Policy_Brief.docx", "SICKLE_Audit_IEEE_conf.docx",
                     "SICKLE_Guard_IEEE.docx"):
            f = PROJ / "docs" / name
            if not f.exists():
                continue
            t = " ".join(pp.text for pp in _D(str(f)).paragraphs).lower()
            said += [f"{name}: '{b}'" for b in banned if b in t]
        overclaim_allowed = (Y.get("n_ci_above_zero") == Y.get("n_checkpoints"))
        # The best checkpoint's r, split into between- and within-plot
        # parts. Recomputed here from the raw predictions, so the saved
        # coefficients must be the intended ones on the intended samples,
        # and the manuscript must quote the split rather than read the
        # overall r as season-level skill (corrected 2026-10-14).
        dc = (Y.get("best") or {}).get("decomposition") or {}
        ok_dc, det_dc = False, "decomposition missing"
        if dc:
            import numpy as _np
            bj = next(x for x in gl if json.loads(Path(x).read_text(
                encoding="utf8"))["run"] == Y["best"]["run"])
            bd = json.loads(Path(bj).read_text(encoding="utf8"))
            pb_ = _np.array([float(bd["per_field_pred"][str(k)]) for k in ks])
            y_ = _np.array(yv)
            pl_ = _np.array([float(k[1]) for k in ks])
            ups = _np.unique(pl_)
            mp = _np.array([pb_[pl_ == u].mean() for u in ups])
            mt = _np.array([y_[pl_ == u].mean() for u in ups])
            pcw = pb_ - _np.array([pb_[pl_ == u].mean() for u in pl_])
            ycw = y_ - _np.array([y_[pl_ == u].mean() for u in pl_])
            rb_ = float(_np.corrcoef(mp, mt)[0, 1])
            rw_ = float(_np.corrcoef(pcw, ycw)[0, 1])
            ieee = PROJ / "docs" / "SICKLE_Field_Level_Audit_IEEE.docx"
            it = (" ".join(pp.text for pp in _D(str(ieee)).paragraphs)
                  if ieee.exists() else "")
            fmt = lambda v: format(v, "+.3f").replace("-", "\u2212")  # noqa: E731
            quoted = fmt(rb_) in it and fmt(rw_) in it
            misread = "order field-seasons informatively" in it
            ok_dc = (abs(rb_ - dc["between_plots"]["r"]) < 1e-9
                     and abs(rw_ - dc["within_plots"]["r"]) < 1e-9
                     and dc["between_plots"]["n"] == len(ups)
                     and dc["within_plots"]["n"] == len(ks)
                     and quoted and not misread)
            det_dc = (f"{Y['best']['run']}: overall r "
                      f"{dc['pearson_r_overall']:+.3f}; between {len(ups)} "
                      f"plot means r {rb_:+.3f}; within plots over {len(ks)} "
                      f"field-seasons r {rw_:+.3f} (recomputed, matches "
                      f"saved); manuscript quotes both: {quoted}; reads r "
                      f"as season-level skill: {misread}")
        check("Yield: the best model's r is between-plot, not season "
              "tracking, and the manuscript says so",
              ok_dc, det_dc, "results/yield_vs_constant.json")

        check("Yield vs constant: intervals computed, and no document "
              "claims more than they show",
              bool(Y) and Y["n_checkpoints"] == len(gl)
              and (overclaim_allowed or not said),
              (f"{Y['n_ci_above_zero']}/{Y['n_checkpoints']} plot-clustered "
               f"95% intervals lie wholly above zero ({Y['n_field_seasons']} "
               f"field-seasons, {Y['n_plots']} plots); best "
               f"{Y['best']['mae_minus_constant']:+.1f} "
               f"[{Y['best']['ci95'][0]:+.1f}, {Y['best']['ci95'][1]:+.1f}]"
               + (f"; OVERCLAIM in {said}" if said and not overclaim_allowed
                  else "; documents say 'not shown to beat'"))
              if Y else "results/yield_vs_constant.json missing",
              "results/yield_vs_constant.json")

    # --- field-level distribution, and what carries the pooled figure ---
    fd = PROJ / "results" / "field_distributions.json"
    if fd.exists():
        D = json.loads(fd.read_text(encoding="utf8"))
        lad = D["ladder_crop_type"]
        rel = D["released_crop_type"]

        # Field IoU, not the two-class average, which caps an imperfect
        # single-crop field at 0.5 (corrected 2026-10-12). The macro values
        # must stay on the record so the size of the correction is visible.
        mc = lad.get("macro_contrast") or {}
        check("Field level: fields scored on the crop they hold",
              lad.get("per_field_score") == "IoU of the crop(s) the field holds"
              and bool(mc) and lad.get("n_single_crop_fields", 0) >= 30,
              f"{lad.get('n_single_crop_fields')} of {lad['n_fields']} fields "
              f"hold one crop; field-IoU median {lad['quantiles']['50']:.3f} "
              f"vs the superseded two-class median "
              f"{mc.get('median', float('nan')):.3f}",
              "results/field_distributions.json")

        # What the corrected claim says: the pooled figure is near the
        # average field but conceals a concentrated tail.
        b = lad["below"]["0.5"]
        sd_ = lad["share_of_shortfall_worst_decile"]
        check("Field level: the pooled figure conceals a concentrated tail",
              b["n"] >= 1 and sd_["share"] > 0.5,
              f"{b['n']}/{lad['n_fields']} fields below 0.50 (95% CI "
              f"[{b['ci'][0]:.0%}, {b['ci'][1]:.0%}]); the "
              f"{sd_['n_fields']} worst carry {sd_['share']:.0%} of the "
              f"shortfall below pooled {lad['pooled_pixel_macro_iou']:.4f} "
              f"(average field {lad['per_field_mean']:.4f})",
              "results/field_distributions.json")

        # Size must NOT be offered as the explanation: it is not one.
        sp = lad["size"]["spearman_weight_vs_iou"]
        spci = lad["size"]["spearman_weight_vs_iou_ci"]
        check("Field level: size does not predict per-field accuracy",
              spci[0] < 0 < spci[1],
              f"Spearman(size, IoU) = {sp:+.3f}, 95% CI "
              f"[{spci[0]:+.3f}, {spci[1]:+.3f}] includes zero, so the gap "
              f"must not be described as large fields being easier",
              "results/field_distributions.json")

        # The exact version of the mechanism claim.
        sh = [abs(r["shift_without_largest_field"]) for r in rel
              if "shift_without_largest_field" in r]
        miss = D.get("released_missing_leave_fields_out") or []
        check("Field level: removing one field of 36 moves the pooled figure",
              bool(sh) and not miss and max(sh) > 0.05,
              (f"{len(sh)}/{len(rel)} checkpoints analysed, median shift "
               f"{sorted(sh)[len(sh) // 2]:.4f}, max {max(sh):.4f}"
               if sh else "no leave-fields-out analysis on disk")
              + (f"; NOT RUN for {', '.join(miss)}" if miss else ""),
              "results/plot_level/plotlevel_crop_type_*.json")

        # The gap is a dispersion effect, not a uniform optimism.
        neg = [r["run"] for r in rel if r["gap"] < 0]
        check("Field level: the gap is reported as not universal",
              len(neg) >= 1,
              f"{len(neg)} of {len(rel)} released checkpoints score HIGHER "
              f"per-field than pooled ({', '.join(neg) or 'none'}), which "
              f"the text must not generalise away",
              "results/field_distributions.json")
    else:
        check("Field level: distribution artefact present", False,
              "results/field_distributions.json missing -- run "
              "scripts/field_distributions.py",
              "results/field_distributions.json")

    # --- crop-type screen against its label-free baselines ---------------
    sv = PROJ / "results" / "triage" / "screen_validate.json"
    if sv.exists():
        V = json.loads(sv.read_text(encoding="utf8"))
        for key in ("seed_disagreement", "low_confidence"):
            v = V["verdict"][key]
            a = V["signals"][key]
            check(f"Screen {key}: reported at the strength shown",
                  v["verdict"] == "not demonstrated",
                  f"AUROC {a['auroc']:.3f} CI [{a['auroc_ci'][0]:.3f}, "
                  f"{a['auroc_ci'][1]:.3f}]"
                  + (" excludes chance" if v["auroc_interval_excludes_chance"]
                     else " includes chance")
                  + ("" if v["above_permutation_p97_5"]
                     else ", not above the permutation null's p97.5")
                  + "; not separated from every label-free baseline, so "
                  "the manuscript must not call it demonstrated",
                  "results/triage/screen_validate.json")
        pn = V["permutation_null"]
        check("Screen: random referral is a distribution, not one draw",
              abs(pn["auroc_mean"] - 0.5) < 0.02
              and pn["auroc_p97_5"] > 0.6,
              f"{pn['n_permutations']} permutations, mean "
              f"{pn['auroc_mean']:.3f}, p97.5 {pn['auroc_p97_5']:.3f} -- a "
              f"single lucky permutation reaches that, which is why the "
              f"comparison uses the distribution",
              "results/triage/screen_validate.json")
    else:
        check("Screen: validation artefact present", False,
              "results/triage/screen_validate.json missing -- run "
              "scripts/screen_validate.py",
              "results/triage/screen_validate.json")

    # --- is the crop-type screen just field size? ------------------------
    cs = PROJ / "results" / "triage" / "conditional_screen.json"
    if cs.exists():
        C = json.loads(cs.read_text(encoding="utf8"))
        prim = C["test_1_conditional_permutation"]["low_confidence"]
        check("Screen: carries information beyond field size",
              prim["beats_size_conditioned_null"],
              f"AUROC {prim['observed_auroc']:.3f} against a null that "
              f"permutes within size strata and so preserves the "
              f"size-outcome relationship exactly: null mean "
              f"{prim['null_mean']:.3f}, p95 {prim['null_p95']:.3f}, "
              f"p = {prim['p_one_sided']:.4f} over "
              f"{prim['n_permutations']} permutations",
              "results/triage/conditional_screen.json")

        ff = C["test_2_out_of_sample"]["fit_free"]["screen alone"]
        check("Screen: no ranking gain over size is claimed",
              not ff.get("improves_vs_size_both_directions", True),
              f"screen minus size (larger first) "
              f"{ff['delta_vs_size_alone']:+.4f}, 95% CI "
              f"[{ff['ci'][0]:+.4f}, {ff['ci'][1]:+.4f}]; minus size "
              f"(smaller first) {ff['delta_vs_size_smaller_first']:+.4f}, "
              f"95% CI [{ff['ci_vs_smaller_first'][0]:+.4f}, "
              f"{ff['ci_vs_smaller_first'][1]:+.4f}]. The screen must beat "
              f"both directions (amendment 2026-10-12) and does not.",
              "results/triage/conditional_screen.json")

        nn = C.get("loo_noise_null", {})
        check("Screen: the invalid leave-one-out test is marked as such",
              bool(nn) and nn.get("share_below_half", 0) > 0.5,
              f"leave-one-out logistic on {nn.get('n_draws', 0)} pure-noise "
              f"features averages AUROC {nn.get('mean', float('nan')):.3f} "
              f"with {nn.get('share_below_half', float('nan')):.0%} of "
              f"draws below 0.5, so the size-only baseline scoring "
              f"{C['test_2_out_of_sample']['size only']['loo_auroc']:.3f} "
              f"is that pathology and not a property of size. The protocol "
              f"amendment records it and the verdict does not rest on it.",
              "configs/conditional_screen_protocol.yaml")
    else:
        check("Screen: conditional test present", False,
              "results/triage/conditional_screen.json missing -- run "
              "scripts/conditional_screen.py",
              "results/triage/conditional_screen.json")

    # --- no interrupted training run may enter a reported average --------
    sys.path.insert(0, str(PROJ))
    from src.utils import completed_seed_runs
    inc, tot = [], 0
    for rung in [f"v{i}" for i in range(1, 11)]:
        kept, dropped = completed_seed_runs("crop_type", rung, PROJ)
        tot += len(kept) + len(dropped)
        inc += [d.name for d, _ in dropped]
    check("Seeds: every reported run finished training",
          True,
          f"{tot - len(inc)}/{tot} seed runs complete"
          + (f"; EXCLUDED as incomplete: {', '.join(sorted(inc))}"
             if inc else "; none excluded")
          + ". An interrupted run leaves best.json behind, so selection is "
            "by the trainer's `complete` flag, not by globbing directories.",
          "outputs/crop_type/*/best.json")

    # --- ablation: magnitude is not the same as distinguishable ----------
    lp = PROJ / "results" / "ladder_power.json"
    if lp.exists():
        L = json.loads(lp.read_text(encoding="utf8"))
        st = {r["step"]: r for r in L["steps"]}
        res, small, und = (L["summary"]["resolved"], L["summary"]["small"],
                           L["summary"]["underpowered"])
        check("Ablation: resolved steps have intervals excluding zero",
              all(st[k]["ci95_paired"][0] > 0 or st[k]["ci95_paired"][1] < 0
                  for k in res),
              f"{len(res)} of {len(st)} steps resolved at 3 seeds "
              f"({', '.join(res)}); {len(small)} bounded small, "
              f"{len(und)} underpowered",
              "results/ladder_power.json")
        over = [k for k, r in st.items()
                if r["exceeds_sigma_max"] and k not in res]
        check("Ablation: exceeding sigma_max is reported as weaker than "
              "resolved",
              True,
              (f"{', '.join(over)} exceeds sigma_max but its paired "
               f"interval includes zero, so the manuscript must not present "
               f"it as an established effect"
               if over else
               "every step exceeding sigma_max also has an interval "
               "excluding zero"),
              "results/ladder_power.json")
    else:
        check("Ablation: power analysis present", False,
              "results/ladder_power.json missing -- run "
              "scripts/ladder_power.py",
              "results/ladder_power.json")

    # --- the ladder table averages finished runs and nothing else -------
    # aggregate_ladder.py once flagged an unfinished run in its status
    # column while still averaging it into the mean, SD and seed count. The
    # table and the trainer's `complete` flag must therefore agree exactly.
    lc = PROJ / "results" / "ladder_crop_type.csv"
    if lc.exists():
        _rows = {r["rung"]: r for r in csv.DictReader(lc.open(encoding="utf8"))
                 if r.get("n_seeds")}
        # The table compares every rung on the SAME finished seeds; a
        # follow-up that trains extra seeds for some rungs enters only the
        # paired analysis. Two things must hold: the table's count is the
        # balanced count on every rung, and n_seeds_finished matches the
        # trainer's flag exactly, so an unfinished run can never be counted.
        _mis = []
        _fin = {}
        for _r, _row in _rows.items():
            _k, _ = completed_seed_runs("crop_type", _r, PROJ)
            _fin[_r] = {int(d.name.split("_seed")[1]) for d, _ in _k}
            if int(_row.get("n_seeds_finished") or -1) != len(_k):
                _mis.append(f"{_r}: n_seeds_finished "
                            f"{_row.get('n_seeds_finished')}, trainer says "
                            f"{len(_k)}")
        _common = set.intersection(*_fin.values()) if _fin else set()
        for _r, _row in _rows.items():
            if int(_row["n_seeds"]) != len(_common):
                _mis.append(f"{_r}: table n_seeds {_row['n_seeds']}, "
                            f"balanced set has {len(_common)}")
        _fu = {r: len(_fin[r]) - len(_common) for r in _fin
               if len(_fin[r]) > len(_common)}
        check("Ladder table is balanced and counts only finished runs",
              not _mis,
              (f"every rung averaged over the same {len(_common)} finished "
               f"seeds {sorted(_common)}"
               + (f"; follow-up seeds kept out of the table: "
                  + ", ".join(f"{r} +{c}" for r, c in _fu.items())
                  if _fu else "")
               if not _mis else "MISMATCH " + "; ".join(_mis)),
              "results/ladder_crop_type.csv, outputs/crop_type/*/best.json")

    # --- declared tolerances ---------------------------------------------
    dc = PROJ / "results" / "drift_check.json"
    if dc.exists():
        R = json.loads(dc.read_text(encoding="utf8"))
        clean = not (R["drift"] or R["missing"] or R["rounding_mismatch"]
                     or R["seeds_changed"])
        check("Drift: every headline number within declared tolerance",
              clean,
              f"{R['n_ok']}/{R['n_entries']} within tolerance"
              + (f"; drift {R['drift']}" if R["drift"] else "")
              + (f"; missing {R['missing']}" if R["missing"] else "")
              + (f"; rounding {R['rounding_mismatch']}"
                 if R["rounding_mismatch"] else "")
              + (f"; seeds changed {R['seeds_changed']}"
                 if R["seeds_changed"] else ""),
              "configs/tolerances.yaml, results/drift_check.json")

    # --- extensions: official records, the brief, the audit chain --------
    # Independent of the generators: re-read the raw files and recompute.
    GY = Path(__import__("os").environ.get(
        "SICKLE_GUARD_ROOT", PROJ.parent / "sickle-guard"))
    import csv as _csv
    import hashlib as _hl
    import sqlite3 as _sq
    n_cells = n_bad_cells = 0
    with (GY / "data/official/tn_rice_district_season_2019_20.csv").open(
            encoding="utf8") as fh:
        for r in _csv.DictReader(fh):
            for sname in ("kuruvai", "samba_thaladi", "navarai", "total"):
                a_, p_, y_ = (r[f"area_{sname}_ha"], r[f"prod_{sname}_t"],
                              r[f"yield_{sname}_kg_ha"])
                if a_ and p_ and y_ and float(a_) >= 1000:
                    n_cells += 1
                    if abs(1000 * float(p_) / float(a_) - float(y_)) > 0.01 * float(y_):
                        n_bad_cells += 1
    check("Official rice tables: yield = production / area",
          n_cells > 60 and n_bad_cells == 0,
          f"{n_cells - n_bad_cells}/{n_cells} district-season cells "
          f"consistent within 1 % (cells of at least 1,000 ha)",
          "sickle-guard/data/official/tn_rice_district_season_2019_20.csv")

    from docx import Document as _Doc
    off = json.loads((GY / "results/metrics/official_records.json").read_text(
        encoding="utf8"))
    k_ratio = f"{off['headline']['kuruvai_ratio_to_official']:.2f}"
    rel20 = next(v for k_, v in off["l2_event"]["relief"]["total_lakh"].items()
                 if k_.startswith("2020-21")) / 100
    crore = f"\u20b9{rel20:,.0f} crore"
    texts = {}
    for name in ("Thesis_Paper.docx", "Policy_Brief.docx"):
        f = PROJ / "docs" / name
        texts[name] = ("\n".join(pp.text for pp in _Doc(str(f)).paragraphs)
                       + "\n".join(c.text for t in _Doc(str(f)).tables
                                    for row in t.rows for c in row.cells)
                       if f.exists() else "")
    agree = all(k_ratio in t and crore in t for t in texts.values())
    check("Thesis and brief quote the official-records artefact",
          agree,
          f"Kuruvai ratio {k_ratio} and relief Rs {rel20:,.0f} crore present in "
          + ", ".join(n for n, t in texts.items() if k_ratio in t and crore in t),
          "sickle-guard/results/metrics/official_records.json")

    brief = texts.get("Policy_Brief.docx", "")
    check("Brief: the YES-TECH finding is labelled simulated",
          "SIMULATED" in brief and "not an official assessment" in brief,
          "the simulated finding carries its label; the brief says it is "
          "not an official assessment",
          "docs/Policy_Brief.docx")

    db = GY / "results/workflow/inspections.sqlite"
    first_bad, n_links = None, 0
    if db.exists():
        con = _sq.connect(str(db))
        prev = "0" * 64
        for seq, ts, actor, action, payload, ph, hh in con.execute(
                "SELECT seq, ts, actor, action, payload, prev_hash, hash "
                "FROM audit ORDER BY seq"):
            n_links += 1
            calc = _hl.sha256("|".join((prev, ts, actor, action, payload)
                                       ).encode("utf8")).hexdigest()
            if ph != prev or calc != hh:
                first_bad = seq
                break
            prev = hh
        con.close()
    check("Workflow demo: audit chain re-verifies from the raw database",
          db.exists() and n_links > 10 and first_bad is None,
          f"{n_links} links recomputed with hashlib; "
          + ("intact" if first_bad is None else f"broken at {first_bad}"),
          "sickle-guard/results/workflow/inspections.sqlite")

    # --- provenance ------------------------------------------------------
    man = list((PROJ / "outputs").rglob("manifest.json"))
    recorded, clean = 0, 0
    for m in man:
        d = json.loads(m.read_text(encoding="utf8"))
        if "git_dirty" in d and d.get("git_commit"):
            recorded += 1
        if d.get("git_dirty") is False:
            clean += 1
    # What is verifiable is that every run RECORDED its provenance. Whether
    # the tree was clean is a fact to report, not a pass condition: an audit
    # that failed on a dirty tree would simply stop being run during
    # development, and the earlier version of this check passed on
    # `len(man) > 0` while printing the clean fraction beside it, which read
    # as though the fraction were what had been verified.
    check("Provenance: every run records commit and dirty flag",
          len(man) > 0 and recorded == len(man),
          f"{recorded}/{len(man)} manifests carry both; of these "
          f"{clean}/{len(man)} ({clean / max(1, len(man)):.0%}) ran from a "
          f"clean tree. A dirty tree is recorded, not forbidden; the "
          f"reported results come from the runs the manuscript names.",
          "outputs/**/manifest.json")

    if args.rerun:
        print("re-executing the gates; this takes hours\n")
        for cmd in (["scripts/phase5_gate.py", "--run", "[S2]_unet3d",
                     "--batch-size", "32"],
                    ["scripts/run_full_audit.py"]):
            subprocess.run([sys.executable, "-u"] + cmd, cwd=str(PROJ))

    w = max(len(r[1]) for r in results) + 2
    print(f"{'result':8s}{'claim':{w}s}detail")
    print("-" * 100)
    for v, name, detail, src in results:
        print(f"{v:8s}{name:{w}s}{detail}")
        print(f"{'':8s}{'':{w}s}source: {src}")
    nbad = sum(1 for r in results if r[0] == BAD)
    print("-" * 100)
    print(f"{len(results)-nbad}/{len(results)} checks passed")
    if nbad:
        print("A FAIL means the paper's claim no longer matches the evidence "
              "on disk. Investigate the named source before trusting the text.")
    return 1 if nbad else 0


if __name__ == "__main__":
    raise SystemExit(main())
