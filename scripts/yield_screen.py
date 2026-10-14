"""Can inter-model disagreement flag unreliable YIELD estimates?

    python scripts/yield_screen.py

Executes configs/yield_screen_protocol.yaml, which was frozen before any
result was computed. The protocol is loaded, not re-stated: if this file's
behaviour and the protocol disagree, the run aborts rather than silently
preferring the code.

WHY THIS EXPERIMENT
-------------------
The audit's existing screening result (Section V-F) is on crop-type IoU. IoU
is not what a yield-based assessment consumes. A screen that works on
segmentation overlap does not establish that a yield figure can be trusted,
and the manuscript should not be read as if it did. This asks the question
directly on yield.

THE SIGNAL IS LABEL-FREE; THE TARGET IS NOT
-------------------------------------------
Signal  : dispersion of the released checkpoints' predictions for a field-
          season. Computable at inference time. No ground truth.
Target  : whether the ensemble mean's absolute percentage error exceeds the
          protocol threshold. Ground truth is used ONLY here, to score the
          signal after the fact, never to produce it.

INDEPENDENT UNIT IS THE PLOT, NOT THE FIELD-SEASON
--------------------------------------------------
41 field-seasons come from 14 plots, up to four seasons each. Seasons of one
plot share soil, geometry and management. Thresholds are therefore set
leave-one-PLOT-out and the bootstrap resamples PLOTS. n = 14 is the real
sample size and every interval reflects it.

Outputs results/triage/yield_screen.json and yield_per_field.csv.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "triage"
PROTOCOL = PROJ / "configs" / "yield_screen_protocol.yaml"


# ----------------------------------------------------------------- inputs
def load_protocol() -> dict:
    if not PROTOCOL.exists():
        raise SystemExit(f"frozen protocol missing: {PROTOCOL}")
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("protocol is not marked frozen_before_results")
    return p


def load_models() -> tuple[list[str], dict, dict, list[str]]:
    """Per-field-season predictions from every non-debug yield checkpoint."""
    preds, truth, names, skipped = {}, {}, [], []
    for f in sorted(glob.glob(str(
            PROJ / "results/regression_gate/crop_yield_*_bs32.json"))):
        if "_debug" in f:
            continue
        d = json.loads(Path(f).read_text(encoding="utf8"))
        if "per_field_pred" not in d:
            skipped.append(d.get("run", Path(f).stem))
            continue
        names.append(d["run"])
        for k, v in d["per_field_pred"].items():
            preds.setdefault(ast.literal_eval(k), []).append(float(v))
        for k, v in d["per_field_true"].items():
            key = ast.literal_eval(k)
            tv = float(v)
            if key in truth and abs(truth[key] - tv) > 1e-6:
                raise SystemExit(
                    f"ground truth disagrees between checkpoints for {key}: "
                    f"{truth[key]} vs {tv}. Refusing to average over "
                    f"incompatible evaluations.")
            truth[key] = tv
    return names, preds, truth, skipped


# ------------------------------------------------------------- estimators
def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """Rank-based AUROC with explicit tie handling (Mann-Whitney U)."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _require_two_plots(sig, plot_of):
    """Leave-one-plot-out is undefined with a single plot.

    Returning a silent default here would produce a plausible number with no
    held-out comparison behind it, so this refuses instead.
    """
    n = len({plot_of[f] for f in sig})
    if n < 2:
        raise ValueError(
            f"leave-one-plot-out needs at least 2 plots, got {n}; with one "
            f"plot there is no held-out set to set a threshold from")
    return n


def loo_plot_screen(sig: dict, bad: set, plot_of: dict, workload: float):
    """Referral with the threshold set from OTHER PLOTS only."""
    _require_two_plots(sig, plot_of)
    fields = sorted(sig)
    flagged = set()
    for f in fields:
        others = [sig[g] for g in fields if plot_of[g] != plot_of[f]]
        if sig[f] >= float(np.quantile(others, 1.0 - workload)):
            flagged.add(f)
    tp = len(flagged & bad)
    return {"target_workload": workload,
            "workload_realised": len(flagged) / len(fields),
            "n_referred": len(flagged), "n_caught": tp,
            "n_missed": len(bad) - tp,
            "recall": tp / len(bad) if bad else None,
            "precision": tp / len(flagged) if flagged else None}


def plot_bootstrap(sig, bad, plot_of, workload, n_boot, seed):
    """CI on recall AND on the realised workload, resampling PLOTS."""
    _require_two_plots(sig, plot_of)
    fields = sorted(sig)
    plots = sorted({plot_of[f] for f in fields})
    by_plot = {p: [f for f in fields if plot_of[f] == p] for p in plots}
    thr = {f: float(np.quantile(
        [sig[g] for g in fields if plot_of[g] != plot_of[f]], 1.0 - workload))
        for f in fields}
    rng = np.random.default_rng(seed)
    rec, wl = [], []
    for _ in range(n_boot):
        samp = [plots[i] for i in rng.integers(0, len(plots), len(plots))]
        fs = [f for p in samp for f in by_plot[p]]
        sbad = [f for f in fs if f in bad]
        if not sbad:
            continue
        rec.append(sum(1 for f in sbad if sig[f] >= thr[f]) / len(sbad))
        wl.append(sum(1 for f in fs if sig[f] >= thr[f]) / len(fs))
    if not rec:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    r, w = np.array(rec), np.array(wl)
    return {"lo": float(np.percentile(r, 2.5)),
            "hi": float(np.percentile(r, 97.5)),
            "workload_lo": float(np.percentile(w, 2.5)),
            "workload_hi": float(np.percentile(w, 97.5)),
            "n_boot": len(r)}


def auroc_bootstrap(sig, bad, plot_of, n_boot, seed):
    _require_two_plots(sig, plot_of)
    fields = sorted(sig)
    plots = sorted({plot_of[f] for f in fields})
    by_plot = {p: [f for f in fields if plot_of[f] == p] for p in plots}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = [plots[i] for i in rng.integers(0, len(plots), len(plots))]
        fs = [f for p in samp for f in by_plot[p]]
        y = np.array([f in bad for f in fs])
        if y.all() or not y.any():
            continue
        vals.append(auroc(np.array([sig[f] for f in fs]), y))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    v = np.array(vals)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n_boot": len(v)}


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="run with fewer than the protocol's expected_n")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    P = load_protocol()
    thr_ape = float(P["target"]["primary_threshold_ape_pct"])
    workloads = [float(w) for w in P["workloads"]]
    min_mean = float(P["signal"]["denominator_guard"]["min_abs_ensemble_mean"])
    min_true = float(P["target"]["denominator_guard"]["min_abs_truth"])
    expect_n = int(P["checkpoints"]["expected_n"])

    names, preds, truth, skipped = load_models()
    print(f"protocol v{P['version']} frozen {P['frozen_on']}")
    print(f"{len(names)} checkpoints with predictions; {len(skipped)} without")
    if len(names) < expect_n and not args.allow_partial:
        print(f"\nPROTOCOL expects {expect_n} checkpoints, found {len(names)}."
              f"\nMissing: {', '.join(skipped) if skipped else '(unknown)'}"
              f"\nRe-run scripts/regression_gate.py for those runs, or pass "
              f"--allow-partial to proceed and have the shortfall recorded.")
        return 2

    fields = sorted(set(preds) & set(truth))
    # Every field-season must be scored by every checkpoint, or the
    # dispersion of one field is computed over a different model set.
    ragged = [f for f in fields if len(preds[f]) != len(names)]
    if ragged:
        print(f"\n{len(ragged)} field-seasons are not scored by all "
              f"{len(names)} checkpoints; excluding them so dispersion is "
              f"computed over a constant model set.")
        fields = [f for f in fields if f not in set(ragged)]

    # Denominator guards: asserted, not patched with an epsilon.
    dropped_truth = [f for f in fields if abs(truth[f]) < min_true]
    if dropped_truth:
        print(f"excluded {len(dropped_truth)} field-seasons with |truth| < "
              f"{min_true}")
        fields = [f for f in fields if f not in set(dropped_truth)]
    means = {f: float(np.mean(preds[f])) for f in fields}
    bad_mean = [f for f in fields if abs(means[f]) < min_mean]
    if bad_mean:
        raise SystemExit(
            f"{len(bad_mean)} field-seasons have |ensemble mean| < {min_mean}; "
            f"the protocol forbids substituting an epsilon. Investigate "
            f"before proceeding.")

    plot_of = {f: f[1] for f in fields}
    plots = sorted({plot_of[f] for f in fields})
    print(f"{len(fields)} field-seasons from {len(plots)} plots "
          f"(independent unit = plot, n = {len(plots)})\n")

    sig = {f: float(np.std(preds[f], ddof=1) / abs(means[f])) for f in fields}
    ape = {f: abs(means[f] - truth[f]) / abs(truth[f]) * 100.0 for f in fields}
    bad = {f for f in fields if ape[f] > thr_ape}
    print(f"ensemble-mean APE: median {np.median(list(ape.values())):.1f}% | "
          f"{len(bad)}/{len(fields)} above {thr_ape:.0f}% = unreliable")
    if not bad or len(bad) == len(fields):
        print("Degenerate split at this threshold; cannot screen.")
        return 1

    x = np.array([sig[f] for f in fields])
    y = np.array([ape[f] for f in fields])
    pos = np.array([f in bad for f in fields])
    sp = spearman(x, y)
    au = auroc(x, pos)
    au_ci = auroc_bootstrap(sig, bad, plot_of, args.n_boot, args.seed)
    print(f"\nTHRESHOLD-FREE  Spearman(signal, APE) = {sp:+.3f} | "
          f"AUROC = {au:.3f}  95% CI [{au_ci['lo']:.3f}, {au_ci['hi']:.3f}]"
          f"   (random = 0.500)")
    au_sig = not (au_ci["lo"] <= 0.5 <= au_ci["hi"])
    print(f"  AUROC CI {'excludes' if au_sig else 'INCLUDES'} 0.5 -> "
          f"{'informative' if au_sig else 'NOT demonstrated'}")

    # ---- baselines on the same footing ---------------------------------
    area = {}
    for c in Path("D:/FPD/sickledataset extracted").glob("*.csv"):
        with c.open(encoding="utf8") as fh:
            for row in csv.DictReader(fh):
                if row.get("PLOT_ID", "").strip() and row.get("AREA", "").strip():
                    try:
                        area[float(row["PLOT_ID"])] = float(row["AREA"])
                    except ValueError:
                        pass
        break
    base = {"disagreement": sig,
            "small_area_first": {f: -area[plot_of[f]] for f in fields
                                 if plot_of[f] in area},
            "low_predicted_yield_first": {f: -means[f] for f in fields}}
    print("\nBASELINE AUROCs (random = 0.500)")
    base_au = {}
    for nm, s in base.items():
        if len(s) < len(fields):
            print(f"  {nm:<28s} skipped ({len(s)}/{len(fields)} fields have it)")
            continue
        base_au[nm] = auroc(np.array([s[f] for f in fields]), pos)
        print(f"  {nm:<28s} {base_au[nm]:.3f}")

    # ---- screening at each workload ------------------------------------
    print(f"\nSCREENING, threshold leave-one-plot-out, CI bootstraps plots")
    print(f"{'target':>7s}{'realised':>10s}{'referred':>10s}{'recall':>8s}"
          f"{'95% CI':>18s}{'prec':>7s}{'missed':>8s}{'vs random':>11s}")
    rows = []
    for w in workloads:
        s = loo_plot_screen(sig, bad, plot_of, w)
        ci = plot_bootstrap(sig, bad, plot_of, w, args.n_boot, args.seed)
        s["recall_ci"] = ci
        # Random referral at the realised workload has expected recall equal
        # to that workload. The question is whether the CI clears it.
        s["random_expected_recall"] = s["workload_realised"]
        s["beats_random"] = ci["lo"] > s["workload_realised"]
        rows.append(s)
        print(f"{w:>7.0%}{s['workload_realised']:>10.0%}{s['n_referred']:>10d}"
              f"{s['recall']:>8.0%}"
              f"{f'[{ci[chr(108)+chr(111)]:.0%}, {ci[chr(104)+chr(105)]:.0%}]':>18s}"
              f"{s['precision']:>7.0%}{s['n_missed']:>8d}"
              f"{('yes' if s['beats_random'] else 'no'):>11s}")

    # ---- selective prediction ------------------------------------------
    order = sorted(fields, key=lambda f: sig[f])
    sel = []
    print(f"\nSELECTIVE PREDICTION (APE among the automatically accepted)")
    print(f"{'coverage':>9s}{'n kept':>8s}{'median APE':>12s}{'mean APE':>10s}")
    for cov in (1.0, 0.75, 0.67, 0.50):
        keep = order[:max(1, int(round(cov * len(order))))]
        m = [ape[f] for f in keep]
        sel.append({"coverage": cov, "n_kept": len(keep),
                    "median_ape": float(np.median(m)),
                    "mean_ape": float(np.mean(m))})
        print(f"{cov:>9.0%}{len(keep):>8d}{np.median(m):>12.1f}"
              f"{np.mean(m):>10.1f}")
    print("  100% coverage is the unscreened baseline. Read the median AND "
          "the mean:\n  a screen that works moves both. A median that falls "
          "while the mean does\n  not is consistent with reordering noise, "
          "not with identifying error.\n  Neither is a reduction in anyone's "
          "error: referred cases still need resolving.")

    # ---- sensitivity to the unreliability threshold ---------------------
    print(f"\nSENSITIVITY to the unreliability threshold (AUROC)")
    sens = []
    for t in [thr_ape] + [float(v) for v in
                          P["target"]["sensitivity_thresholds_ape_pct"]]:
        b = {f for f in fields if ape[f] > t}
        if not b or len(b) == len(fields):
            sens.append({"ape_threshold": t, "n_bad": len(b), "auroc": None})
            print(f"  APE > {t:>5.0f}%  n_bad {len(b):>3d}   degenerate")
            continue
        a = auroc(x, np.array([f in b for f in fields]))
        sens.append({"ape_threshold": t, "n_bad": len(b), "auroc": a})
        print(f"  APE > {t:>5.0f}%  n_bad {len(b):>3d}   AUROC {a:.3f}")

    # ---- ensemble-independence audit ------------------------------------
    M = np.array([[preds[f][i] for f in fields] for i in range(len(names))])
    C = np.corrcoef(M)
    off = C[np.triu_indices(len(names), 1)]
    fam = {}
    for i, nm in enumerate(names):
        arch = nm.split("_", 1)[1].replace("_season", "") if "_" in nm else nm
        fam.setdefault(arch, []).append(i)
    print(f"\nENSEMBLE INDEPENDENCE  {len(names)} checkpoints, pairwise "
          f"prediction correlation: median {np.median(off):.3f}, "
          f"min {off.min():.3f}, max {off.max():.3f}")
    print(f"  architecture families: "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(fam.items())))
    contrib = {names[i]: float(np.mean(np.abs(M[i] - M.mean(axis=0))))
               for i in range(len(names))}
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    print("  largest mean deviation from the ensemble: "
          + ", ".join(f"{k} ({v:.0f})" for k, v in top))

    # Subset sensitivity, required by the protocol's independence_note: the
    # _season variants are near-duplicates of their base runs, so a
    # dispersion computed over all 24 may be driven by duplicated members.
    subsets = {}
    base_idx = [i for i, nm in enumerate(names) if not nm.endswith("_season")]
    seas_idx = [i for i, nm in enumerate(names) if nm.endswith("_season")]
    for label, idx in (("base_runs_only", base_idx),
                       ("season_runs_only", seas_idx)):
        if len(idx) < 3:
            continue
        sub_mean = {f: float(np.mean([preds[f][i] for i in idx]))
                    for f in fields}
        if any(abs(v) < min_mean for v in sub_mean.values()):
            continue
        sub_sig = {f: float(np.std([preds[f][i] for i in idx], ddof=1)
                            / abs(sub_mean[f])) for f in fields}
        sub_ape = {f: abs(sub_mean[f] - truth[f]) / abs(truth[f]) * 100.0
                   for f in fields}
        sub_bad = np.array([sub_ape[f] > thr_ape for f in fields])
        if sub_bad.all() or not sub_bad.any():
            continue
        subsets[label] = {
            "n_models": len(idx),
            "auroc": auroc(np.array([sub_sig[f] for f in fields]), sub_bad),
            "n_bad": int(sub_bad.sum()),
            "median_ape": float(np.median(list(sub_ape.values())))}
    if subsets:
        print("\nMODEL-SUBSET SENSITIVITY (dispersion over fewer, less "
              "duplicated members)")
        for k, v in subsets.items():
            print(f"  {k:<20s} {v['n_models']:>2d} models  AUROC "
                  f"{v['auroc']:.3f}  n_bad {v['n_bad']}  median APE "
                  f"{v['median_ape']:.1f}%")

    eq = None
    av = {f: area[plot_of[f]] for f in fields if plot_of[f] in area}
    if len(av) >= 10:
        aa = np.array([av[f] for f in av])
        eq = {"n": len(av),
              "spearman_area_vs_ape": spearman(
                  aa, np.array([ape[f] for f in av])),
              "spearman_area_vs_signal": spearman(
                  aa, np.array([sig[f] for f in av]))}
        print(f"\nFIELD SIZE (n={eq['n']}): Spearman area vs APE "
              f"{eq['spearman_area_vs_ape']:+.3f}, area vs signal "
              f"{eq['spearman_area_vs_signal']:+.3f}")

    verdict = ("informative" if au_sig else "not demonstrated")
    res = {"protocol_version": P["version"], "protocol_frozen": P["frozen_on"],
           "n_checkpoints": len(names), "checkpoints": names,
           "checkpoints_without_predictions": skipped,
           "n_field_seasons": len(fields), "n_plots": len(plots),
           "independent_unit": "plot",
           "excluded_ragged": len(ragged),
           "excluded_low_truth": len(dropped_truth),
           "ape_threshold": thr_ape, "n_bad": len(bad),
           "threshold_free": {"spearman_signal_vs_ape": sp, "auroc": au,
                              "auroc_ci": au_ci,
                              "auroc_ci_excludes_half": au_sig},
           "baseline_auroc": base_au,
           "screening": rows, "selective_prediction": sel,
           "sensitivity": sens,
           "ensemble_independence": {
               "pairwise_corr_median": float(np.median(off)),
               "pairwise_corr_min": float(off.min()),
               "pairwise_corr_max": float(off.max()),
               "families": {k: len(v) for k, v in fam.items()}},
           "model_subset_sensitivity": subsets,
           "field_size": eq,
           "verdict": verdict,
           "caveat": P["power_statement"]}
    (OUT / "yield_screen.json").write_text(json.dumps(res, indent=2),
                                           encoding="utf8")
    with (OUT / "yield_per_field.csv").open("w", newline="",
                                            encoding="utf8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["sample", "plot_id", "disagreement_cv", "ensemble_pred",
                     "truth", "ape_pct", "is_bad", "area", "n_models"])
        for f in fields:
            w_.writerow([f[0], f"{f[1]:.0f}", f"{sig[f]:.8f}",
                         f"{means[f]:.4f}", f"{truth[f]:.4f}",
                         f"{ape[f]:.4f}", int(f in bad),
                         area.get(plot_of[f], ""), len(preds[f])])

    print(f"\nVERDICT: the disagreement signal is {verdict.upper()} on this "
          f"sample.")
    print(f"CAVEAT: {len(fields)} field-seasons from {len(plots)} plots. One "
          f"plot is {100/len(plots):.0f}% of the independent sample.")
    print(f"\nwrote {OUT / 'yield_screen.json'}")
    print(f"wrote {OUT / 'yield_per_field.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
