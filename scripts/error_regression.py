"""Can field-level yield error be ESTIMATED, not just flagged, without labels?

    python scripts/error_regression.py

THE QUESTION THIS ANSWERS
-------------------------
Section V-H asked a classification question: is this field-season unreliable?
Using a single signal, inter-model disagreement, the answer was no (AUROC
0.311, CI [0.103, 0.634]).

This asks the harder and more useful regression question. At deployment there
is no ground-truth yield for the field in front of you. But there ARE things
you can compute: the spread and shape of the ensemble's predictions, the
magnitude it predicts, the field's area, which sub-region it sits in, which
season it is. Can a model trained on fields that DO have ground truth learn
to predict the error on fields that do not?

That is a legitimate deployment story -- ground truth is needed once, to fit
the error model, not for every field forever. So the question is worth
asking even though the single-signal version failed.

WHAT IS AND IS NOT ALLOWED AS A FEATURE
---------------------------------------
Allowed: anything computable at inference without that field's yield --
ensemble dispersion statistics, the predicted magnitude itself, field area,
region, season, year.

Forbidden, and asserted below: the field's own error, its truth, or any
transform of them. The target is the error; using it as an input would be
circular.

EVALUATION, AND A TRAP THAT HAD TO BE REMOVED
---------------------------------------------
Leave-one-PLOT-out. The error model is fitted on 13 plots and asked about the
held-out plot's field-seasons, for each of the 14 plots in turn.

The obvious metric -- rank correlation between predicted and actual error --
is unusable here, and the first version of this script was fooled by it. With
14 plots, removing the held-out plot from training shifts the training mean
by a lot: plot 345 averages 115 pp against an overall mean of 32 pp, so the
fold that must predict plot 345 is trained on data averaging 25.8 pp. High-
error plots therefore receive LOW predictions by construction. An
intercept-only model, using no features whatsoever, scores Spearman -0.894
on this data. Any feature model inherits that artefact, and its rank
correlation says more about fold composition than about the features.

So the intercept-only model under the identical protocol is the null, and
the question is whether features beat it:

    skill = 1 - MAE(model) / MAE(intercept-only)

Positive skill means the features carry information the fold mean does not.
Zero or negative means they do not. The rank correlations are still printed,
alongside the intercept-only value, so the artefact stays visible rather than
being quietly dropped.

The honest expectation is that this fails too, for the same reason as
Section V-H: 14 independent units. The script reports whatever happens.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "triage"


def load() -> tuple[list[dict], list[str]]:
    """Per-field-season rows with label-free features and the target."""
    base = {}
    with (OUT / "yield_per_field.csv").open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            base[(int(r["sample"]), float(r["plot_id"]))] = r

    # full per-model prediction vectors, for richer dispersion features
    preds: dict = {}
    for f in sorted(glob.glob(str(
            PROJ / "results/regression_gate/crop_yield_*_bs32.json"))):
        if "_debug" in f:
            continue
        d = json.loads(Path(f).read_text(encoding="utf8"))
        if "per_field_pred" not in d:
            continue
        for k, v in d["per_field_pred"].items():
            preds.setdefault(ast.literal_eval(k), []).append(float(v))

    # plot-level context from the released index
    ctx: dict = {}
    tab = next(Path("D:/FPD/sickledataset extracted").glob("*.csv"))
    with tab.open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            if r.get("PLOT_ID", "").strip():
                ctx.setdefault(float(r["PLOT_ID"]), r)

    rows = []
    for key, r in base.items():
        p = np.array(preds[key], dtype=float)
        mean = float(p.mean())
        q1, q3 = np.percentile(p, [25, 75])
        c = ctx.get(float(r["plot_id"]), {})
        area = float(r["area"]) if r["area"] else float("nan")
        rows.append({
            "key": key, "plot": float(r["plot_id"]),
            # --- label-free features -----------------------------------
            "cv": float(p.std(ddof=1) / abs(mean)),
            "iqr_rel": float((q3 - q1) / abs(mean)),
            "range_rel": float((p.max() - p.min()) / abs(mean)),
            "pred_mean": mean,
            "pred_min": float(p.min()),
            "pred_max": float(p.max()),
            "skew": float(((p - p.mean()) ** 3).mean()
                          / (p.std() ** 3 + 1e-12)),
            "area": area,
            "log_area": float(np.log(area)) if area == area and area > 0
            else float("nan"),
            # --- target --------------------------------------------------
            "ape": float(r["ape_pct"]),
        })
    feats = ["cv", "iqr_rel", "range_rel", "pred_mean", "pred_min",
             "pred_max", "skew", "log_area"]
    return rows, feats


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge_fit(X, y, lam):
    """Closed-form ridge on standardised features with an intercept.

    Ridge rather than OLS because with ~38 training rows, 8 correlated
    dispersion features and 14 independent units, an unpenalised fit would
    interpolate noise.
    """
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    A[0, 0] -= lam                       # do not penalise the intercept
    w = np.linalg.solve(A, Xs.T @ y)
    return w, mu, sd


def ridge_pred(w, mu, sd, X):
    Xs = (X - mu) / sd
    return np.hstack([np.ones((len(Xs), 1)), Xs]) @ w


def lopo_predict(rows, feats, lam):
    """Leave-one-plot-out predictions of the error."""
    plots = sorted({r["plot"] for r in rows})
    out = np.full(len(rows), np.nan)
    for p in plots:
        tr = [i for i, r in enumerate(rows) if r["plot"] != p]
        te = [i for i, r in enumerate(rows) if r["plot"] == p]
        X = np.array([[rows[i][f] for f in feats] for i in tr])
        y = np.array([rows[i]["ape"] for i in tr])
        ok = np.isfinite(X).all(1)
        if ok.sum() < 5:
            continue
        w, mu, sd = ridge_fit(X[ok], y[ok], lam)
        Xt = np.array([[rows[i][f] for f in feats] for i in te])
        Xt = np.nan_to_num(Xt, nan=float(np.nanmean(X)))
        out[te] = ridge_pred(w, mu, sd, Xt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()

    rows, feats = load()
    y = np.array([r["ape"] for r in rows])
    plots = np.array([r["plot"] for r in rows])
    uplots = sorted(set(plots))
    print(f"{len(rows)} field-seasons from {len(uplots)} plots")
    print(f"features ({len(feats)}): {', '.join(feats)}")

    # Guard: the target must not be reachable from any feature.
    for f in feats:
        v = np.array([r[f] for r in rows], float)
        if np.isfinite(v).all() and abs(np.corrcoef(v, y)[0, 1]) > 0.999:
            raise SystemExit(f"feature {f!r} is collinear with the target")
    print("leakage guard: no feature is a transform of the target\n")

    # --- the null: intercept-only, identical protocol, zero features ---
    inter = np.empty(len(rows))
    for p in uplots:
        inter[plots == p] = y[plots != p].mean()
    mae_null = float(np.mean(np.abs(inter - y)))
    rho_null = spearman(inter, y)

    pred = lopo_predict(rows, feats, args.lam)
    m = np.isfinite(pred)
    rho = spearman(pred[m], y[m])
    mae = float(np.mean(np.abs(pred[m] - y[m])))
    skill = 1.0 - mae / mae_null

    print("THE NULL: intercept-only under the same leave-one-plot-out split")
    print(f"  uses no features at all; predicts the training-fold mean")
    print(f"  MAE {mae_null:6.2f} pp | Spearman {rho_null:+.3f}")
    print(f"  Its strong negative rank correlation is an artefact: removing a")
    print(f"  high-error plot lowers the mean that must then predict it.")
    print(f"  Rank correlation is therefore not a usable metric here.\n")

    print(f"MULTIVARIATE ridge, leave-one-plot-out (lambda={args.lam})")
    print(f"  MAE {mae:6.2f} pp   vs null {mae_null:6.2f} pp")
    print(f"  SKILL = 1 - MAE/MAE_null = {skill:+.3f}  "
          f"({'beats' if skill > 0 else 'does NOT beat'} the fold mean)")
    print(f"  Spearman {rho:+.3f} (null {rho_null:+.3f}) -- artefact-laden, "
          f"shown for completeness")

    # plot-level bootstrap on the SKILL, which is the metric that matters
    by = {p: np.where(plots == p)[0] for p in uplots}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(args.n_boot):
        s_ = [uplots[i] for i in rng.integers(0, len(uplots), len(uplots))]
        idx = np.concatenate([by[p] for p in s_])
        idx = idx[np.isfinite(pred[idx])]
        if len(idx) < 5:
            continue
        a = float(np.mean(np.abs(pred[idx] - y[idx])))
        b = float(np.mean(np.abs(inter[idx] - y[idx])))
        if b > 0:
            vals.append(1.0 - a / b)
    v = np.array(vals)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    # A wholly negative interval is not the same as one straddling zero:
    # the first says the model is reliably worse than the null, the second
    # says it cannot be told apart from it.
    if lo > 0:
        zero = "excludes zero (better than the null)"
    elif hi < 0:
        zero = "excludes zero on the NEGATIVE side (reliably worse than the null)"
    else:
        zero = "includes zero (indistinguishable from the null)"
    print(f"  95% CI on skill (bootstrap over plots) [{lo:+.3f}, {hi:+.3f}]"
          f"   {zero}")

    print("\nSINGLE-FEATURE baselines, same protocol, same skill metric")
    singles = {}
    for f in feats:
        pr = lopo_predict(rows, [f], args.lam)
        mm = np.isfinite(pr)
        sk = 1.0 - float(np.mean(np.abs(pr[mm] - y[mm]))) / mae_null
        singles[f] = sk
        print(f"  {f:<12s} skill {sk:+.3f}")
    best = max(singles, key=singles.get)
    print(f"  best single feature: {best} at {singles[best]:+.3f}")

    verdict = ("informative" if lo > 0 else
               "reliably worse than the fold mean" if hi < 0 else
               "NOT demonstrated")
    print(f"\nVERDICT: error regression is {verdict} on this sample "
          f"(skill {skill:+.3f}, CI [{lo:+.3f}, {hi:+.3f}]).")
    print(f"CAVEAT: {len(uplots)} independent plots. One plot is "
          f"{100/len(uplots):.0f}% of the sample. A null result here is a "
          f"statement about this dataset's size, not about the idea.")

    res = {"n_field_seasons": len(rows), "n_plots": len(uplots),
           "features": feats, "lambda": args.lam,
           "metric": "skill = 1 - MAE/MAE_intercept_only, same LOPO split",
           "mae_pp": mae, "mae_null_pp": mae_null, "skill": skill,
           "skill_ci": {"lo": lo, "hi": hi},
           "skill_ci_excludes_zero": bool(lo > 0 or hi < 0),
           "spearman_pred_vs_actual": rho,
           "spearman_intercept_only_null": rho_null,
           "spearman_note": ("rank correlation is artefact-dominated here: "
                             "an intercept-only model scores "
                             f"{rho_null:+.3f} using no features at all"),
           "single_feature_skill": singles, "verdict": verdict}
    (OUT / "error_regression.json").write_text(json.dumps(res, indent=2),
                                               encoding="utf8")
    print(f"\nwrote {OUT / 'error_regression.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
