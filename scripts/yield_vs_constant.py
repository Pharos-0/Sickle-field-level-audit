"""Released yield checkpoints against a constant, with plot-clustered intervals.

    python scripts/yield_vs_constant.py

Writes results/yield_vs_constant.json.

WHY THIS EXISTS
---------------
The manuscript compared each released yield checkpoint's MAE with that of a
constant predictor (the training-split mean) and reported "24 of 24 lose".
That is a count of point estimates. The 41 validation field-seasons come from
only 14 plots, so the question a reader needs answered is whether any of
those differences is distinguishable from zero. Added 2026-10-13 after an
external review asked for uncertainty on every headline claim.

METHOD, STATED SO IT CAN BE CHECKED
-----------------------------------
  constant     mean YIELD over SPLIT == "train" rows with YIELD > 0 in the
               dataset's tabular CSV. Fixed: it is a function of the training
               split, not of the validation sample, so it is not resampled.
  pairing      for each checkpoint, per field-season
                   d_i = |pred_i - y_i| - |c - y_i|
               on the SAME field-seasons. Positive d means the checkpoint
               erred more than the constant on that field-season.
  statistic    mean(d) = MAE(checkpoint) - MAE(constant). Sign convention:
               positive = checkpoint worse.
  resampling   cluster bootstrap over PLOTS: draw 14 plots with replacement
               and take every field-season of each drawn plot, so repeated
               field-seasons of one plot always travel together. The same
               draws are used for every checkpoint.
  interval     percentile, 95 %, B = 10,000, seed 0; recomputed at seed 1 to
               show the Monte Carlo error is negligible.
  caveat       a percentile bootstrap with 14 clusters tends to be too
               NARROW, not too wide. An interval that includes zero here would
               only widen under a better method, so "not distinguishable from
               zero" is the conservative reading; "above zero" would need a
               second method before being claimed.

The correlation of the best checkpoint with the reference yield is given the
same treatment, and is then DECOMPOSED (added 2026-10-14). An earlier
manuscript read r = +0.559 as "it does order field-seasons informatively".
A loss assessment turns on a season's shortfall, so the question is whether
the association comes from the model tracking a plot's better and worse
seasons or only from knowing which plots usually yield more:
  between plots  Pearson r over the 14 plot means of prediction and of
                 reference yield (each plot's field-seasons averaged with
                 equal weight); n = number of plots.
  within plots   Pearson r over all 41 field-seasons after subtracting each
                 plot's own mean from its prediction and from its reference;
                 only season-to-season variation inside a plot remains.
Both are computed on the same prediction set as the overall r, and both get
the same plot-clustered bootstrap. With two to four seasons per plot the
within-plot figure can show the absence of evidence for season tracking, not
that such skill is impossible.

SECOND METHODS for the MAE gaps (added 2026-10-14): a t-interval on the 14
plot means of the paired difference, and an exact-style sign-flip test that
flips each plot's mean difference (20,000 draws). Both weight plots equally
rather than field-seasons, so they answer a slightly different question;
they are reported as a robustness check, not as the headline interval.
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "yield_vs_constant.json"
B = 10_000


def data_root() -> str:
    for line in (PROJ / "configs/base.yaml").read_text(encoding="utf8").splitlines():
        if line.strip().startswith("root:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    raise SystemExit("configs/base.yaml has no data root")


def load():
    gl = [x for x in sorted(glob.glob(str(
        PROJ / "results/regression_gate/crop_yield_*_bs32.json")))
        if "_debug" not in x]
    gl = [x for x in gl
          if "per_field_pred" in json.loads(Path(x).read_text(encoding="utf8"))]
    if not gl:
        raise SystemExit("no released yield checkpoints with per-field predictions")
    runs = [json.loads(Path(x).read_text(encoding="utf8")) for x in gl]
    truth = {ast.literal_eval(k): float(v)
             for k, v in runs[0]["per_field_true"].items()}
    keys = sorted(truth)
    # every checkpoint must be scored on exactly the same field-seasons, with
    # the same reference values, or the pairing below is meaningless
    for r in runs:
        t = {ast.literal_eval(k): float(v) for k, v in r["per_field_true"].items()}
        if t != truth or set(map(str, keys)) != set(r["per_field_pred"]):
            raise SystemExit(f"{r['run']}: evaluated on different field-seasons")
    return runs, keys, truth


def check_keys(keys):
    """Keys are (sample index, plot id). Confirm against the per-field CSV the
    yield screen wrote, which carries both columns by name."""
    f = PROJ / "results/triage/yield_per_field.csv"
    with open(f, encoding="utf8") as fh:
        rows = {(int(r["sample"]), float(r["plot_id"])) for r in csv.DictReader(fh)}
    mine = {(int(k[0]), float(k[1])) for k in keys}
    if rows != mine:
        raise SystemExit(f"key layout disagrees with {f.name}: "
                         f"{len(rows ^ mine)} mismatches")


def main() -> int:
    runs, keys, truth = load()
    check_keys(keys)
    y = np.array([truth[k] for k in keys])
    plot = np.array([float(k[1]) for k in keys])
    plots = np.unique(plot)
    members = {p: np.nonzero(plot == p)[0] for p in plots}

    trn = []
    with open(f"{data_root()}/sickle_dataset_tabular.csv", encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["YIELD"])
            except (TypeError, ValueError):
                continue
            if r["SPLIT"] == "train" and v > 0:
                trn.append(v)
    c = float(np.mean(trn))
    const_err = np.abs(c - y)

    def draws(seed):
        rng = np.random.default_rng(seed)
        return [np.concatenate([members[p] for p in
                                rng.choice(plots, len(plots), replace=True)])
                for _ in range(B)]

    D0, D1 = draws(0), draws(1)
    out = []
    for r in runs:
        pred = np.array([float(r["per_field_pred"][str(k)]) for k in keys])
        d = np.abs(pred - y) - const_err
        b0 = np.array([d[i].mean() for i in D0])
        b1 = np.array([d[i].mean() for i in D1])
        ci0 = np.percentile(b0, [2.5, 97.5])
        ci1 = np.percentile(b1, [2.5, 97.5])
        out.append({
            "run": r["run"],
            "mae": float(np.abs(pred - y).mean()),
            "mae_minus_constant": float(d.mean()),
            "ci95": [float(ci0[0]), float(ci0[1])],
            "ci95_seed1": [float(ci1[0]), float(ci1[1])],
            "share_of_draws_worse": float((b0 > 0).mean()),
            "field_seasons_where_worse": int((d > 0).sum()),
            "ci_above_zero": bool(ci0[0] > 0),
            "ci_below_zero": bool(ci0[1] < 0),
        })
    out.sort(key=lambda o: o["mae_minus_constant"])

    best = out[0]
    rb = next(r for r in runs if r["run"] == best["run"])
    pb = np.array([float(rb["per_field_pred"][str(k)]) for k in keys])
    rs = np.array([np.corrcoef(pb[i], y[i])[0, 1] for i in D0])
    r_ci = np.percentile(rs[np.isfinite(rs)], [2.5, 97.5])

    # ---- decomposition of the best checkpoint's correlation
    plot_ix = np.array([int(np.nonzero(plots == p)[0][0]) for p in plot])
    pm_pred = np.array([pb[members[p]].mean() for p in plots])
    pm_true = np.array([y[members[p]].mean() for p in plots])
    pc = pb - pm_pred[plot_ix]                  # within-plot deviations
    yc = y - pm_true[plot_ix]
    r_between = float(np.corrcoef(pm_pred, pm_true)[0, 1])
    r_within = float(np.corrcoef(pc, yc)[0, 1])
    rho = float(stats.spearmanr(pb, y)[0])
    bw, wi, sp = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(B):
        drawn = rng.choice(len(plots), len(plots), replace=True)
        bw.append(np.corrcoef(pm_pred[drawn], pm_true[drawn])[0, 1])
        idx = np.concatenate([members[plots[j]] for j in drawn])
        wi.append(np.corrcoef(pc[idx], yc[idx])[0, 1])
        sp.append(stats.spearmanr(pb[idx], y[idx])[0])

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    loo = [float(np.corrcoef(pb[plot != p], y[plot != p])[0, 1]) for p in plots]
    decomposition = {
        "pearson_r_overall": float(np.corrcoef(pb, y)[0, 1]),
        "between_plots": {
            "r": r_between, "ci95": ci(bw), "n": int(len(plots)),
            "p_naive": float(stats.pearsonr(pm_pred, pm_true)[1]),
            "method": "Pearson r over plot means (equal weight per "
                      "field-season within a plot); n = plots"},
        "within_plots": {
            "r": r_within, "ci95": ci(wi), "n": int(len(keys)),
            "method": "Pearson r over field-seasons after subtracting each "
                      "plot's own mean from prediction and reference"},
        # a near-zero within-plot r could also mean the model predicts the
        # same value every season; the spreads show whether it moves at all
        "within_plot_sd_pred": float(pc.std()),
        "within_plot_sd_ref": float(yc.std()),
        "spearman_overall": {"rho": rho, "ci95": ci(sp)},
        "leave_one_plot_out_r": [min(loo), max(loo)],
        "bootstrap": "plot-clustered, B = 10,000, seed 0 (separate stream "
                     "from the MAE-gap draws)",
        "reading": "between-plot association with no evidence of within-plot "
                   "season tracking; absence of evidence, not of skill",
    }

    # ---- second methods on the MAE gaps: plots weighted equally
    flips = np.random.default_rng(0).choice([-1.0, 1.0],
                                            size=(20_000, len(plots)))
    tq = stats.t.ppf(0.975, len(plots) - 1)
    second = []
    for r in runs:
        pred = np.array([float(r["per_field_pred"][str(k)]) for k in keys])
        dp = np.array([(np.abs(pred - y) - const_err)[members[p]].mean()
                       for p in plots])
        m_, se_ = dp.mean(), dp.std(ddof=1) / np.sqrt(len(dp))
        null = (flips * dp).mean(1)
        second.append({
            "run": r["run"],
            "plot_weighted_gap": float(m_),
            "t_ci95": [float(m_ - tq * se_), float(m_ + tq * se_)],
            "sign_flip_p": float((np.abs(null) >= abs(m_) - 1e-12).mean()),
        })
    second_summary = {
        "method": "t-interval on plot means of the paired difference; "
                  "sign-flip test over plots, 20,000 draws, seed 0; no "
                  "multiple-comparison correction (one could only reduce "
                  "the counts below)",
        "n_t_ci_above_zero": sum(x["t_ci95"][0] > 0 for x in second),
        "n_sign_flip_p_below_0_05": sum(x["sign_flip_p"] < 0.05 for x in second),
        "per_checkpoint": second,
    }

    mc = max(max(abs(o["ci95"][0] - o["ci95_seed1"][0]),
                 abs(o["ci95"][1] - o["ci95_seed1"][1])) for o in out)
    res = {
        "constant": c,
        "constant_source": "mean YIELD of SPLIT == train rows, YIELD > 0",
        "constant_mae": float(const_err.mean()),
        "n_checkpoints": len(out),
        "n_field_seasons": len(keys),
        "n_plots": int(len(plots)),
        "field_seasons_per_plot": sorted(int(len(v)) for v in members.values()),
        "method": {
            "statistic": "MAE(checkpoint) - MAE(constant) on the same "
                         "field-seasons; positive = checkpoint worse",
            "resampling": "cluster bootstrap over plots; all field-seasons of "
                          "a drawn plot kept together; same draws for every "
                          "checkpoint",
            "interval": "percentile 95 %", "B": B, "seed": 0,
            "monte_carlo_check": "recomputed at seed 1",
            "max_bound_change_seed0_vs_seed1": float(mc),
            "caveat": "with 14 clusters a percentile bootstrap tends to be "
                      "too narrow; an interval including zero is the "
                      "conservative reading",
        },
        "n_worse_point_estimate": sum(o["mae_minus_constant"] > 0 for o in out),
        "n_ci_above_zero": sum(o["ci_above_zero"] for o in out),
        "n_ci_below_zero": sum(o["ci_below_zero"] for o in out),
        "per_checkpoint": out,
        "best": {**best,
                 "pearson_r": float(np.corrcoef(pb, y)[0, 1]),
                 "pearson_r_ci95": [float(r_ci[0]), float(r_ci[1])],
                 "decomposition": decomposition},
        "second_methods": second_summary,
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"constant {c:.1f} kg/acre, MAE {res['constant_mae']:.1f}; "
          f"{len(keys)} field-seasons from {len(plots)} plots")
    print(f"{res['n_worse_point_estimate']}/{len(out)} worse on the point "
          f"estimate; {res['n_ci_above_zero']} with a 95 % interval above "
          f"zero; {res['n_ci_below_zero']} below zero")
    print(f"best {best['run']}: {best['mae_minus_constant']:+.1f} "
          f"[{best['ci95'][0]:+.1f}, {best['ci95'][1]:+.1f}]; r "
          f"{res['best']['pearson_r']:+.3f} "
          f"[{r_ci[0]:+.3f}, {r_ci[1]:+.3f}]")
    print(f"  between plots r {r_between:+.3f} {ci(bw)} (n {len(plots)}); "
          f"within plots r {r_within:+.3f} {ci(wi)} (n {len(keys)})")
    print(f"second methods: t-intervals above zero "
          f"{second_summary['n_t_ci_above_zero']}/{len(second)}; sign-flip "
          f"p < 0.05 {second_summary['n_sign_flip_p_below_0_05']}/{len(second)}")
    print(f"Monte Carlo: bounds move by at most {mc:.1f} kg/acre between seeds")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
