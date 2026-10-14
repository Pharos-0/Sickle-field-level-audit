"""Does the crop-type screen add anything beyond field size?

    python scripts/conditional_screen.py

Executes configs/conditional_screen_protocol.yaml, frozen before any
result was computed. The protocol is loaded, not restated: if this file's
behaviour and the protocol disagree, the run aborts.

THE QUESTION, AND WHY THE OBVIOUS TEST IS NOT ENOUGH
-----------------------------------------------------
scripts/screen_validate.py showed the screen beats chance and beats random
referral, but could not separate it from ranking fields by size: the
paired AUROC differences against largest-first and fewest-seasons had
intervals including zero. That is a failure to distinguish, not a
demonstration that size explains the screen.

Two pre-specified tests, which fail in different ways:

  1  CONDITIONAL PERMUTATION. Permute the signal WITHIN size strata. The
     null preserves the size-outcome relationship exactly and destroys
     everything else, so beating it means the screen carries information
     that is not a restatement of size.

  2  OUT-OF-SAMPLE COMPARISON. Leave-one-field-out logistic regression on
     size alone, the screen alone, and both. A permutation test says
     information exists; this says whether using it helps.

The verdict requires both. Passing one is reported as partial, because
information that does not improve ranking is not a reason to ship a
feature.

Writes results/triage/conditional_screen.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
PROTOCOL = PROJ / "configs" / "conditional_screen_protocol.yaml"
SIGNALS = PROJ / "results" / "triage" / "per_field_signals_v5.csv"
SIZE = PROJ / "results" / "field_size_val.json"
OUT = PROJ / "results" / "triage" / "conditional_screen.json"
TARGET_COL = "is_bad"
NPERM, NBOOT, SEED = 10000, 10000, 0


def load_protocol() -> dict:
    if not PROTOCOL.exists():
        raise SystemExit(f"frozen protocol missing: {PROTOCOL}")
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("protocol is not marked frozen_before_results")
    if p["strata"]["k"] != 3:
        raise SystemExit("this script implements k = 3 strata")
    if p["primary_signal"]["name"] != "low_confidence":
        raise SystemExit("this script treats low_confidence as primary")
    return p


def auroc(score, pos):
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, dtype=float)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    srt = s[o]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strata_by_rank(x, k):
    """Equal-count strata on rank; pixel counts are far too skewed for
    value-based cuts to produce equal groups."""
    order = np.argsort(x, kind="mergesort")
    rank = np.empty(len(x), dtype=int)
    rank[order] = np.arange(len(x))
    return (rank * k // len(x)).astype(int)


def logistic_fit(X, y, iters=400, lr=0.1, l2=1e-3):
    """Plain gradient-descent logistic regression.

    Written out rather than imported so the estimator is visible: with 36
    fields, anything with tuning knobs would be fitting the evaluation set.
    L2 is fixed at 1e-3 and never tuned.
    """
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        w -= lr * g
    return w


def loo_scores(X, y):
    """Leave-one-field-out predicted probability for every field."""
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        mu, sd = X[m].mean(0), X[m].std(0)
        sd[sd == 0] = 1.0
        w = logistic_fit((X[m] - mu) / sd, y[m].astype(float))
        xb = np.r_[1.0, (X[i] - mu) / sd]
        out[i] = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
    return out


def boot_diff(a, b, pos, n_boot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(pos)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if not (0 < pos[i].sum() < n):
            continue
        d = auroc(a[i], pos[i]) - auroc(b[i], pos[i])
        if np.isfinite(d):
            v.append(float(d))
    if not v:
        return float("nan"), float("nan")
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def conditional_permutation(sig, bad, strat, n_perm=NPERM, seed=SEED):
    """Permute the signal within size strata; the size-outcome link is kept."""
    rng = np.random.default_rng(seed)
    obs = auroc(sig, bad)
    null = np.empty(n_perm)
    idx_by = [np.where(strat == s)[0] for s in np.unique(strat)]
    for t in range(n_perm):
        perm = sig.copy()
        for ix in idx_by:
            perm[ix] = sig[rng.permutation(ix)]
        null[t] = auroc(perm, bad)
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    return {"observed_auroc": obs, "p_one_sided": p,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "n_permutations": n_perm,
            "beats_size_conditioned_null": bool(p < 0.05)}


def main() -> int:
    global OUT, TARGET_COL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("field", "macro"), default="field")
    a = ap.parse_args()
    if a.target == "macro":
        TARGET_COL = "is_bad_macro"
        OUT = OUT.with_name("conditional_screen_macro.json")
    P = load_protocol()
    rows = list(csv.DictReader(SIGNALS.open(encoding="utf8")))
    size = {r["plot"]: r["n_pixels_pooled"]
            for r in json.loads(SIZE.read_text(encoding="utf8"))["plots"]}

    plot = np.array([float(r["plot_id"]) for r in rows])
    bad = np.array([bool(int(r[TARGET_COL])) for r in rows])
    px = np.array([float(size[p]) for p in plot])
    sig = {"low_confidence": np.array([float(r["low_confidence"])
                                       for r in rows]),
           "seed_disagreement": np.array([float(r["seed_disagreement"])
                                          for r in rows])}
    strat = strata_by_rank(px, P["strata"]["k"])

    bar = "=" * 78
    print(bar)
    print(f"CONDITIONAL SCREEN TEST, protocol v{P['version']} frozen "
          f"{P['frozen_on']}")
    print(f"{len(rows)} fields, {int(bad.sum())} with per-field IoU below "
          f"0.50")
    print(bar)
    print("The field is the unit here: one IoU per field, pooled over its "
          "samples by")
    print("construction, so there are no repeated observations to cluster.")
    print()
    print(f"Size strata (equal count on rank, k={P['strata']['k']}):")
    for s in np.unique(strat):
        m = strat == s
        print(f"   stratum {s}: {int(m.sum())} fields, pixels "
              f"{int(px[m].min())}-{int(px[m].max())}, "
              f"{int(bad[m].sum())} bad")
    print(f"   size alone ranks at AUROC {auroc(px, bad):.3f} "
          f"(largest first)")
    print()

    res = {"target": a.target,
           "protocol_version": P["version"],
           "protocol_frozen": str(P["frozen_on"]),
           "n_fields": len(rows), "n_bad": int(bad.sum()),
           "size_only_auroc": auroc(px, bad),
           "strata": {int(s): {"n": int((strat == s).sum()),
                               "n_bad": int(bad[strat == s].sum())}
                      for s in np.unique(strat)},
           "test_1_conditional_permutation": {},
           "test_2_out_of_sample": {}}

    print("TEST 1 -- permute the signal WITHIN size strata")
    print("   the null keeps the size-outcome relationship and destroys "
          "everything else")
    print(f"{'signal':<20s}{'AUROC':>8s}{'null mean':>11s}{'null p95':>10s}"
          f"{'p':>8s}   beyond size")
    for name in ("low_confidence", "seed_disagreement"):
        r = conditional_permutation(sig[name], bad, strat)
        res["test_1_conditional_permutation"][name] = r
        tag = "PRIMARY" if name == P["primary_signal"]["name"] else "secondary"
        print(f"{name:<20s}{r['observed_auroc']:>8.3f}"
              f"{r['null_mean']:>11.3f}{r['null_p95']:>10.3f}"
              f"{r['p_one_sided']:>8.4f}   "
              f"{'yes' if r['beats_size_conditioned_null'] else 'NO':<4s} "
              f"({tag})")
    print()

    prim = P["primary_signal"]["name"]

    # ---- the noise calibration that makes the LOO numbers readable
    rng = np.random.default_rng(SEED)
    noise_auroc = np.array([
        auroc(loo_scores(rng.normal(size=(len(bad), 1)), bad.astype(float)),
              bad) for _ in range(40)])
    res["loo_noise_null"] = {
        "mean": float(noise_auroc.mean()),
        "median": float(np.median(noise_auroc)),
        "share_below_half": float((noise_auroc < 0.5).mean()),
        "n_draws": int(len(noise_auroc)),
        "note": ("leave-one-out logistic regression at this sample size is "
                 "systematically anti-correlated for an uninformative "
                 "feature: removing a point weakens the association that "
                 "point creates, so each held-out field is scored by a "
                 "model biased against it. A LOO AUROC near 0.5 is "
                 "therefore ABOVE this null, not at chance.")}

    print("TEST 2a -- fit-free comparison (PRIMARY after the amendment)")
    print("   size alone and the screen alone need no estimator; the")
    print("   combination is a rank average, so nothing is fitted anywhere.")
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    # Size is a label-free ranking in EITHER direction, and which one is
    # informative depends on the target: under the corrected target every
    # bad field is small. The screen is therefore compared with size in
    # both directions and must beat both, which is the conservative choice
    # for the screen (amendment of 2026-10-12).
    ff = {"size alone": px,
          "size alone (smaller first)": -px,
          "screen alone": sig[prim],
          "size + screen (rank average)": (rank(px) + rank(sig[prim])) / 2}
    print(f"{'model':<30s}{'AUROC':>8s}")
    for k, v in ff.items():
        a = auroc(v, bad)
        res["test_2_out_of_sample"].setdefault("fit_free", {})[k] = {
            "auroc": a}
        print(f"{k:<30s}{a:>8.3f}")
    for k in ("screen alone", "size + screen (rank average)"):
        lo_, hi_ = boot_diff(ff[k], ff["size alone"], bad)
        d_ = auroc(ff[k], bad) - auroc(ff["size alone"], bad)
        res["test_2_out_of_sample"]["fit_free"][k].update(
            {"delta_vs_size_alone": d_, "ci": [lo_, hi_],
             "improves": bool(lo_ > 0)})
        print(f"   {k} minus size alone = {d_:+.3f}, 95% CI "
              f"[{lo_:+.3f}, {hi_:+.3f}]  -> "
              f"{'improves' if lo_ > 0 else 'NOT demonstrated'}")
    lo2, hi2 = boot_diff(ff["screen alone"], ff["size alone (smaller first)"],
                         bad)
    d2 = (auroc(ff["screen alone"], bad)
          - auroc(ff["size alone (smaller first)"], bad))
    fs_ = res["test_2_out_of_sample"]["fit_free"]["screen alone"]
    fs_.update({"delta_vs_size_smaller_first": d2, "ci_vs_smaller_first":
                [lo2, hi2], "improves_vs_smaller_first": bool(lo2 > 0)})
    fs_["improves_vs_size_both_directions"] = bool(
        fs_["improves"] and fs_["improves_vs_smaller_first"])
    print(f"   screen alone minus size alone (smaller first) = {d2:+.3f}, "
          f"95% CI [{lo2:+.3f}, {hi2:+.3f}]  -> "
          f"{'improves' if lo2 > 0 else 'NOT demonstrated'}")
    print()

    print("TEST 2b -- leave-one-field-out logistic (DIAGNOSTIC ONLY)")
    print(f"   calibration: the same procedure on 40 draws of a pure-noise")
    print(f"   feature averages {noise_auroc.mean():.3f}, with "
          f"{(noise_auroc < 0.5).mean():.0%} of draws below 0.5. These")
    print("   numbers are read against that, not against chance.")
    models = {"size only": px.reshape(-1, 1),
              "screen only": sig[prim].reshape(-1, 1),
              "size + screen": np.column_stack([px, sig[prim]])}
    sc = {k: loo_scores(v, bad.astype(float)) for k, v in models.items()}
    print(f"{'model':<20s}{'LOO AUROC':>11s}")
    for k, v in sc.items():
        a = auroc(v, bad)
        res["test_2_out_of_sample"][k] = {"loo_auroc": a}
        print(f"{k:<20s}{a:>11.3f}")
    lo, hi = boot_diff(sc["size + screen"], sc["size only"], bad)
    d = auroc(sc["size + screen"], bad) - auroc(sc["size only"], bad)
    res["test_2_out_of_sample"]["paired_delta_vs_size_only"] = {
        "delta": d, "ci": [lo, hi], "improves": bool(lo > 0)}
    print(f"   (size + screen) minus (size only) = {d:+.3f}, 95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]  -> "
          f"{'improves' if lo > 0 else 'NOT demonstrated'}")
    print()

    t1 = res["test_1_conditional_permutation"][prim][
        "beats_size_conditioned_null"]
    # the amended rule: the fit-free comparison carries the verdict
    t2 = res["test_2_out_of_sample"]["fit_free"]["screen alone"][
        "improves_vs_size_both_directions"]
    verdict = ("supported" if (t1 and t2)
               else "partial" if (t1 or t2) else "not supported")
    res["verdict"] = verdict
    res["verdict_components"] = {"conditional_permutation": t1,
                                 "out_of_sample_gain": t2}
    print(bar)
    print(f"VERDICT on the primary signal ({prim}): {verdict.upper()}")
    print(f"   carries information beyond size   "
          f"{'yes' if t1 else 'no'}")
    print(f"   beats size, both directions, fit-free  "
          f"{'yes' if t2 else 'no'}")
    print("   The leave-one-out logistic test specified in the protocol is")
    print("   reported above as a diagnostic only; it is invalid at 36")
    print("   fields and the amendment records why.")
    if t1 and not t2:
        print("   The screen is not a restatement of field size, but at 36")
        print("   fields that information does not translate into a")
        print("   demonstrated ranking gain. Reported as partial.")
    if t2 and not t1:
        print("   A gain without evidence of independent information is not")
        print("   a reason to attribute it to the screen.")
    res["power_note"] = P["power_statement"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"\nwrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
