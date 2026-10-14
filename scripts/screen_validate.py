"""Item 5: does the crop-type reliability screen beat the obvious alternatives?

    python scripts/screen_validate.py

WHY THIS EXISTS
---------------
scripts/yield_screen.py already answers this question for the YIELD screen,
under a protocol frozen before results, with baselines, workload sweeps and
plot-bootstrap intervals. It returned "not demonstrated" and the manuscript
says so.

The CROP-TYPE screen (results/triage/triage_v5.json) was never held to the
same standard. It reports recall and precision at several workloads and a
geographic holdout, but no comparison against random referral, no interval
on AUROC, and no alternative ranking an operator could use without any
model at all. A screen that merely beats nothing is not evidence.

This script supplies the missing comparisons, re-deriving everything from
the stored per-field signals, so no model is re-run and no number here can
drift from the artefact it audits.

THE TWO RESULTS ARE KEPT SEPARATE
---------------------------------
Crop-type screening and yield screening are different tasks with different
targets and different evidence. They live in different files and must not
be summarised as one "reliability" claim.

WHAT COUNTS AS A BASELINE
-------------------------
Every baseline is computable at inference time without labels, the same
constraint the screen itself operates under:

  smallest_first  refer the fields contributing fewest pixels. Needs no
                  model at all, only the mask.
  largest_first   the opposite, in case the relationship runs that way.
  fewest_seasons  refer fields observed in the fewest validation samples.

Random referral is handled as a permutation NULL, not as a sixth ranking.
One random permutation of 36 fields can score an AUROC far from 0.5 by
luck, so treating a single draw as "the random baseline" would let the
comparison turn on which draw was taken. The null is the distribution over
many permutations: its mean is 0.5 by construction, and its spread is what
says whether an observed AUROC is remarkable at this sample size.

A COST THAT IS EASY TO MISS
---------------------------
`seed_disagreement` needs several independently seeded models of the same
configuration. A deployment that trains once cannot compute it at all, so
it is not a free signal: it costs as many trainings as there are seeds.
`low_confidence` needs one model.

DECISION RULE, fixed before the numbers are read
------------------------------------------------
  * informative      the AUROC bootstrap interval excludes 0.5
  * useful           the paired recall excess over random excludes 0 at
                     some workload
  * demonstrated     both of those, AND the paired AUROC difference
                     against every label-free baseline excludes 0
The region holdout is reported but secondary, because regions hold between
2 and 8 fields.


TARGET (corrected 2026-10-12)
-----------------------------
`is_bad` in per_field_signals_v5.csv now marks fields whose field IoU (the
IoU of the crop the field holds) is below 0.50. The superseded target, macro
IoU over both classes, marked most fields with even one wrong pixel. Run
with --target macro to reproduce the original analysis; it is written to a
separate *_macro.json so both stay on the record.

Writes results/triage/screen_validate.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "results" / "triage" / "per_field_signals_v5.csv"
SIZE = PROJ / "results" / "field_size_val.json"
TRIAGE = PROJ / "results" / "triage" / "triage_v5.json"
OUT = PROJ / "results" / "triage" / "screen_validate.json"
TARGET_COL = "is_bad"
WORKLOADS = (0.25, 0.33, 0.50, 0.75)
NBOOT = 10000
SEED = 0
SCREENS = ("seed_disagreement", "low_confidence")
BASELINES = ("smallest_first", "largest_first", "fewest_seasons")


# ----------------------------------------------------------------- stats
def auroc(score, pos):
    """P(score of a bad field > score of a good one), ties counted half."""
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


def at_workload(score, bad, frac):
    """Refer the top `frac` by score. Returns realised workload and counts."""
    n = len(score)
    k = max(1, int(round(frac * n)))
    sel = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:k]
    m = np.zeros(n, dtype=bool)
    m[sel] = True
    caught = int((m & bad).sum())
    return {"n_referred": k, "workload_realised": k / n,
            "n_caught": caught, "n_missed": int(bad.sum()) - caught,
            "recall": caught / max(1, int(bad.sum())),
            "precision": caught / k}


def boot_ci(fn, n, n_boot=NBOOT, seed=SEED):
    """Percentile interval, resampling FIELDS.

    For crop type the field is the independent unit: one IoU per field,
    pooled over all of that field's samples by construction.
    """
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        r = fn(rng.integers(0, n, n))
        if r is not None and np.isfinite(r):
            v.append(float(r))
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def permutation_null(bad, n_perm=NBOOT, seed=SEED):
    """AUROC and recall-at-workload under random referral."""
    rng = np.random.default_rng(seed)
    n = len(bad)
    a, rec = [], {w: [] for w in WORKLOADS}
    for _ in range(n_perm):
        s = rng.permutation(n).astype(float)
        a.append(auroc(s, bad))
        for w in WORKLOADS:
            rec[w].append(at_workload(s, bad, w)["recall"])
    a = np.array(a)
    out = {"auroc_mean": float(a.mean()),
           "auroc_p2_5": float(np.percentile(a, 2.5)),
           "auroc_p97_5": float(np.percentile(a, 97.5)),
           "n_permutations": n_perm, "recall": {}}
    for w in WORKLOADS:
        v = np.array(rec[w])
        out["recall"][str(w)] = {"mean": float(v.mean()),
                                 "p97_5": float(np.percentile(v, 97.5))}
    return out


def auroc_diff_ci(sa, sb, bad, n_boot=NBOOT, seed=SEED):
    """Paired interval on AUROC(a) minus AUROC(b), same resampled fields."""
    rng = np.random.default_rng(seed)
    n = len(bad)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if not (0 < bad[i].sum() < n):
            continue
        da = auroc(sa[i], bad[i]) - auroc(sb[i], bad[i])
        if np.isfinite(da):
            v.append(float(da))
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def recall_excess_ci(s, bad, wl, n_boot=NBOOT, seed=SEED):
    """Interval on recall minus the random expectation, paired per replicate.

    Comparing a bootstrap lower bound against a FIXED constant is a
    different test: the number of bad fields moves under resampling, and
    so does the random expectation. Pairing them inside the replicate is
    the comparison that is actually meant.
    """
    rng = np.random.default_rng(seed)
    n = len(bad)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if bad[i].sum() == 0:
            continue
        r = at_workload(s[i], bad[i], wl)
        v.append(r["recall"] - r["workload_realised"])
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------ data
def load():
    rows = list(csv.DictReader(SRC.open(encoding="utf8")))
    size = {r["plot"]: r for r in json.loads(
        SIZE.read_text(encoding="utf8"))["plots"]}
    out = []
    for r in rows:
        p = float(r["plot_id"])
        s = size.get(p, {})
        out.append({
            "plot": p,
            "seed_disagreement": float(r["seed_disagreement"]),
            "low_confidence": float(r["low_confidence"]),
            "mean_iou": float(r["mean_iou"]),
            "bad": bool(int(r[TARGET_COL])),
            "region": r["river_part"],
            "n_pixels": float(s.get("n_pixels_pooled", float("nan"))),
            "n_samples": float(s.get("n_samples", float("nan")))})
    return out


def signals(d):
    """Every ranking under test. Higher score means refer sooner."""
    px = np.array([r["n_pixels"] for r in d])
    ns = np.array([r["n_samples"] for r in d])
    return {
        "seed_disagreement": np.array([r["seed_disagreement"] for r in d]),
        "low_confidence": np.array([r["low_confidence"] for r in d]),
        "smallest_first": -px,
        "largest_first": px,
        "fewest_seasons": -ns}


def geographic_holdout(d, key):
    """Threshold set from the OTHER regions at a 50% workload, then applied.

    Pooled over regions and scored once, which triage_v5.json does not do:
    it reports per-region recalls on 2 to 8 fields, two of which are
    undefined because the region contains no bad field.
    """
    s = np.array([r[key] for r in d])
    bad = np.array([r["bad"] for r in d])
    reg = np.array([r["region"] for r in d])
    flag = np.zeros(len(d), dtype=bool)
    per_region = []
    for g in sorted(set(reg)):
        m = reg == g
        rest = s[~m]
        if rest.size == 0:
            continue
        thr = float(np.quantile(rest, 0.5))
        flag[m] = s[m] > thr
        per_region.append({"region": g, "n_fields": int(m.sum()),
                           "n_bad": int(bad[m].sum()),
                           "n_flagged": int(flag[m].sum()), "threshold": thr,
                           "n_fields_setting_threshold": int((~m).sum())})
    caught = int((flag & bad).sum())
    return {
        "per_region": per_region,
        "pooled": {
            "n_fields": len(d), "n_bad": int(bad.sum()),
            "n_flagged": int(flag.sum()),
            "workload_realised": float(flag.mean()),
            "n_caught": caught,
            "recall": caught / max(1, int(bad.sum())),
            "precision": caught / max(1, int(flag.sum())),
            "recall_ci": boot_ci(
                lambda i: ((flag[i] & bad[i]).sum() / bad[i].sum()
                           if bad[i].sum() else None), len(d)),
            "random_expected_recall": float(flag.mean())},
        "note": ("no field contributes to the threshold that judges it. "
                 "Regions are small, so only the pooled line is "
                 "interpretable; per-region recalls on 2-8 fields are not.")}


def _and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- report
def main() -> int:
    global OUT, TARGET_COL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("field", "macro"), default="field")
    a = ap.parse_args()
    if a.target == "macro":
        TARGET_COL = "is_bad_macro"
        OUT = OUT.with_name("screen_validate_macro.json")
    d = load()
    bad = np.array([r["bad"] for r in d])
    n = len(d)
    sig = signals(d)
    null = permutation_null(bad)
    tri = json.loads(TRIAGE.read_text(encoding="utf8"))
    bar = "=" * 76

    print(bar)
    print(f"CROP-TYPE RELIABILITY SCREEN, rung {tri['config']}, "
          f"seeds {tri['seeds']}")
    print(f"{n} validation fields, {int(bad.sum())} of them bad "
          f"(per-field IoU < {tri['bad_threshold']})")
    print(bar)
    print("The IoU < 0.50 cut is this study's stated convention, not an "
          "official criterion.")
    print("seed_disagreement needs several independently seeded models and "
          "is not free;")
    print("low_confidence needs one model.")
    print()

    res = {"target": a.target, "n_fields": n, "n_bad": int(bad.sum()),
           "bad_threshold": tri["bad_threshold"], "rung": tri["config"],
           "seeds": tri["seeds"], "n_boot": NBOOT,
           "permutation_null": null, "signals": {}}

    print("RANKING POWER (AUROC; 1.0 perfect, 0.5 chance)")
    print(f"  random referral over {null['n_permutations']} permutations: "
          f"mean {null['auroc_mean']:.3f}, central 95% "
          f"[{null['auroc_p2_5']:.3f}, {null['auroc_p97_5']:.3f}]")
    print(f"  a lucky single permutation reaches {null['auroc_p97_5']:.3f}, "
          f"which is why one draw is not used as the baseline")
    print()
    print(f"{'signal':<20s}{'AUROC':>8s}{'95% CI':>20s}{'excl 0.5':>10s}"
          f"{'> null p97.5':>14s}")
    for name, s in sig.items():
        a = auroc(s, bad)
        lo, hi = boot_ci(
            lambda i, s=s: (auroc(s[i], bad[i])
                            if 0 < bad[i].sum() < len(i) else None), n)
        res["signals"][name] = {
            "auroc": a, "auroc_ci": [lo, hi], "informative": bool(lo > 0.5),
            "above_permutation_p97_5": bool(a > null["auroc_p97_5"]),
            "workloads": []}
        print(f"{name:<20s}{a:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20s}"
              f"{('yes' if lo > 0.5 else 'NO'):>10s}"
              f"{('yes' if a > null['auroc_p97_5'] else 'no'):>14s}")
    print("  An interval that includes 0.5 is reported as not demonstrated, "
          "never as a")
    print("  positive result.")
    print()

    print("AGAINST EACH LABEL-FREE BASELINE, paired on the same resampled "
          "fields")
    print(f"{'screen':<20s}{'baseline':<18s}{'dAUROC':>9s}{'95% CI':>20s}"
          f"{'wins':>7s}")
    for a_name in SCREENS:
        for b_name in BASELINES:
            da = (res["signals"][a_name]["auroc"]
                  - res["signals"][b_name]["auroc"])
            lo, hi = auroc_diff_ci(sig[a_name], sig[b_name], bad)
            res["signals"][a_name].setdefault("vs_baseline", {})[b_name] = {
                "delta_auroc": da, "ci": [lo, hi], "wins": bool(lo > 0)}
            print(f"{a_name:<20s}{b_name:<18s}{da:>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>20s}"
                  f"{('yes' if lo > 0 else 'no'):>7s}")
    print()

    print("RECALL AT A FIXED WORKLOAD, paired against random referral")
    print(f"{'signal':<20s}{'budget':>7s}{'recall':>8s}{'random':>8s}"
          f"{'excess':>8s}{'95% CI':>20s}{'beats':>7s}{'prec':>6s}")
    for name, s in sig.items():
        for wl in WORKLOADS:
            r = at_workload(s, bad, wl)
            lo, hi = recall_excess_ci(s, bad, wl)
            beats = bool(lo > 0)
            r.update({"target_workload": wl,
                      "random_expected_recall": r["workload_realised"],
                      "recall_excess": r["recall"] - r["workload_realised"],
                      "recall_excess_ci": [lo, hi], "beats_random": beats})
            res["signals"][name]["workloads"].append(r)
            print(f"{name:<20s}{wl:>7.2f}{r['recall']:>8.2f}"
                  f"{r['workload_realised']:>8.2f}{r['recall_excess']:>+8.2f}"
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>20s}"
                  f"{('yes' if beats else 'no'):>7s}{r['precision']:>6.2f}")
    print("  excess is recall minus what random referral would get at the "
          "same realised")
    print("  workload, computed inside each bootstrap replicate.")
    print()

    print("HELD OUT BY REGION: threshold from the other regions only")
    res["geographic_holdout"] = {}
    for key in SCREENS:
        h = geographic_holdout(d, key)
        res["geographic_holdout"][key] = h
        q = h["pooled"]
        ok = bool(q["recall_ci"][0] > q["random_expected_recall"])
        q["beats_random"] = ok
        print(f"  {key}: workload {q['workload_realised']:.0%}, recall "
              f"{q['recall']:.2f} 95% CI [{q['recall_ci'][0]:.2f}, "
              f"{q['recall_ci'][1]:.2f}], precision {q['precision']:.2f}; "
              f"random {q['random_expected_recall']:.2f}")
        print("    -> " + ("beats random" if ok else
                           "does NOT beat random at the interval level"))
    print("  Regions hold 2 to 8 fields each, so only the pooled line is "
          "interpretable.")
    print()

    verdict = {}
    for key in SCREENS:
        a = res["signals"][key]
        anywl = any(w["beats_random"] for w in a["workloads"])
        beaten = {b: a["vs_baseline"][b]["wins"] for b in BASELINES}
        held = res["geographic_holdout"][key]["pooled"]["beats_random"]
        ok = bool(a["informative"] and anywl and all(beaten.values()))
        verdict[key] = {
            "auroc_interval_excludes_chance": a["informative"],
            "above_permutation_p97_5": a["above_permutation_p97_5"],
            "recall_beats_random_at_some_workload": anywl,
            "beats_every_label_free_baseline": all(beaten.values()),
            "per_baseline": beaten,
            "held_out_by_region_beats_random": held,
            "verdict": "demonstrated" if ok else "not demonstrated",
            # A single word hides the shape of this result, so the sentence
            # the manuscript should use is written out here rather than
            # left to be reconstructed from the flags above.
            # Derived, not typed: an earlier version hard-coded the
            # workloads at which the screen beat random, which became false
            # the moment the target changed.
            "claim_supported": (
                "ranks bad fields better than chance"
                + ("" if a["above_permutation_p97_5"] else
                   " on the bootstrap interval, though not above the "
                   "permutation null's 97.5th percentile")
                + ", and better than random referral at a "
                + _and([f"{w['target_workload']:.0%}"
                        for w in a["workloads"] if w["beats_random"]])
                + " workload; NOT separable at this sample size from the "
                "weaker label-free baselines "
                + ", ".join(b for b, w in beaten.items() if not w)
                + ", whose own AUROC intervals also include chance -- the "
                "comparison is inconclusive for want of fields, not "
                "because those baselines were shown to be as good")
            if (a["informative"] and anywl and not all(beaten.values()))
            else ("demonstrated against every label-free baseline tested"
                  if ok else
                  "not shown to rank bad fields better than chance")}
        v = verdict[key]
        print(f"VERDICT, {key}: {v['verdict'].upper()}")
        print("  AUROC interval excludes chance            "
              + ("yes" if v["auroc_interval_excludes_chance"] else "no"))
        print("  AUROC above the permutation null p97.5    "
              + ("yes" if v["above_permutation_p97_5"] else "no"))
        print("  recall beats random at some workload      "
              + ("yes" if v["recall_beats_random_at_some_workload"]
                 else "no"))
        print("  beats every label-free baseline           "
              + ("yes" if v["beats_every_label_free_baseline"] else "no")
              + f"   {beaten}")
        print("  survives the region holdout               "
              + ("yes" if v["held_out_by_region_beats_random"] else "no")
              + "   (secondary; regions are tiny)")
        print("  what can be claimed:")
        for ln in __import__("textwrap").wrap(v["claim_supported"], 68):
            print("    " + ln)
    res["verdict"] = verdict
    res["decision_rule"] = (
        "demonstrated requires all three of: the AUROC bootstrap interval "
        "excludes 0.5; the paired recall excess over random excludes 0 at "
        "some workload; and the paired AUROC difference against EVERY "
        "label-free baseline excludes 0. The region holdout is reported "
        "but secondary, because regions hold 2 to 8 fields.")
    res["scope_note"] = (
        "CROP-TYPE screening only. The yield screen is a different "
        "experiment with a different target, reported separately in "
        "results/triage/yield_screen.json. The two must not be combined "
        "into one reliability claim.")
    res["cost_note"] = (
        "seed_disagreement requires several independently seeded models of "
        "the same configuration and cannot be computed by a deployment "
        "that trains once. low_confidence requires one model.")
    res["power_note"] = (
        f"{n} fields, {int(bad.sum())} bad. One field is {1 / n:.1%} of "
        f"workload and {1 / max(1, int(bad.sum())):.1%} of recall. "
        f"Intervals are wide and this cannot establish an operating "
        f"threshold.")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
