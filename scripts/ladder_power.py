"""Item 4: what three seeds can and cannot resolve, and what more would buy.

    python scripts/ladder_power.py

THE PROBLEM WITH THE CURRENT RULE
---------------------------------
The ablation partitions each rung-to-rung step by comparing its magnitude
against sigma_max, the largest across-seed SD on the ladder. That is a
defensible yardstick with three seeds and the manuscript says so, but it
collapses two very different situations into the single word
"inconclusive":

  genuinely small    the effect is near zero and three seeds are already
                     enough to say it is not large
  underpowered       the effect could be substantial and three seeds
                     cannot tell

A reader cannot distinguish them from the current table, and the second
case is the one that would change if the ladder were re-run with more
seeds. This script separates them.

WHAT IS COMPUTED, AND THE ASSUMPTIONS IT RESTS ON
-------------------------------------------------
Seeds are PAIRED across rungs: seed 0 of v4 and seed 0 of v5 share an
initialisation and a data order, so the step is estimated as a paired
difference, which removes the seed effect common to both rungs and is
materially more powerful than comparing two independent means. This is a
real property of how the ladder was run, not an assumption.

What IS assumed, and cannot be checked at n = 3:

  * that the per-seed differences are roughly normal. With three values
    this is untestable. The t interval below is therefore indicative, and
    is reported alongside the raw three differences so a reader can see
    exactly what it rests on.
  * that the seeds are exchangeable, which follows from how they were
    drawn.

Required-n is the classical paired-t sample size for 80 % power at
alpha = 0.05 two-sided, using the OBSERVED difference and the OBSERVED SD
as if they were the true values. That overstates certainty: both are
estimated from three numbers, and the required n is itself very uncertain.
It is reported as an order-of-magnitude guide to which rungs are worth
re-running, not as a plan.

Writes results/ladder_power.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.utils import completed_seed_runs  # noqa: E402

OUT = PROJ / "results" / "ladder_power.json"
RUNGS = [f"v{i}" for i in range(1, 11)]
POWER, ALPHA = 0.80, 0.05
# Normal approximation constants. With n = 3 a t-based sample-size formula
# would itself need the answer it is computing, so the usual (z_a/2 + z_b)^2
# form is used and the result is treated as a guide.
Z_HALF_ALPHA, Z_POWER = 1.959963985, 0.841621234

# Two-sided t critical values, 95 %, by degrees of freedom. Tabulated rather
# than pulled from scipy so this script has no dependency beyond numpy.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       14: 2.145, 19: 2.093, 29: 2.045}


def t_crit(df):
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    if df < keys[0]:
        return T95[keys[0]]
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return T95[lo]
    w = (df - lo) / (hi - lo)
    return T95[lo] + w * (T95[hi] - T95[lo])


def per_seed(rung):
    kept, _ = completed_seed_runs("crop_type", rung, PROJ)
    out = {}
    for d, b in kept:
        s = d.name.split("_seed")[1]
        out[int(s)] = float(b["macro_iou"])
    return out


def required_n(delta, sd):
    """Paired-t sample size for 80 % power, normal approximation."""
    if sd <= 0:
        return 2 if abs(delta) > 0 else None
    if delta == 0:
        return None
    n = ((Z_HALF_ALPHA + Z_POWER) * sd / abs(delta)) ** 2
    return int(np.ceil(max(2.0, n)))


def main() -> int:
    vals = {r: per_seed(r) for r in RUNGS}
    have = {r: v for r, v in vals.items() if v}
    # sigma_max is the SAME yardstick the balanced table uses: the largest
    # across-seed SD over seeds finished on every rung. Computing it over
    # all seeds let the v3/v4 follow-up runs move it (0.0417 -> 0.0436),
    # so the table and this analysis judged steps against different noise.
    common = set.intersection(*[set(v) for v in have.values()])
    sigma_max = max(float(np.std([v[s_] for s_ in sorted(common)], ddof=1))
                    for v in have.values() if len(common) > 1)

    rows = []
    for a, b in zip(RUNGS, RUNGS[1:]):
        if a not in have or b not in have:
            continue
        shared = sorted(set(have[a]) & set(have[b]))
        if len(shared) < 2:
            continue
        diffs = np.array([have[b][s] - have[a][s] for s in shared])
        n = len(diffs)
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / np.sqrt(n)
        tc = t_crit(n - 1)
        lo, hi = mean - tc * se, mean + tc * se
        unpaired = float(np.mean(list(have[b].values()))
                         - np.mean(list(have[a].values())))
        req = required_n(mean, sd)
        # the distinction the sigma_max rule cannot make
        if lo > 0 or hi < 0:
            # the label carries the seed count it was resolved at, because
            # v3 and v4 are re-run past three and "resolved at n=3" would
            # then misstate how the step was settled
            verdict = f"resolved at n={n}"
        elif abs(hi) < sigma_max and abs(lo) < sigma_max:
            verdict = "small: the interval rules out a large effect"
        else:
            verdict = "underpowered: a large effect is still compatible"
        rows.append({
            "step": f"{a}->{b}", "n_seeds": n, "seeds": shared,
            "per_seed_differences": [round(float(d), 4) for d in diffs],
            "paired_mean": mean, "paired_sd": sd,
            "ci95_paired": [lo, hi], "t_crit": tc,
            "unpaired_mean": unpaired,
            "exceeds_sigma_max": bool(abs(mean) >= sigma_max),
            "required_n_for_80pc_power": req,
            "verdict": verdict})

    print("=" * 78)
    print(f"LADDER POWER, sigma_max = {sigma_max:.4f} "
          f"(largest across-seed SD on the ladder)")
    print("=" * 78)
    print("Steps are PAIRED on seed: seed k of one rung against seed k of "
          "the next, which")
    print("removes the shared seed effect. The sigma_max rule in the "
          "manuscript compares")
    print("magnitudes only; the column that matters here is the last one.")
    print()
    print(f"{'step':10s}{'paired d':>10s}{'sd':>8s}{'95% CI':>20s}"
          f"{'>sig':>6s}{'n for 80%':>11s}  verdict")
    for r in rows:
        ci = f"[{r['ci95_paired'][0]:+.4f}, {r['ci95_paired'][1]:+.4f}]"
        req = ("-" if r["required_n_for_80pc_power"] is None
               else str(r["required_n_for_80pc_power"]))
        print(f"{r['step']:10s}{r['paired_mean']:>+10.4f}{r['paired_sd']:>8.4f}"
              f"{ci:>20s}{('yes' if r['exceeds_sigma_max'] else 'no'):>6s}"
              f"{req:>11s}  {r['verdict']}")

    res = {"resolved": [r["step"] for r in rows
                        if r["verdict"].startswith("resolved")],
           "small": [r["step"] for r in rows
                     if r["verdict"].startswith("small")],
           "underpowered": [r["step"] for r in rows
                            if r["verdict"].startswith("underpowered")]}
    print()
    print(f"  resolved (paired interval excludes 0)  "
          f"{len(res['resolved'])}: {', '.join(res['resolved']) or 'none'}")
    print(f"  small, a large effect is ruled out     "
          f"{len(res['small'])}: {', '.join(res['small']) or 'none'}")
    print(f"  underpowered, still open               "
          f"{len(res['underpowered'])}: "
          f"{', '.join(res['underpowered']) or 'none'}")
    print()
    print("  The manuscript's inconclusive rungs are NOT all the same. The")
    print("  ones listed as small are inconclusive because the effect is")
    print("  small; the underpowered ones are inconclusive because three")
    print("  seeds cannot see them, and those are the ones more seeds would")
    print("  settle.")

    worth = [(r["step"], r["required_n_for_80pc_power"]) for r in rows
             if r["verdict"].startswith("underpowered")
             and r["required_n_for_80pc_power"] is not None]
    worth.sort(key=lambda t: t[1])
    if worth:
        print()
        print("  Cheapest open questions first, by seeds needed for 80 % "
              "power:")
        for st, k in worth:
            print(f"    {st:10s} about {k} seeds "
                  f"({max(0, k - 3)} more than we have)")
        print("  Those counts come from an observed mean and an observed SD,")
        print("  each estimated from three numbers, so they are an "
              "order-of-magnitude")
        print("  guide to what to re-run, not a plan with a guarantee.")

    print()
    print("HONESTY NOTE")
    print("  A 95 % interval from three paired differences uses t with 2 "
          "degrees of")
    print(f"  freedom, where the critical value is {T95[2]}. Such intervals "
          "are very wide")
    print("  and assume a normality that three points cannot support. The "
          "three raw")
    print("  differences are therefore recorded for every step in the JSON, "
          "so a")
    print("  reader can judge the interval rather than take it.")

    OUT.write_text(json.dumps(
        {"sigma_max": sigma_max, "power": POWER, "alpha": ALPHA,
         "sigma_max_seed_set": sorted(common),
         "n_seeds_available": {r: len(v) for r, v in have.items()},
         "steps": rows, "summary": res,
         "method_note": (
             "Paired on seed. Required n is the normal-approximation "
             "paired-t sample size for 80% power at alpha=0.05 two-sided, "
             "using the observed mean and SD as if they were true values. "
             "With three seeds both are badly estimated, so required n is a "
             "guide to which rungs are worth re-running and not a "
             "commitment."),
         "interpretation_note": (
             "resolved = the paired interval excludes zero; small = the "
             "interval lies entirely within +/- sigma_max, so a large "
             "effect is ruled out; underpowered = the interval still admits "
             "an effect larger than sigma_max. The manuscript's single "
             "'inconclusive' label covers the last two, which are not the "
             "same claim.")},
        indent=2), encoding="utf8")
    print()
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
