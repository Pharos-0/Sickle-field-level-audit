"""Can unreliable fields be identified WITHOUT ground truth?

    python scripts/triage_eval.py --config v5

The audit shows per-field failure is systematic. That is only useful if the
difficult fields can be spotted at inference time, when no truth is available.
This tests two signals that are:

  seed_disagreement  the fraction of a field's pixels on which independently
                     seeded models do not all agree. Requires several trained
                     models, which a deployment already has if it trains with
                     more than one seed, and requires NO labels.

  mean_confidence    1 - mean softmax probability of the predicted class,
                     averaged over the field. Requires one model, no labels.

and asks whether either predicts the fields that actually score badly.

Evaluated as a screening test, because that is how it would be used:

  workload   share of fields flagged for human verification
  recall     share of genuinely bad fields that the flag catches
  precision  share of flagged fields that are genuinely bad

A flag is only worth deploying if it catches most bad fields while flagging
far fewer than all of them. The baseline to beat is "check everything"
(workload 1.0, recall 1.0) and "check nothing" (workload 0, recall 0).

CAUTION, and it bounds everything here: there are 36 validation fields. Each
one is 2.8 percentage points of WORKLOAD, and recall is measured over the
unreliable fields only, of which there are few. These numbers indicate
whether the idea is worth pursuing; they do not establish an operating
threshold.

WHICH FIELDS ARE "UNRELIABLE" (corrected 2026-10-12)
----------------------------------------------------
A field is unreliable when its field IoU -- the IoU of the crop(s) it holds,
src/evaluation/metrics.py -- averaged over the seeds is below the threshold.
Before the correction the macro IoU over both classes was used. 35 of 36
fields hold one crop, and for those the macro value falls to 0.5 or less on
a single wrong pixel, so the old set (12 fields) mostly held fields with a
few wrong pixels. Both sets are written: `is_bad` (field IoU, primary) and
`is_bad_macro` (the superseded definition), so every downstream script can
report the original analysis beside the corrected one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import patches, utils                       # noqa: E402
from src.dataset import build_loader                 # noqa: E402
from src.model import build_model                    # noqa: E402
from src.evaluation.metrics import (plot_macro_iou,  # noqa: E402
                                    plot_own_class_iou)
from src.evaluation.regression_metrics import (      # noqa: E402
    geographic_holdout)

OUT = PROJ / "results" / "triage"


def collect(cfg, ckpt, device, task, ig):
    """Per-pixel predicted class and confidence for one checkpoint."""
    model, _ = build_model(cfg, device)
    state = torch.load(ckpt, map_location=torch.device(device),
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    preds, confs, trues, pids = [], [], [], []
    loader, _ = build_loader(cfg, cfg["data"]["eval_split"], "eval")
    with torch.no_grad():
        for batch in loader:
            inputs, masks = batch
            inputs = {k: (v[0].to(device), v[1].to(device))
                      for k, v in inputs.items()}
            prob = nn.Softmax(dim=1)(model(inputs))
            preds.append(prob.argmax(1).cpu().numpy())
            confs.append(prob.max(1).values.cpu().numpy())
            trues.append(masks[task].long().numpy())
            pids.append(masks["plot_mask"].numpy())
    del model
    torch.cuda.empty_cache()
    return (np.concatenate([a.ravel() for a in preds]),
            np.concatenate([a.ravel() for a in confs]),
            np.concatenate([a.ravel() for a in trues]),
            np.concatenate([a.ravel() for a in pids]))


def screen_loo(signal: dict, bad: set, workload: float):
    """Leave-one-out screening at a target workload.

    The rule -- refer the fields whose signal is in the top `workload`
    fraction -- needs no labels, so there is no label leakage in the rule
    itself. What it does use is the signal distribution of the evaluation
    fields, which includes the field being judged. Under LOO, each field's
    threshold is computed from the OTHER fields only, so no field helps set
    the threshold that decides its own fate.
    """
    fields = sorted(signal)
    flagged = set()
    for f in fields:
        others = np.array([signal[g] for g in fields if g != f])
        thr = np.quantile(others, 1.0 - workload)
        if signal[f] >= thr:
            flagged.add(f)
    tp = len(flagged & bad)
    return {"target_workload": workload,
            "workload": len(flagged) / len(fields),
            "recall": tp / len(bad) if bad else float("nan"),
            "precision": tp / len(flagged) if flagged else float("nan"),
            "n_flagged": len(flagged), "n_caught": tp}


def bootstrap_recall(signal: dict, bad: set, workload: float,
                     n_boot: int = 2000, seed: int = 0):
    """CI on recall by resampling FIELDS -- the uncertainty, not just a caveat."""
    fields = sorted(signal)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = [fields[i] for i in rng.integers(0, len(fields), len(fields))]
        x = np.array([signal[f] for f in samp])
        thr = np.quantile(x, 1.0 - workload)
        nb = sum(1 for f in samp if f in bad)
        if nb == 0:
            continue
        tp = sum(1 for f in samp if f in bad and signal[f] >= thr)
        vals.append(tp / nb)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    v = np.array(vals)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n_boot": len(v)}


def screen(signal: dict, bad: set, thresholds):
    """Workload / recall / precision as the flag threshold varies."""
    fields = sorted(signal)
    rows = []
    for t in thresholds:
        flagged = {f for f in fields if signal[f] >= t}
        tp = len(flagged & bad)
        rows.append({
            "threshold": float(t),
            "workload": len(flagged) / len(fields),
            "recall": tp / len(bad) if bad else float("nan"),
            "precision": tp / len(flagged) if flagged else float("nan"),
            "n_flagged": len(flagged), "n_caught": tp,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bad-threshold", type=float, default=0.5)
    ap.add_argument("--field-score", choices=("field", "macro"),
                    default="field",
                    help="which per-field IoU defines an unreliable field; "
                         "'macro' reproduces the superseded analysis")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override; use 0 when another job is running")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    patches.install()
    cfg = utils.load_config(args.config)
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    device = args.device or utils.pick_device(cfg["runtime"]["device"])
    task, ig = cfg["data"]["task"], cfg["data"]["ignore_index"]

    runs = []
    for s in args.seeds:
        c = utils.run_dir(cfg, s) / "checkpoint_best.pth.tar"
        if c.exists():
            runs.append((s, c))
    if len(runs) < 2:
        print("need at least two seeds for a disagreement signal")
        return 1
    print(f"{args.config}: {len(runs)} seeds on {device}\n")

    P, C = [], []
    for s, c in runs:
        utils.set_seed(s)
        p, conf, y_true, pid = collect(cfg, c, device, task, ig)
        P.append(p); C.append(conf)
        print(f"  seed {s}: predictions collected")
    P = np.stack(P); C = np.stack(C)

    valid = (y_true != ig) & (pid > 0)
    # ---- signals, neither of which uses y_true -------------------------
    disagree_px = (P != P[0]).any(axis=0)              # any seed differs
    unconf_px = 1.0 - C.mean(axis=0)

    sig_dis, sig_conf, per_field_iou = {}, {}, {}
    for f in np.unique(pid[valid]):
        m = valid & (pid == f)
        if m.sum() == 0:
            continue
        sig_dis[float(f)] = float(disagree_px[m].mean())
        sig_conf[float(f)] = float(unconf_px[m].mean())

    # ---- truth, used ONLY to score the signals -------------------------
    ious, ious_macro = [], []
    for i in range(len(runs)):
        _, pf = plot_own_class_iou(y_true, P[i], pid, ignore_index=ig,
                                   return_per_plot=True)
        _, pm = plot_macro_iou(y_true, P[i], pid, ignore_index=ig,
                               return_per_plot=True)
        ious.append(pf)
        ious_macro.append(pm)
    per_field_macro = {}
    for f in sig_dis:
        v = [d[f] for d in ious if f in d]
        vm = [d[f] for d in ious_macro if f in d]
        if v:
            per_field_iou[f] = float(np.mean(v))
        if vm:
            per_field_macro[f] = float(np.mean(vm))
    if args.field_score == "macro":
        per_field_iou = per_field_macro

    common = sorted(set(sig_dis) & set(per_field_iou))
    bad = {f for f in common if per_field_iou[f] < args.bad_threshold}
    bad_macro = {f for f in common
                 if per_field_macro.get(f, 1.0) < args.bad_threshold}
    print(f"\n  {len(common)} fields, {len(bad)} with mean IoU < "
          f"{args.bad_threshold} ({100*len(bad)/len(common):.0f}%)")

    results = {"config": args.config, "seeds": [s for s, _ in runs],
               "n_fields": len(common), "n_bad": len(bad),
               "bad_threshold": args.bad_threshold,
               "field_score": args.field_score,
               "n_bad_macro": len(bad_macro),
               "bad_fields": sorted(bad),
               "bad_fields_macro": sorted(bad_macro)}

    for name, sig in (("seed_disagreement", sig_dis),
                      ("low_confidence", sig_conf)):
        x = np.array([sig[f] for f in common])
        y = np.array([per_field_iou[f] for f in common])
        r = float(np.corrcoef(x, y)[0, 1])
        qs = np.quantile(x, [0.5, 0.6, 0.667, 0.75, 0.8, 0.9])
        rows = screen(sig, bad, qs)
        # Pre-specifiable, label-free operating points, validated
        # leave-one-out so no field sets its own threshold.
        loo = []
        for w in (0.50, 0.33, 0.25):
            d_ = screen_loo(sig, bad, w)
            d_["recall_ci"] = bootstrap_recall(sig, bad, w)
            loo.append(d_)
        results[name] = {"corr_with_iou": r, "screening": rows,
                         "leave_one_out": loo}

        print(f"\n  === {name} ===")
        print(f"    correlation with per-field IoU: {r:+.3f} "
              f"(negative is the useful direction)")
        print(f"    {'flag at':>9s}{'workload':>10s}{'recall':>9s}"
              f"{'precision':>11s}   caught")
        for row in rows:
            print(f"    {row['threshold']:9.3f}{row['workload']:10.0%}"
                  f"{row['recall']:9.0%}{row['precision']:11.0%}"
                  f"   {row['n_caught']}/{len(bad)}")
        print(f"    leave-one-out, with 95% CI on recall from resampling "
              f"fields:")
        for d_ in loo:
            ci = d_["recall_ci"]
            print(f"      refer top {d_['target_workload']:.0%}: "
                  f"workload {d_['workload']:.0%}  "
                  f"recall {d_['recall']:.0%} "
                  f"[{ci['lo']:.0%}, {ci['hi']:.0%}]  "
                  f"precision {d_['precision']:.0%}")

    # ---- geographically held-out check ---------------------------------
    # The reviewer's request: does a threshold set in one part of the delta
    # transfer to another? SICKLE labels each plot with a RIVER_PART. We hold
    # out each sub-region in turn, set the referral threshold from the
    # remaining regions only, and apply it to the held-out one.
    import pandas as pd
    tab = pd.read_csv(Path(cfg["data"]["root"]) / cfg["data"]["tabular"])
    region = (tab.dropna(subset=["PLOT_ID"])
              .groupby("PLOT_ID").RIVER_PART.first().to_dict())
    geo = {"note": ("Threshold set from the other regions only -- no field "
                    "contributes to the threshold that judges it; see "
                    "tests/test_geographic_holdout.py. Reported per region "
                    "with sample sizes, because the regions are small."),
           "target_workload": 0.5, "regions": []}
    for name, sig in (("seed_disagreement", sig_dis),
                      ("low_confidence", sig_conf)):
        for r_ in geographic_holdout(sig, bad, region, workload=0.5):
            r_["signal"] = name
            geo["regions"].append(r_)
    results["geographic_holdout"] = geo
    print("\n  === GEOGRAPHICALLY HELD-OUT (threshold from other regions) ===")
    print(f"    {'signal':20s}{'region':18s}{'fields':>7s}{'bad':>5s}"
          f"{'flagged':>9s}{'recall':>9s}")
    for r_ in geo["regions"]:
        rec = "n/a" if r_["recall"] is None else f"{r_['recall']:.0%}"
        print(f"    {r_['signal']:20s}{r_['region']:18s}{r_['n_fields']:7d}"
              f"{r_['n_bad']:5d}{r_['n_flagged']:9d}{rec:>9s}")
    big = [r_ for r_ in geo["regions"] if r_["n_fields"] >= 6]
    print(f"\n    Only {len(big)} of {len(geo['regions'])} region-signal "
          f"combinations have 6 or more fields. The remainder cannot support "
          f"a rate at all.")

    # ---- export the per-field signals ---------------------------------
    # Without this, every screening number in the paper can only be rechecked
    # by re-running inference from the checkpoints. Writing the four per-field
    # quantities to a plain CSV makes the whole screening calculation
    # independently reproducible with numpy alone.
    sig_csv = OUT / f"per_field_signals_{args.config}.csv"
    with sig_csv.open("w", newline="", encoding="utf8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["plot_id", "seed_disagreement", "low_confidence",
                     "mean_iou", "is_bad", "river_part", "mean_macro_iou",
                     "is_bad_macro"])
        for f in common:
            w_.writerow([f"{f:.0f}", f"{sig_dis[f]:.10f}",
                         f"{sig_conf[f]:.10f}", f"{per_field_iou[f]:.10f}",
                         int(f in bad), region.get(f, ""),
                         f"{per_field_macro.get(f, float('nan')):.10f}",
                         int(f in bad_macro)])
    print(f"wrote {sig_csv}  ({len(common)} fields)")

    utils.write_json(OUT / f"triage_{args.config}.json", results)
    print(f"\nwrote {OUT / f'triage_{args.config}.json'}")
    print(f"\nCAUTION: {len(common)} fields, {len(bad)} of them unreliable. "
          f"One field is {100/len(common):.1f} pp of WORKLOAD but "
          f"{100/len(bad):.1f} pp of RECALL, since recall is measured over "
          f"the unreliable fields only. The bootstrap intervals above are "
          f"the honest statement of that uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
