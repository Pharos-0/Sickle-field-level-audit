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
one is 2.8 percentage points of any rate below. These numbers indicate whether
the idea is worth pursuing; they do not establish an operating threshold.
"""

from __future__ import annotations

import argparse
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
from src.evaluation.metrics import plot_macro_iou    # noqa: E402

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
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    patches.install()
    cfg = utils.load_config(args.config)
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
    ious = []
    for i in range(len(runs)):
        _, pf = plot_macro_iou(y_true, P[i], pid, ignore_index=ig,
                               return_per_plot=True)
        ious.append(pf)
    for f in sig_dis:
        v = [d[f] for d in ious if f in d]
        if v:
            per_field_iou[f] = float(np.mean(v))

    common = sorted(set(sig_dis) & set(per_field_iou))
    bad = {f for f in common if per_field_iou[f] < args.bad_threshold}
    print(f"\n  {len(common)} fields, {len(bad)} with mean IoU < "
          f"{args.bad_threshold} ({100*len(bad)/len(common):.0f}%)")

    results = {"config": args.config, "seeds": [s for s, _ in runs],
               "n_fields": len(common), "n_bad": len(bad),
               "bad_threshold": args.bad_threshold}

    for name, sig in (("seed_disagreement", sig_dis),
                      ("low_confidence", sig_conf)):
        x = np.array([sig[f] for f in common])
        y = np.array([per_field_iou[f] for f in common])
        r = float(np.corrcoef(x, y)[0, 1])
        qs = np.quantile(x, [0.5, 0.6, 0.667, 0.75, 0.8, 0.9])
        rows = screen(sig, bad, qs)
        results[name] = {"corr_with_iou": r, "screening": rows}

        print(f"\n  === {name} ===")
        print(f"    correlation with per-field IoU: {r:+.3f} "
              f"(negative is the useful direction)")
        print(f"    {'flag at':>9s}{'workload':>10s}{'recall':>9s}"
              f"{'precision':>11s}   caught")
        for row in rows:
            print(f"    {row['threshold']:9.3f}{row['workload']:10.0%}"
                  f"{row['recall']:9.0%}{row['precision']:11.0%}"
                  f"   {row['n_caught']}/{len(bad)}")

    utils.write_json(OUT / f"triage_{args.config}.json", results)
    print(f"\nwrote {OUT / f'triage_{args.config}.json'}")
    print(f"\nCAUTION: {len(common)} fields. One field is "
          f"{100/len(common):.1f} percentage points of any rate above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
