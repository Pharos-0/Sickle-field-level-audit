"""Training engine for one ladder rung at one seed.

    python -m src.train --config v1
    python -m src.train --config v1 --seed 1 --epochs 2 --smoke

Every run writes, into `outputs/<task>/<version>_seed<N>/`:

    manifest.json    config hash, git commit + dirty flag, device, batch size
    trainlog.json    per-epoch train/val metrics, same shape as the authors'
    best.json        the selected epoch and its full metric set
    checkpoint_best.pth.tar

Both metrics are computed every epoch -- pixel-pooled (benchmark-comparable)
and Plot-Macro (this study's spatial unit) -- because they disagree materially
on this dataset and selecting on one while reporting the other is exactly the
failure mode this study exists to characterise.

MODEL SELECTION IS ON PIXEL-POOLED MACRO IoU, declared here before any run, so
that the ladder stays comparable to the released baselines. Plot-Macro IoU is
recorded at the same epoch but is NOT used to select, which keeps it an
out-of-selection measurement rather than something the run was tuned against.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import patches, utils                               # noqa: E402
from src.dataset import (build_loader, class_weights_from_train,  # noqa: E402
                         effective_micro_batch)
from src.loss import build_loss                              # noqa: E402
from src.model import build_model, count_params              # noqa: E402
from src.evaluation.metrics import (bootstrap_plot_ci,       # noqa: E402
                                    crop_type_metrics,
                                    plot_macro_iou)

SELECTION_METRIC = "macro_iou"      # pixel-pooled; declared before any run


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(c, device) for c in x)
    return x


def build_optimizer(cfg: dict, model):
    t = cfg["training"]
    kind = t["optimizer"].lower()
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"],
                                weight_decay=t["weight_decay"])
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                 weight_decay=t["weight_decay"])
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=0.9,
                               weight_decay=t["weight_decay"])
    raise ValueError(f"training.optimizer {kind!r} not supported")


def build_scheduler(cfg: dict, opt, epochs: int):
    t = cfg["training"]
    kind = (t.get("scheduler") or "none").lower()
    if kind in ("none", "null"):
        return None
    if kind == "cosine":
        warm = int(t.get("warmup_epochs", 0))
        if warm > 0:
            # Linear warmup then cosine. SequentialLR needs the milestone to be
            # the warmup length exactly, or the LR jumps at the handover.
            w = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=warm)
            c = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - warm))
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[w, c], milestones=[warm])
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    raise ValueError(f"training.scheduler {kind!r} not supported")


@torch.no_grad()
def evaluate(model, loader, cfg, device, task, boot_n=0):
    """Both metric families in one pass over the validation split."""
    model.eval()
    ps, ts, gs = [], [], []
    for batch in loader:
        inputs, masks = to_device(batch, device)
        out = nn.Softmax(dim=1)(model(inputs))
        ps.append(out.argmax(1).cpu().numpy())
        ts.append(masks[task].long().cpu().numpy())
        gs.append(masks["plot_mask"].cpu().numpy())

    y_pred = np.concatenate([a.ravel() for a in ps])
    y_true = np.concatenate([a.ravel() for a in ts])
    pid = np.concatenate([a.ravel() for a in gs])
    ig = cfg["data"]["ignore_index"]

    pooled = crop_type_metrics(y_true, y_pred, ignore_index=ig)
    res = dict(pooled.as_dict())

    if cfg["evaluation"]["plot_macro"]:
        summary, per_plot = plot_macro_iou(y_true, y_pred, pid,
                                           ignore_index=ig,
                                           return_per_plot=True)
        res.update({f"plot_{k}" if not k.startswith("plot_") else k: v
                    for k, v in summary.items()})
        if boot_n:
            ci = bootstrap_plot_ci(per_plot, n_boot=boot_n)
            res["plot_ci_lo"], res["plot_ci_hi"] = ci["lo"], ci["hi"]
        res["_per_plot"] = {str(k): v for k, v in sorted(per_plot.items())}
    return res


def train_one_epoch(model, loader, crit, opt, device, task, log_every=10,
                    max_batches=None, accum=1, amp=False):
    """One epoch, with gradient accumulation and optional bf16 autocast.

    `accum` micro-batches contribute to each optimiser step, so the effective
    batch stays `training.batch_size` while only `micro_batch_size` samples sit
    on the GPU at once. The loss is divided by `accum` so the accumulated
    gradient equals the gradient of the mean over the full effective batch.
    """
    model.train()
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    pending = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        inputs, masks = to_device(batch, device)
        y = masks[task].long()

        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(inputs), y)
        else:
            loss = crit(model(inputs), y)

        if not torch.isfinite(loss):
            # A non-finite loss poisons every subsequent step. Skip this
            # micro-batch rather than letting NaNs propagate into a checkpoint
            # that then reports a plausible-looking number.
            print(f"   WARNING micro-batch {i}: non-finite loss "
                  f"{loss.item()}, skipped", flush=True)
            continue

        (loss / accum).backward()
        pending += 1
        total += float(loss.item())
        n += 1

        if pending == accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending = 0

        if log_every and (i + 1) % (log_every * accum) == 0:
            print(f"   micro-batch {i + 1}/{len(loader)}  "
                  f"loss {total / max(n,1):.4f}", flush=True)

    # A trailing partial accumulation would otherwise be silently discarded.
    if pending:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ladder rung, e.g. v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the pipeline moves; not a result")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="override")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override dataloader workers")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.smoke:
        cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 1)

    # Install the UNet3D masked-mean fix BEFORE the model is built. Without it
    # 0.7% of train samples produce NaN logits and, with no guard anywhere in
    # the reference implementation, a single NaN batch destroys every weight.
    # See src/patches.py for the mechanism and the measurement.
    patch_id = patches.install()

    utils.set_seed(args.seed)
    device = utils.pick_device(args.device or cfg["runtime"]["device"])
    task = cfg["data"]["task"]
    epochs = cfg["training"]["epochs"]

    out = utils.run_dir(cfg, args.seed)
    print("=" * 72)
    print(f"{cfg['project']['version']}  seed {args.seed}  ->  {out}")
    print(f"  {cfg['project']['description']}")
    print(f"  device {device}   batch_size {cfg['training']['batch_size']}   "
          f"epochs {epochs}")
    print(f"  sensors {sorted(cfg['satellites'])}   temporal "
          f"{cfg['temporal']['mode']}   model {cfg['model']['name']}")
    print("=" * 72, flush=True)

    micro, accum = effective_micro_batch(cfg)
    use_amp = bool(cfg["training"].get("amp")) and device == "cuda"
    if accum > 1 or use_amp:
        print(f"  micro-batch {micro} x {accum} accumulation "
              f"= effective batch {cfg['training']['batch_size']}"
              + ("   bf16 autocast" if use_amp else ""), flush=True)

    tr_loader, tr_df = build_loader(cfg, cfg["data"]["train_split"], "train",
                                    seed=args.seed)
    va_loader, va_df = build_loader(cfg, cfg["data"]["eval_split"], "eval")
    print(f"train {len(tr_df)} samples / {tr_df.PLOT_ID.nunique()} plots")
    print(f"val   {len(va_df)} samples / {va_df.PLOT_ID.nunique()} plots",
          flush=True)

    model, ns = build_model(cfg, device)
    nparams = count_params(model)
    print(f"trainable params: {nparams:,}", flush=True)

    cw = class_weights_from_train(cfg, tr_df)
    if cw is not None:
        print(f"class weights (paddy, non-paddy): "
              f"{cw[0]:.3f}, {cw[1]:.3f}", flush=True)
    crit = build_loss(cfg, cw, device)
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt, epochs)

    man = utils.build_manifest(cfg, args.seed, device, extra={
        "trainable_params": nparams,
        "train_samples": int(len(tr_df)),
        "train_plots": int(tr_df.PLOT_ID.nunique()),
        "val_samples": int(len(va_df)),
        "val_plots": int(va_df.PLOT_ID.nunique()),
        "selection_metric": SELECTION_METRIC,
        "class_weights": (None if cw is None else [float(x) for x in cw]),
        "smoke": bool(args.smoke),
        "runtime_patch": patch_id,
        "micro_batch_size": micro,
        "grad_accum_steps": accum,
        "amp_bf16": use_amp,
        "precision_note": ("Training uses bf16 autocast and gradient "
                           "accumulation to fit an 8.5 GB card. EVALUATION "
                           "runs in fp32 at the pinned batch size, so every "
                           "reported metric is computed exactly as the "
                           "reproduction gate computed it."),
    })
    utils.write_json(out / "manifest.json", man)
    utils.write_json(out / "config_resolved.json", cfg)

    log, best, best_ep = {}, None, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        loss = train_one_epoch(model, tr_loader, crit, opt, device, task,
                               max_batches=args.max_batches,
                               accum=accum, amp=use_amp)
        if sched is not None:
            sched.step()

        boot = cfg["evaluation"]["bootstrap_n"] if ep == epochs else 0
        m = evaluate(model, va_loader, cfg, device, task, boot_n=boot)
        per_plot = m.pop("_per_plot", None)

        log[str(ep)] = {"train_loss": loss, "lr": opt.param_groups[0]["lr"],
                        "seconds": round(time.time() - te, 1),
                        **{f"val_{k}": v for k, v in m.items()}}
        print(f"ep {ep:3d}/{epochs}  loss {loss:.4f}  "
              f"val_macro_iou {m['macro_iou']:.4f}  "
              f"paddy {m['paddy_iou']:.4f}  "
              f"plot_macro {m.get('plot_macro_iou', float('nan')):.4f}  "
              f"[{time.time() - te:.0f}s]", flush=True)

        if best is None or m[SELECTION_METRIC] > best[SELECTION_METRIC]:
            best, best_ep = dict(m), ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "config_hash": man["config_hash"], "seed": args.seed},
                       out / "checkpoint_best.pth.tar")
            if per_plot:
                utils.write_json(out / "best_per_plot_iou.json", per_plot)
            # Written every time the best improves, not only at the end. A run
            # that is killed -- and these are multi-hour CPU runs -- must still
            # leave behind what it had reached, matching the checkpoint on disk.
            utils.write_json(out / "best.json", {
                "selection_metric": SELECTION_METRIC, "best_epoch": best_ep,
                "epochs_run_so_far": ep, "epochs_planned": epochs,
                "complete": False, **best})

        utils.write_json(out / "trainlog.json", log)

    man["finished_utc"] = utils.datetime.now(utils.timezone.utc).isoformat(
        timespec="seconds")
    man["total_seconds"] = round(time.time() - t0, 1)
    utils.write_json(out / "manifest.json", man)
    utils.write_json(out / "best.json", {
        "selection_metric": SELECTION_METRIC, "best_epoch": best_ep,
        "epochs_run": epochs, "epochs_planned": epochs,
        "complete": True, **(best or {})})

    print("-" * 72)
    if best is None:
        print("NO EPOCH COMPLETED -- no result.")
        return 1
    print(f"best epoch {best_ep}  (selected on {SELECTION_METRIC})")
    print(f"  pixel-pooled macro IoU {best['macro_iou']:.4f}   "
          f"paddy {best['paddy_iou']:.4f}   non-paddy {best['non_paddy_iou']:.4f}")
    if "plot_macro_iou" in best:
        print(f"  plot-macro IoU         {best['plot_macro_iou']:.4f}"
              + (f"   95% CI [{best['plot_ci_lo']:.4f}, {best['plot_ci_hi']:.4f}]"
                 if "plot_ci_lo" in best else ""))
    if best["paddy_iou"] < 1e-6:
        print("  DEGENERATE: paddy IoU is 0 -- the model predicts one class "
              "everywhere. Report this as degenerate, not as a result.")
    if best_ep == 1 and epochs > 1:
        print("  WARNING: best epoch is 1. The run may never have trained; "
              "three of the four released UTAE baselines fail this way.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
