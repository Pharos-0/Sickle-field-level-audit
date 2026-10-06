"""Whole-ladder progress: what is done, what is running, what is left.

    python scripts/progress.py

Epoch counts come from `trainlog.json`, which is rewritten EVERY epoch, not
from `best.json`, which is only rewritten when the best score improves. Reading
the latter makes a run that has stopped improving look like a run that has
stopped progressing -- v8 showed "epoch 19" while actually on epoch 44.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
LADDER = [f"v{i}" for i in range(1, 11)]
SEEDS = [0, 1, 2]
TASK = "crop_type"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception:
        return None


def main() -> int:
    base = PROJ / "outputs" / TASK
    rows, done_h, n_done, n_run = [], 0.0, 0, 0
    running = None

    for s in SEEDS:
        for v in LADDER:
            d = base / f"{v}_seed{s}"
            if not d.is_dir():
                rows.append((f"{v}_seed{s}", "not started", None, None))
                continue
            log = load(d / "trainlog.json") or {}
            best = load(d / "best.json") or {}
            man = load(d / "manifest.json") or {}
            eps = len(log)
            total = man.get("total_seconds")
            if best.get("complete"):
                n_done += 1
                if total:
                    done_h += total / 3600
                state = "COMPLETE"
            else:
                n_run += 1
                state = f"running {eps}/100"
                secs = [log[k].get("seconds", 0) for k in log]
                rate = (sum(secs[-5:]) / min(len(secs), 5)) if secs else 0
                running = (f"{v}_seed{s}", eps, rate)
            rows.append((f"{v}_seed{s}", state, best, total))

    print(f"{'run':14s}{'state':16s}{'pixel':>8s}{'plot':>8s}{'paddy':>8s}{'time':>8s}")
    print("-" * 62)
    for name, state, best, total in rows:
        if state == "not started":
            print(f"{name:14s}{'NOT RUN':16s}{'--':>8s}{'--':>8s}{'--':>8s}{'--':>8s}")
            continue
        b = best or {}
        flag = ""
        if b.get("complete") and b.get("paddy_iou", 1) < 1e-6:
            flag = "  DEGENERATE"
        if b.get("complete") and b.get("best_epoch") == 1:
            flag += "  best epoch 1"
        print(f"{name:14s}{state:16s}"
              f"{b.get('macro_iou', 0):8.4f}{b.get('plot_macro_iou', 0):8.4f}"
              f"{b.get('paddy_iou', 0):8.4f}"
              + (f"{total/3600:7.2f}h" if total else f"{'--':>8s}") + flag)

    print("-" * 62)
    print(f"COMPLETE {n_done}/{len(LADDER)*len(SEEDS)}   "
          f"in progress {n_run}   compute on finished runs {done_h:.1f} h")

    if running:
        name, eps, rate = running
        left_this = (100 - eps) * rate / 3600
        print(f"\ncurrent: {name} at epoch {eps}/100, {rate:.0f} s/epoch "
              f"-> {left_this:.1f} h left on this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
