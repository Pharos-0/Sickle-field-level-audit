"""Progress of a rung, readable while it is still running.

Cross-shell (the .sh version needs bash, which VS Code on Windows does not use
by default). Reads trainlog.json, which is rewritten every epoch, rather than
best.json, so it works mid-run.

    python scripts/watch.py          # v1 seed 0
    python scripts/watch.py v2 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def main() -> int:
    rung = sys.argv[1] if len(sys.argv) > 1 else "v1"
    seed = sys.argv[2] if len(sys.argv) > 2 else "0"
    d = PROJ / "outputs" / "crop_type" / f"{rung}_seed{seed}"
    log = PROJ / "outputs" / "logs" / f"{rung}_seed{seed}.log"

    print(f"=== {rung} seed {seed} ===")
    if not d.exists():
        print(f"  NOT RUN (nothing at {d})")
        return 0

    tl = d / "trainlog.json"
    if tl.exists():
        try:
            t = json.loads(tl.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            # The trainer rewrites this file every epoch; a read can land
            # mid-write. That is not an error worth alarming about.
            print("  trainlog.json is being written right now, retry")
            return 0
        eps = sorted(t, key=int)
        print(f"  epochs recorded : {len(eps)}")
        for e in eps[-3:]:
            r = t[e]
            print(f"    ep {e:>3s}  loss {r.get('train_loss', 0):.4f}  "
                  f"macro {r.get('val_macro_iou', 0):.4f}  "
                  f"paddy {r.get('val_paddy_iou', 0):.4f}  "
                  f"plot {r.get('val_plot_macro_iou', 0):.4f}  "
                  f"[{r.get('seconds', 0):.0f}s]")

    bj = d / "best.json"
    if bj.exists():
        b = json.loads(bj.read_text(encoding="utf8"))
        state = "COMPLETE" if b.get("complete") else "INCOMPLETE"
        if b.get("killed"):
            state = "KILLED"
        print(f"  best            : epoch {b.get('best_epoch')}  "
              f"pixel {b.get('macro_iou', 0):.4f}  "
              f"plot {b.get('plot_macro_iou', 0):.4f}  "
              f"paddy {b.get('paddy_iou', 0):.4f}   <{state}>")
        if b.get("paddy_iou", 1) < 1e-6:
            print("  DEGENERATE: paddy IoU is 0, one class predicted everywhere")

    if log.exists():
        txt = log.read_text(encoding="utf8", errors="replace").splitlines()
        nan = sum(1 for ln in txt if "non-finite" in ln)
        print(f"  NaN batches     : {nan}" + ("" if nan == 0 else "   <-- INVESTIGATE"))
        tail = [ln for ln in txt if ln.strip()][-1:]
        for ln in tail:
            print(f"  last log line   : {ln.strip()[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
