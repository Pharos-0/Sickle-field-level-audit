"""How much of the published per-field IoU distribution is the metric's step?

For a field holding one class, macro IoU over {paddy, non-paddy} is 1.0 when
every pixel is right and at most 0.5 as soon as one pixel is predicted as the
absent class (that class's IoU becomes 0). This compares, for the published
rung (v5, three seeds), the per-field macro IoU against two measures without
the step: the IoU of the field's own class, and pixel accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/FPD/sickle-project/scripts")
sys.path.insert(0, "D:/FPD/sickle-project")
import render_sickle_gallery as g  # noqa: E402

per = {"macro": {}, "own_class_iou": {}, "pixel_acc": {}}
pooled = []
for seed in (0, 1, 2):
    R = g.run_rung("v5", seed)
    ig = R["ignore_index"]
    t, p, pid = R["truth"].ravel(), R["pred"].ravel(), R["pid"].ravel()
    k = t != ig
    t, p, pid = t[k], p[k], pid[k]
    pooled.append(R["pooled_macro_iou"])
    for f in np.unique(pid):
        s = pid == f
        tt, pp = t[s], p[s]
        cls = np.unique(tt)
        own = []
        for c in cls:
            inter = np.sum((tt == c) & (pp == c))
            union = np.sum((tt == c) | (pp == c))
            own.append(inter / union)
        per["macro"].setdefault(f, []).append(R["per_plot_iou"][str(float(f))])
        per["own_class_iou"].setdefault(f, []).append(float(np.mean(own)))
        per["pixel_acc"].setdefault(f, []).append(float(np.mean(tt == pp)))

out = {"rung": "v5", "seeds": [0, 1, 2],
       "pooled_macro_iou_mean": float(np.mean(pooled))}
for name, d in per.items():
    x = np.array([np.mean(v) for v in d.values()])
    out[name] = {"median": float(np.median(x)), "mean": float(x.mean()),
                 "n_below_0.5": int((x < 0.5).sum()), "n_fields": int(x.size),
                 "n_at_1": int((x > 0.9999).sum())}
single = 0
R0 = g.run_rung("v5", 0)
t0, pid0 = R0["truth"].ravel(), R0["pid"].ravel()
k0 = t0 != R0["ignore_index"]
for f in np.unique(pid0[k0]):
    single += int(np.unique(t0[k0][pid0[k0] == f]).size == 1)
out["single_class_fields"] = single
print(json.dumps(out, indent=2))
Path("D:/FPD/sickle-project/results/field_metric_check.json").write_text(json.dumps(out, indent=2), encoding="utf8")
