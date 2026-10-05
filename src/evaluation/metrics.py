"""Metrics for the SICKLE binary crop-type task.

Implements the metric set specified in `docs/SICKLE_MASTER_PROMPT.md` Section 10.

CLASS CONVENTION, verified from `reference_impl/utils/dataset.py`:

    mask[1] -= 1                 # crop code 1..15 -> 0..14
    mask[1][mask[1] > 1] = 1     # anything above paddy collapses to 1

    => class 0 = PADDY, class 1 = NON-PADDY

Pixels equal to `ignore_index` are excluded everywhere.

Two levels of metric, and they answer different questions:

  * **Pixel-pooled** (`macro_iou`, `class_iou`, ...) — the SICKLE-comparable
    benchmark metrics. Every pixel counts equally, so large plots dominate.
  * **Plot-Macro IoU** (`plot_macro_iou`) — each plot counts equally. This is
    the metric that matches the study's statistical claim, since the plot is
    the independent spatial unit. It is SECONDARY and does not replace the
    benchmark metric.

Pixel accuracy is deliberately not offered as a headline metric: with this
class balance a model that predicts one class everywhere scores well on it.
The released `[S2]_utae` run is exactly such a model (0.0% paddy IoU).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

PADDY, NON_PADDY = 0, 1
CLASS_NAMES = {PADDY: "paddy", NON_PADDY: "non_paddy"}


# --------------------------------------------------------------------------
# confusion matrix
# --------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2,
              ignore_index: int = -999) -> np.ndarray:
    """(n_classes, n_classes) counts, rows = true, cols = predicted."""
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: true {t.shape} vs pred {p.shape}")
    keep = t != ignore_index
    t, p = t[keep].astype(np.int64), p[keep].astype(np.int64)
    if t.size == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    if t.min() < 0 or t.max() >= n_classes or p.min() < 0 or p.max() >= n_classes:
        raise ValueError("labels outside [0, n_classes) after removing "
                         "ignore_index; check the ignore_index value")
    return np.bincount(t * n_classes + p,
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU. NaN where a class has no true and no predicted pixels."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(1) + cm.sum(0) - tp
    out = np.full(cm.shape[0], np.nan)
    np.divide(tp, denom, out=out, where=denom > 0)
    return out


def prf_from_confusion(cm: np.ndarray):
    """Per-class (precision, recall, f1); NaN where undefined."""
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(0).astype(np.float64)
    true = cm.sum(1).astype(np.float64)
    prec = np.full(cm.shape[0], np.nan)
    rec = np.full(cm.shape[0], np.nan)
    np.divide(tp, pred, out=prec, where=pred > 0)
    np.divide(tp, true, out=rec, where=true > 0)
    f1 = np.full(cm.shape[0], np.nan)
    s = prec + rec
    np.divide(2 * prec * rec, s, out=f1, where=(s > 0) & np.isfinite(s))
    # A class that IS present but was never got right has F1 = 0, not undefined.
    # Only a class absent from both truth and prediction stays NaN.
    present = (true > 0) | (pred > 0)
    f1[present & ~np.isfinite(f1)] = 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------

@dataclass
class CropTypeMetrics:
    macro_iou: float            # PRIMARY
    paddy_iou: float
    non_paddy_iou: float
    macro_f1: float             # SECONDARY
    paddy_precision: float
    paddy_recall: float
    non_paddy_precision: float
    non_paddy_recall: float
    n_pixels: int

    def as_dict(self) -> dict:
        return asdict(self)


def crop_type_metrics(y_true, y_pred, ignore_index: int = -999) -> CropTypeMetrics:
    """Pixel-pooled benchmark metrics for the binary task."""
    cm = confusion(y_true, y_pred, 2, ignore_index)
    iou = iou_from_confusion(cm)
    prec, rec, f1 = prf_from_confusion(cm)
    return CropTypeMetrics(
        macro_iou=float(np.nanmean(iou)),
        paddy_iou=float(iou[PADDY]),
        non_paddy_iou=float(iou[NON_PADDY]),
        macro_f1=float(np.nanmean(f1)),
        paddy_precision=float(prec[PADDY]),
        paddy_recall=float(rec[PADDY]),
        non_paddy_precision=float(prec[NON_PADDY]),
        non_paddy_recall=float(rec[NON_PADDY]),
        n_pixels=int(cm.sum()),
    )


# --------------------------------------------------------------------------
# Plot-level generalisation
# --------------------------------------------------------------------------

def plot_macro_iou(y_true, y_pred, plot_ids, ignore_index: int = -999,
                   min_pixels: int = 1, return_per_plot: bool = False):
    """Mean per-plot IoU, every plot weighted equally.

    Rules, fixed in advance because the data forces the question (plots rotate
    crops across seasons, so one plot contributes several samples):

      * **Aggregation unit:** all of a plot's valid pixels, across all of its
        samples, are pooled into ONE confusion matrix before its IoU is taken.
        Per-sample IoU is NOT averaged.
      * **Undefined classes:** a class absent from a plot (no true and no
        predicted pixels) has undefined IoU there. It is EXCLUDED from that
        plot's macro average, never replaced by 0.
      * A plot with fewer than `min_pixels` valid pixels is skipped and counted.

    `plot_ids` must broadcast against `y_true` (band 0 of the SICKLE mask).
    """
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    g = np.asarray(plot_ids).ravel()
    if not (t.shape == p.shape == g.shape):
        raise ValueError(f"shape mismatch: true {t.shape}, pred {p.shape}, "
                         f"plot_ids {g.shape}")

    keep = t != ignore_index
    t, p, g = t[keep], p[keep], g[keep]

    per_plot: dict[float, float] = {}
    skipped = 0
    excluded_classes = 0
    for pid in np.unique(g):
        if pid == ignore_index or pid < 0:
            continue
        m = g == pid
        if m.sum() < min_pixels:
            skipped += 1
            continue
        cm = confusion(t[m], p[m], 2, ignore_index)
        iou = iou_from_confusion(cm)
        excluded_classes += int(np.isnan(iou).sum())
        if np.all(np.isnan(iou)):
            skipped += 1
            continue
        per_plot[float(pid)] = float(np.nanmean(iou))

    vals = np.array(list(per_plot.values()), dtype=np.float64)
    summary = {
        "plot_macro_iou": float(vals.mean()) if vals.size else float("nan"),
        "plot_iou_sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        "n_plots": int(vals.size),
        "n_plots_skipped": int(skipped),
        "n_undefined_class_instances": int(excluded_classes),
    }
    return (summary, per_plot) if return_per_plot else summary


def bootstrap_plot_ci(per_plot: dict[float, float], n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0):
    """95% CI for Plot-Macro IoU by resampling PLOTS, never pixels.

    With 36 validation plots, resampling pixels would understate the interval
    by orders of magnitude: pixels within a plot are not independent.
    """
    vals = np.array(list(per_plot.values()), dtype=np.float64)
    if vals.size < 2:
        return {"lo": float("nan"), "hi": float("nan"), "n_plots": int(vals.size)}
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_plots": int(vals.size),
            "n_boot": int(n_boot)}


# --------------------------------------------------------------------------
# paired seed-wise comparison
# --------------------------------------------------------------------------

def paired_delta(baseline: dict[int, float], variant: dict[int, float]):
    """Paired difference over the SAME seeds. Keys are seeds."""
    seeds = sorted(set(baseline) & set(variant))
    if not seeds:
        raise ValueError("no shared seeds between baseline and variant")
    d = np.array([variant[s] - baseline[s] for s in seeds], dtype=np.float64)
    mean, sd = float(d.mean()), (float(d.std(ddof=1)) if d.size > 1 else float("nan"))
    # Per the prompt: a mean smaller than its own sd is inconclusive, NOT zero.
    if d.size > 1 and abs(mean) < sd:
        verdict = "inconclusive: small relative to seed-to-seed variability"
    elif d.size > 1:
        verdict = "mean exceeds seed-to-seed sd"
    else:
        verdict = "single seed: no variability estimate"
    return {"seeds": seeds, "n": len(seeds), "deltas": d.tolist(),
            "mean_delta": mean, "sd_delta": sd, "verdict": verdict}
