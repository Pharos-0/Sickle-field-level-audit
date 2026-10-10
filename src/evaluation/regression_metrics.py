"""Per-field metrics for the REGRESSION tasks: yield, and the phenology dates.

`metrics.py` handles crop type, which is segmentation (IoU). Yield, sowing,
transplanting and harvesting day are regression, so the aggregate-vs-field
question has to be asked in error terms instead.

WHY THE FIELD IS FORCED HERE, NOT ARGUED FOR
--------------------------------------------
For crop type the plot had to be argued for as the right unit. For these tasks
it is not optional:

  * A farmer's YIELD is one number for their plot. The SICKLE mask stores
    band 5 as yield DIVIDED BY the plot's pixel count, so a per-pixel error is
    an error on a quantity that scales inversely with field area and means
    nothing. The reference implementation's `get_combined_yield` sums over the
    plot for exactly this reason.
  * A HARVEST DATE is one date for a plot. Half a field is not harvested on a
    different day from the other half.

So "pixel-level error" is not a weaker measurement here, it is a meaningless
one. Everything below aggregates to the field first.

WHAT IS REPORTED, AND WHY EACH
------------------------------
  MAE   mean absolute error -- in the unit itself (kg/ha, or days)
  RMSE  penalises large misses; the pair MAE vs RMSE shows whether error is
        spread evenly or driven by a few bad fields
  MAPE  relative error; the SICKLE paper reports yield as MAPE, so this is the
        number comparable to the published 46.74%

MAPE is undefined at zero and explodes near it, so fields whose true value is
below `mape_floor` are excluded from MAPE only -- never from MAE or RMSE -- and
the count of exclusions is returned. Silently dropping them would flatter the
result on precisely the hardest fields.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float | None
    n_fields: int
    n_excluded_from_mape: int
    mape_floor: float

    def as_dict(self):
        return asdict(self)


def aggregate_to_fields(y_true, y_pred, plot_ids, ignore_index: int = -999,
                        how: str = "sum"):
    """Collapse pixel arrays to one value per field.

    `how="sum"` for yield: band 5 is yield-per-pixel, so a field's yield is the
    sum over its pixels. `how="mean"` for dates: every pixel of a field carries
    the same date, so the mean recovers it and is robust to a stray pixel.

    Pixels marked `ignore_index` -- which includes the adjacent-plot masking
    that keeps other splits' fields out -- are dropped before aggregating.
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    plot_ids = np.asarray(plot_ids).ravel()

    valid = (y_true != ignore_index) & (plot_ids > 0) & np.isfinite(y_true) \
        & np.isfinite(y_pred)
    y_true, y_pred, plot_ids = y_true[valid], y_pred[valid], plot_ids[valid]

    out_true, out_pred = {}, {}
    for pid in np.unique(plot_ids):
        m = plot_ids == pid
        if how == "sum":
            out_true[float(pid)] = float(y_true[m].sum())
            out_pred[float(pid)] = float(y_pred[m].sum())
        else:
            out_true[float(pid)] = float(y_true[m].mean())
            out_pred[float(pid)] = float(y_pred[m].mean())
    return out_true, out_pred


def aggregate_to_field_seasons(y_true, y_pred, plot_ids, ignore_index=-999,
                               how: str = "sum", require_positive=True):
    """Collapse to one value per (SAMPLE, plot) -- a field in a given season.

    This, not `aggregate_to_fields`, is the unit the reference implementation
    uses (`get_combined_yield`) and the unit an insurance decision is made on:
    a crop is insured for a season, not a field in perpetuity. A plot that
    appears in three season-samples contributes three observations.

    Getting this wrong is not a rounding difference. Pooling a plot's pixels
    across all its seasons instead gave MAPE 0.2874 against the released
    0.3274 -- a 12% relative error, enough to fail the reproduction gate --
    while this unit gives 0.3269.

    Arrays must keep their sample axis: shape (N, H, W) or (N, H*W).

    `require_positive` mirrors the reference, which skips a (sample, plot)
    whose true total is not > 0. For yield those are fields with no recorded
    harvest, which carry no percentage error.
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plot_ids = np.asarray(plot_ids)
    if y_true.ndim < 2:
        raise ValueError(
            "arrays must keep their sample axis, e.g. (N, H, W); a flattened "
            "array has already lost the season boundary this function needs")

    n = y_true.shape[0]
    yt = y_true.reshape(n, -1)
    yp = y_pred.reshape(n, -1)
    pm = plot_ids.reshape(n, -1)

    out_true, out_pred = {}, {}
    for i in range(n):
        for pid in np.unique(pm[i]):
            if pid <= 0 or pid == ignore_index:
                continue
            m = (pm[i] == pid) & (yt[i] != ignore_index) & np.isfinite(yt[i])
            if not m.any():
                continue
            tv = yt[i][m].sum() if how == "sum" else yt[i][m].mean()
            if require_positive and not (tv > 0):
                continue
            pv = yp[i][m].sum() if how == "sum" else yp[i][m].mean()
            out_true[(i, float(pid))] = float(tv)
            out_pred[(i, float(pid))] = float(pv)
    return out_true, out_pred


def field_errors(true_by_field: dict, pred_by_field: dict) -> dict:
    """Absolute error per field. The distribution is the point, not the mean."""
    common = sorted(set(true_by_field) & set(pred_by_field))
    return {p: abs(pred_by_field[p] - true_by_field[p]) for p in common}


def regression_metrics(true_by_field: dict, pred_by_field: dict,
                       mape_floor: float = 1e-6) -> RegressionMetrics:
    common = sorted(set(true_by_field) & set(pred_by_field))
    if not common:
        raise ValueError("no fields shared between truth and prediction")

    t = np.array([true_by_field[p] for p in common], dtype=np.float64)
    p = np.array([pred_by_field[p] for p in common], dtype=np.float64)
    err = p - t

    keep = np.abs(t) > mape_floor
    mape = (float(np.mean(np.abs(err[keep] / t[keep])) * 100.0)
            if keep.any() else None)

    return RegressionMetrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mape=mape,
        n_fields=len(common),
        n_excluded_from_mape=int((~keep).sum()),
        mape_floor=mape_floor,
    )


def bootstrap_field_ci(true_by_field: dict, pred_by_field: dict,
                       metric: str = "mape", n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0,
                       mape_floor: float = 1e-6) -> dict:
    """CI by resampling FIELDS, never pixels.

    Pixels inside one field are not independent observations -- they are one
    measurement repeated. Resampling them would shrink the interval by orders
    of magnitude and claim a precision the data does not contain.
    """
    common = sorted(set(true_by_field) & set(pred_by_field))
    if len(common) < 2:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0,
                "n_fields": len(common)}

    t = np.array([true_by_field[p] for p in common], dtype=np.float64)
    p = np.array([pred_by_field[p] for p in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(common)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tb, pb = t[idx], p[idx]
        e = pb - tb
        if metric == "mae":
            vals.append(np.mean(np.abs(e)))
        elif metric == "rmse":
            vals.append(np.sqrt(np.mean(e ** 2)))
        elif metric == "mape":
            k = np.abs(tb) > mape_floor
            if k.any():
                vals.append(np.mean(np.abs(e[k] / tb[k])) * 100.0)
        else:
            raise ValueError(f"unknown metric {metric!r}")

    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0,
                "n_fields": n}
    v = np.array(vals)
    return {"lo": float(np.percentile(v, 100 * alpha / 2)),
            "hi": float(np.percentile(v, 100 * (1 - alpha / 2))),
            "n_boot": len(v), "n_fields": n}


def error_distribution(errors: dict, thresholds=None) -> dict:
    """The audit result: how bad is it on individual fields, not on average?

    An aggregate error hides its own tail. This reports the quantiles and the
    share of fields beyond each threshold, which is what a decision made per
    field actually depends on.
    """
    v = np.array(sorted(errors.values()), dtype=np.float64)
    if v.size == 0:
        return {"n_fields": 0}
    out = {
        "n_fields": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }
    if thresholds:
        out["share_above"] = {
            str(t): float((v > t).mean()) for t in thresholds}
    return out
