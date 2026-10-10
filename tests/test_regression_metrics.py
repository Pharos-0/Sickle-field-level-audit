"""Tests for the per-field regression metrics.

These pin the behaviour the audit depends on: that aggregation to fields is
correct, that MAPE exclusions are reported rather than hidden, and that the
bootstrap resamples fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.evaluation.regression_metrics import (  # noqa: E402
    aggregate_to_fields, bootstrap_field_ci, error_distribution,
    field_errors, regression_metrics)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def test_sum_aggregation_recovers_plot_total():
    """Yield is stored per pixel; a field's yield is the SUM over its pixels."""
    pid = np.array([1, 1, 1, 2, 2])
    true = np.array([2.0, 3.0, 5.0, 1.0, 1.0])
    pred = np.array([2.0, 2.0, 2.0, 4.0, 4.0])
    t, p = aggregate_to_fields(true, pred, pid, how="sum")
    assert t == {1.0: 10.0, 2.0: 2.0}
    assert p == {1.0: 6.0, 2.0: 8.0}


def test_mean_aggregation_for_dates():
    """Every pixel of a field carries the same date; the mean recovers it."""
    pid = np.array([7, 7, 7, 7])
    true = np.array([120.0, 120.0, 120.0, 120.0])
    pred = np.array([118.0, 122.0, 119.0, 121.0])
    t, p = aggregate_to_fields(true, pred, pid, how="mean")
    assert t[7.0] == pytest.approx(120.0)
    assert p[7.0] == pytest.approx(120.0)


def test_ignore_index_pixels_are_dropped_before_aggregating():
    """Adjacent-plot masking must not contribute to a field's total."""
    pid = np.array([1, 1, 1])
    true = np.array([5.0, 5.0, -999.0])
    pred = np.array([4.0, 4.0, 1000.0])
    t, p = aggregate_to_fields(true, pred, pid, ignore_index=-999, how="sum")
    assert t == {1.0: 10.0}
    assert p == {1.0: 8.0}        # the ignored pixel's 1000 must not appear


def test_background_plot_id_zero_is_excluded():
    pid = np.array([0, 0, 3])
    true = np.array([9.0, 9.0, 4.0])
    pred = np.array([1.0, 1.0, 4.0])
    t, _ = aggregate_to_fields(true, pred, pid, how="sum")
    assert set(t) == {3.0}


def test_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        aggregate_to_fields([1.0], [1.0], [1], how="median")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_perfect_prediction_is_zero_error():
    t = {1.0: 10.0, 2.0: 20.0}
    m = regression_metrics(t, dict(t))
    assert m.mae == 0.0 and m.rmse == 0.0 and m.mape == pytest.approx(0.0)


def test_mae_and_rmse_differ_when_one_field_is_badly_wrong():
    """The pair reveals whether error is spread or concentrated."""
    t = {1.0: 10.0, 2.0: 10.0, 3.0: 10.0, 4.0: 10.0}
    p = {1.0: 10.0, 2.0: 10.0, 3.0: 10.0, 4.0: 50.0}
    m = regression_metrics(t, p)
    assert m.mae == pytest.approx(10.0)
    assert m.rmse == pytest.approx(20.0)
    assert m.rmse > m.mae


def test_mape_exclusions_are_counted_not_hidden():
    """A zero-truth field cannot have a percentage error; say so."""
    t = {1.0: 100.0, 2.0: 0.0}
    p = {1.0: 110.0, 2.0: 5.0}
    m = regression_metrics(t, p)
    assert m.n_excluded_from_mape == 1
    assert m.n_fields == 2                 # still counted for MAE/RMSE
    assert m.mape == pytest.approx(10.0)   # from field 1 only


def test_mape_is_none_when_every_field_is_excluded():
    m = regression_metrics({1.0: 0.0}, {1.0: 3.0})
    assert m.mape is None
    assert m.n_excluded_from_mape == 1


def test_raises_when_no_fields_shared():
    with pytest.raises(ValueError):
        regression_metrics({1.0: 1.0}, {2.0: 1.0})


# --------------------------------------------------------------------------
# distribution -- the actual audit output
# --------------------------------------------------------------------------

def test_distribution_exposes_a_tail_the_mean_hides():
    """Two sets with the same mean error, very different per-field risk."""
    even = {float(i): 10.0 for i in range(10)}
    spiky = {float(i): 0.0 for i in range(9)}
    spiky[9.0] = 100.0
    de, ds = error_distribution(even), error_distribution(spiky)
    assert de["mean"] == pytest.approx(ds["mean"])
    assert de["median"] == 10.0 and ds["median"] == 0.0
    assert ds["max"] == 100.0 and de["max"] == 10.0


def test_share_above_threshold():
    errs = {float(i): float(i) for i in range(10)}   # 0..9
    d = error_distribution(errs, thresholds=[4.5])
    assert d["share_above"]["4.5"] == pytest.approx(0.5)


def test_field_errors_are_absolute():
    e = field_errors({1.0: 10.0}, {1.0: 4.0})
    assert e == {1.0: 6.0}


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    t = {float(i): 100.0 for i in range(40)}
    p = {float(i): 100.0 + rng.normal(0, 10) for i in range(40)}
    a = bootstrap_field_ci(t, p, metric="mae", n_boot=400, seed=7)
    b = bootstrap_field_ci(t, p, metric="mae", n_boot=400, seed=7)
    assert a == b
    point = regression_metrics(t, p).mae
    assert a["lo"] <= point <= a["hi"]
    assert a["n_fields"] == 40


def test_bootstrap_degrades_gracefully_on_one_field():
    r = bootstrap_field_ci({1.0: 5.0}, {1.0: 6.0}, n_boot=100)
    assert np.isnan(r["lo"]) and r["n_boot"] == 0


# --------------------------------------------------------------------------
# field-SEASON aggregation -- the unit the reproduction gate actually needs
# --------------------------------------------------------------------------

def test_field_season_keeps_seasons_separate():
    """One plot in two samples is TWO observations, not one pooled total.

    Pooling instead gave MAPE 0.2874 against a released 0.3274 and failed the
    gate; this unit gives 0.3269.
    """
    from src.evaluation.regression_metrics import aggregate_to_field_seasons
    pid = np.array([[[5, 5]], [[5, 5]]])          # same plot, two samples
    true = np.array([[[10.0, 10.0]], [[30.0, 30.0]]])
    pred = np.array([[[9.0, 9.0]], [[31.0, 31.0]]])
    t, p = aggregate_to_field_seasons(true, pred, pid, how="sum")
    assert set(t) == {(0, 5.0), (1, 5.0)}
    assert t[(0, 5.0)] == 20.0 and t[(1, 5.0)] == 60.0
    assert p[(0, 5.0)] == 18.0 and p[(1, 5.0)] == 62.0


def test_field_season_rejects_flattened_input():
    """A flattened array has already lost the season boundary -- fail loudly."""
    from src.evaluation.regression_metrics import aggregate_to_field_seasons
    with pytest.raises(ValueError, match="sample axis"):
        aggregate_to_field_seasons(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                   np.array([1, 1]))


def test_require_positive_skips_zero_truth_field_seasons():
    """Mirrors the reference: a field-season with no recorded yield is skipped."""
    from src.evaluation.regression_metrics import aggregate_to_field_seasons
    pid = np.array([[[1, 1, 2, 2]]])
    true = np.array([[[5.0, 5.0, 0.0, 0.0]]])
    pred = np.array([[[4.0, 4.0, 9.0, 9.0]]])
    t, _ = aggregate_to_field_seasons(true, pred, pid, how="sum",
                                      require_positive=True)
    assert set(t) == {(0, 1.0)}
    t2, _ = aggregate_to_field_seasons(true, pred, pid, how="sum",
                                       require_positive=False)
    assert set(t2) == {(0, 1.0), (0, 2.0)}


def test_field_season_drops_ignore_index_pixels():
    from src.evaluation.regression_metrics import aggregate_to_field_seasons
    pid = np.array([[[3, 3, 3]]])
    true = np.array([[[4.0, 4.0, -999.0]]])
    pred = np.array([[[3.0, 3.0, 500.0]]])
    t, p = aggregate_to_field_seasons(true, pred, pid, ignore_index=-999,
                                      how="sum")
    assert t[(0, 3.0)] == 8.0 and p[(0, 3.0)] == 6.0
