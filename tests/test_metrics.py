"""Unit tests for the metric implementation.

These pin behaviour that the study depends on and that is easy to get subtly
wrong: ignore_index handling, undefined-class exclusion, plot-level pooling,
and the paired-delta verdict wording.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import (  # noqa: E402
    NON_PADDY, PADDY, bootstrap_plot_ci, confusion, crop_type_metrics,
    iou_from_confusion, paired_delta, plot_macro_iou, prf_from_confusion,
)

IGN = -999


# ---------------------------------------------------------------- confusion

def test_perfect_prediction_gives_iou_one():
    y = np.array([0, 0, 1, 1])
    m = crop_type_metrics(y, y)
    assert m.macro_iou == pytest.approx(1.0)
    assert m.paddy_iou == pytest.approx(1.0)
    assert m.non_paddy_iou == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)


def test_completely_wrong_prediction_gives_iou_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([1, 1, 0, 0])
    m = crop_type_metrics(y, p)
    assert m.macro_iou == pytest.approx(0.0)
    # a present-but-never-correct class scores F1 = 0, not NaN
    assert m.macro_f1 == pytest.approx(0.0)


def test_ignore_index_pixels_are_excluded():
    y = np.array([0, 1, IGN, IGN])
    p = np.array([0, 1, 1, 0])          # wrong on the ignored pixels
    m = crop_type_metrics(y, p)
    assert m.macro_iou == pytest.approx(1.0)
    assert m.n_pixels == 2


def test_confusion_orientation_rows_are_truth():
    y = np.array([0, 0, 0, 1])
    p = np.array([0, 1, 1, 1])
    cm = confusion(y, p)
    assert cm[0, 0] == 1 and cm[0, 1] == 2    # true 0 predicted 1 twice
    assert cm[1, 1] == 1


def test_known_iou_value():
    # 2 TP paddy, 1 FP paddy, 1 FN paddy -> IoU = 2 / (2+1+1) = 0.5
    y = np.array([0, 0, 0, 1])
    p = np.array([0, 0, 1, 0])
    cm = confusion(y, p)
    iou = iou_from_confusion(cm)
    assert iou[PADDY] == pytest.approx(2 / 4)


def test_absent_class_iou_is_nan_not_zero():
    y = np.array([0, 0, 0])
    p = np.array([0, 0, 0])
    iou = iou_from_confusion(confusion(y, p))
    assert iou[PADDY] == pytest.approx(1.0)
    assert np.isnan(iou[NON_PADDY]), "absent class must be NaN, never 0"


def test_macro_iou_ignores_nan_classes():
    y = np.array([0, 0, 0])
    p = np.array([0, 0, 0])
    m = crop_type_metrics(y, p)
    assert m.macro_iou == pytest.approx(1.0)   # not 0.5


def test_label_outside_range_raises():
    with pytest.raises(ValueError):
        confusion(np.array([0, 5]), np.array([0, 1]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion(np.array([0, 1]), np.array([0, 1, 1]))


# ------------------------------------------------------------- plot metrics

def test_plot_macro_weights_plots_equally_not_pixels():
    # plot 1: 100 px, both classes, perfect.  plot 2: 4 px, both classes, all wrong.
    t = np.array([0] * 50 + [1] * 50 + [0, 0, 1, 1])
    p = np.array([0] * 50 + [1] * 50 + [1, 1, 0, 0])
    g = np.array([1] * 100 + [2, 2, 2, 2])

    pooled = crop_type_metrics(t, p)
    s = plot_macro_iou(t, p, g)

    # pixel-pooled is dominated by the 100-pixel plot
    assert pooled.macro_iou == pytest.approx(50 / 54)
    assert pooled.macro_iou > 0.9
    # plot-macro gives the 4-pixel plot equal weight: (1.0 + 0.0) / 2
    assert s["plot_macro_iou"] == pytest.approx(0.5)
    assert s["n_plots"] == 2


def test_plot_macro_pools_pixels_within_a_plot():
    # one plot, two "samples" worth of pixels pooled into one confusion matrix
    t = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    g = np.array([7, 7, 7, 7])
    s = plot_macro_iou(t, p, g)
    assert s["n_plots"] == 1
    assert s["plot_macro_iou"] == pytest.approx(1.0)


def test_plot_macro_excludes_undefined_classes():
    # plot has only paddy pixels -> non-paddy IoU undefined there
    t = np.array([0, 0, 0])
    p = np.array([0, 0, 0])
    g = np.array([3, 3, 3])
    s = plot_macro_iou(t, p, g)
    assert s["plot_macro_iou"] == pytest.approx(1.0)
    assert s["n_undefined_class_instances"] == 1


def test_plot_macro_respects_ignore_index():
    t = np.array([0, 0, IGN, IGN])
    p = np.array([0, 0, 1, 1])
    g = np.array([1, 1, 2, 2])
    s = plot_macro_iou(t, p, g)
    # plot 2 is entirely ignore_index, so it contributes nothing
    assert s["n_plots"] == 1


def test_bootstrap_resamples_plots():
    per_plot = {float(i): v for i, v in enumerate([0.1, 0.5, 0.9, 0.4, 0.6])}
    ci = bootstrap_plot_ci(per_plot, n_boot=500, seed=0)
    assert ci["n_plots"] == 5
    assert ci["lo"] <= np.mean(list(per_plot.values())) <= ci["hi"]


def test_bootstrap_is_deterministic_for_a_seed():
    pp = {float(i): v for i, v in enumerate([0.2, 0.4, 0.6, 0.8])}
    a = bootstrap_plot_ci(pp, n_boot=200, seed=7)
    b = bootstrap_plot_ci(pp, n_boot=200, seed=7)
    assert a == b


# ------------------------------------------------------------ paired deltas

def test_paired_delta_uses_shared_seeds_only():
    base = {0: 0.70, 1: 0.72, 2: 0.74, 9: 0.99}
    var = {0: 0.75, 1: 0.76, 2: 0.80}
    d = paired_delta(base, var)
    assert d["seeds"] == [0, 1, 2]
    assert d["n"] == 3
    assert d["mean_delta"] == pytest.approx(np.mean([0.05, 0.04, 0.06]))


def test_paired_delta_flags_inconclusive_when_mean_below_sd():
    base = {0: 0.70, 1: 0.70, 2: 0.70}
    var = {0: 0.80, 1: 0.60, 2: 0.71}      # mean ~0.0033, sd ~0.10
    d = paired_delta(base, var)
    assert abs(d["mean_delta"]) < d["sd_delta"]
    assert d["verdict"].startswith("inconclusive")


def test_paired_delta_reports_clear_effect():
    base = {0: 0.70, 1: 0.71, 2: 0.69}
    var = {0: 0.80, 1: 0.81, 2: 0.79}
    d = paired_delta(base, var)
    assert d["mean_delta"] == pytest.approx(0.10, abs=1e-9)
    assert d["verdict"] == "mean exceeds seed-to-seed sd"


def test_paired_delta_requires_shared_seeds():
    with pytest.raises(ValueError):
        paired_delta({0: 0.1}, {1: 0.2})


# ------------------------------------------------------------- degeneracy

def test_single_class_predictor_is_exposed_by_iou_not_accuracy():
    """The [S2]_utae failure mode: predict non-paddy everywhere."""
    y = np.array([PADDY] * 20 + [NON_PADDY] * 80)
    p = np.full(100, NON_PADDY)
    m = crop_type_metrics(y, p)
    assert m.paddy_iou == pytest.approx(0.0)      # collapse is visible
    assert m.non_paddy_iou == pytest.approx(0.8)
    assert m.macro_iou == pytest.approx(0.4)
    assert m.paddy_recall == pytest.approx(0.0)
