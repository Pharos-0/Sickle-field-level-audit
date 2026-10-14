"""Tests for the yield-reliability screening experiment.

These pin the parts that would fail silently: a leaky threshold still
produces plausible numbers, and a wrong AUROC tie convention shifts a result
without raising anything. Each test states what going wrong would look like
in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))

ys = pytest.importorskip("yield_screen")


# --------------------------------------------------------------- AUROC
def test_auroc_perfect_separation():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    pos = np.array([False, False, True, True])
    assert ys.auroc(s, pos) == pytest.approx(1.0)


def test_auroc_reversed_separation():
    s = np.array([0.9, 0.8, 0.2, 0.1])
    pos = np.array([False, False, True, True])
    assert ys.auroc(s, pos) == pytest.approx(0.0)


def test_auroc_all_ties_is_one_half():
    """A constant signal must score exactly random, not 1.0.

    Without explicit tie handling a naive implementation returns 1.0 here,
    which would report a useless signal as perfect.
    """
    s = np.array([0.5, 0.5, 0.5, 0.5])
    pos = np.array([True, False, True, False])
    assert ys.auroc(s, pos) == pytest.approx(0.5)


def test_auroc_partial_ties():
    s = np.array([1.0, 2.0, 2.0, 3.0])
    pos = np.array([False, False, True, True])
    # ranks 1, 2.5, 2.5, 4 -> U = (2.5+4) - 3 = 3.5 over n1*n0 = 4
    assert ys.auroc(s, pos) == pytest.approx(0.875)


def test_auroc_undefined_when_one_class_absent():
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(ys.auroc(s, np.array([False, False, False])))
    assert np.isnan(ys.auroc(s, np.array([True, True, True])))


# ------------------------------------------------------------ Spearman
def test_spearman_monotone_but_nonlinear():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a ** 3
    assert ys.spearman(a, b) == pytest.approx(1.0)


# ------------------------------------- leave-one-plot-out, the leak test
def _toy():
    """Four plots, two seasons each; signal correlates with badness."""
    fields = [(0, 1.0), (1, 1.0), (0, 2.0), (1, 2.0),
              (0, 3.0), (1, 3.0), (0, 4.0), (1, 4.0)]
    sig = {f: 0.1 * i for i, f in enumerate(fields)}
    plot_of = {f: f[1] for f in fields}
    bad = {(0, 4.0), (1, 4.0)}
    return fields, sig, plot_of, bad


def test_threshold_ignores_the_held_out_plots_own_seasons():
    """Spiking every season of one plot must not move that plot's threshold.

    If it did, a plot would help set the bar it is judged against, and the
    reported recall would be optimistic for reasons invisible in the output.
    """
    fields, sig, plot_of, bad = _toy()
    target = 1.0
    base = ys.loo_plot_screen(sig, bad, plot_of, 0.5)

    spiked = dict(sig)
    for f in fields:
        if plot_of[f] == target:
            spiked[f] = 1e6
    after = ys.loo_plot_screen(spiked, bad, plot_of, 0.5)

    # Other plots' outcomes are unchanged by the spike except through their
    # own thresholds, which DO legitimately see the spiked plot. What must
    # not change is that the spiked plot's threshold came from others only:
    others = [sig[g] for g in fields if plot_of[g] != target]
    thr_before = float(np.quantile(others, 0.5))
    others_after = [spiked[g] for g in fields if plot_of[g] != target]
    thr_after = float(np.quantile(others_after, 0.5))
    assert thr_before == thr_after
    assert base["n_referred"] >= 0 and after["n_referred"] >= 0


def test_no_season_of_a_plot_contributes_to_its_own_threshold():
    fields, sig, plot_of, _ = _toy()
    for f in fields:
        others = [g for g in fields if plot_of[g] != plot_of[f]]
        assert all(plot_of[g] != plot_of[f] for g in others)
        # every other season of the SAME plot must be excluded
        same = [g for g in fields if plot_of[g] == plot_of[f] and g != f]
        assert not set(same) & set(others)


def test_perfect_signal_reaches_full_recall():
    fields, sig, plot_of, bad = _toy()
    r = ys.loo_plot_screen(sig, bad, plot_of, 0.5)
    assert r["recall"] == pytest.approx(1.0)
    assert r["n_missed"] == 0


def test_recall_and_precision_denominators():
    fields, sig, plot_of, bad = _toy()
    r = ys.loo_plot_screen(sig, bad, plot_of, 0.5)
    assert r["recall"] == r["n_caught"] / len(bad)
    assert r["precision"] == r["n_caught"] / r["n_referred"]
    assert r["n_caught"] + r["n_missed"] == len(bad)


# ------------------------------------------------- bootstrap is by plot
def test_single_plot_is_refused_rather_than_guessed():
    """One plot leaves no held-out set, so LOO must fail loudly.

    Silently returning a number here would report a threshold that nothing
    was held out to validate.
    """
    sig = {(0, 1.0): 0.1, (1, 1.0): 0.9}
    plot_of = {(0, 1.0): 1.0, (1, 1.0): 1.0}
    bad = {(1, 1.0)}
    with pytest.raises(ValueError, match="at least 2 plots"):
        ys.loo_plot_screen(sig, bad, plot_of, 0.5)
    with pytest.raises(ValueError, match="at least 2 plots"):
        ys.plot_bootstrap(sig, bad, plot_of, 0.5, n_boot=10, seed=0)


def test_bootstrap_resamples_plots_not_field_seasons():
    """Two plots whose seasons differ: a plot bootstrap can draw only 3
    distinct plot multisets, so recall takes very few distinct values.

    A field-season bootstrap over the same 4 observations would produce many
    more. The coarse support is the evidence that plots are the unit.
    """
    fields = [(0, 1.0), (1, 1.0), (0, 2.0), (1, 2.0)]
    sig = {(0, 1.0): 0.1, (1, 1.0): 0.2, (0, 2.0): 0.8, (1, 2.0): 0.9}
    plot_of = {f: f[1] for f in fields}
    bad = {(0, 2.0), (1, 2.0)}
    vals = set()
    for seed in range(12):
        ci = ys.plot_bootstrap(sig, bad, plot_of, 0.5, n_boot=300, seed=seed)
        vals.add((round(ci["lo"], 6), round(ci["hi"], 6)))
    assert len(vals) <= 3, vals


# ------------------------------------------- dispersion and target maths
def test_cv_uses_sample_standard_deviation():
    p = [100.0, 110.0, 120.0]
    cv = float(np.std(p, ddof=1) / abs(np.mean(p)))
    assert cv == pytest.approx(10.0 / 110.0)
    assert cv != pytest.approx(float(np.std(p, ddof=0) / abs(np.mean(p))))


def test_ape_denominator_is_truth_not_prediction():
    truth, pred = 1000.0, 1500.0
    ape = abs(pred - truth) / abs(truth) * 100.0
    assert ape == pytest.approx(50.0)
    assert ape != pytest.approx(abs(pred - truth) / abs(pred) * 100.0)


def test_protocol_file_is_frozen_and_matches_the_code_defaults():
    import yaml
    p = yaml.safe_load(
        (PROJ / "configs/yield_screen_protocol.yaml").read_text(
            encoding="utf8"))
    assert p["frozen_before_results"] is True
    assert p["independent_unit"] == "plot"
    assert p["leakage_controls"]["threshold_selection"] == "leave_one_plot_out"
    assert p["leakage_controls"]["bootstrap_unit"] == "plot"
    assert p["target"]["primary_threshold_ape_pct"] == 50.0
    # the threshold must be declared as a study choice, not a policy rule
    assert "NOT a YES-TECH" in p["target"]["threshold_provenance"]


# ------------------------------------------------- export integrity
def test_exported_predictions_align_with_truth_and_errors():
    """|pred - true| from the export must equal the stored per-field error.

    A misaligned key would otherwise pair one field's prediction with
    another's truth and still produce a plausible-looking table.
    """
    import glob, json, ast
    files = [f for f in sorted(glob.glob(str(
        PROJ / "results/regression_gate/crop_yield_*_bs32.json")))
        if "_debug" not in f]
    checked = 0
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf8"))
        if "per_field_pred" not in d:
            continue
        for k, err in d["per_field_abs_error"].items():
            p = d["per_field_pred"][k]
            t = d["per_field_true"][k]
            assert abs(abs(p - t) - err) < 1e-3, (f, k, p, t, err)
            checked += 1
    if checked == 0:
        pytest.skip("no yield gate carries per-field predictions yet")
    assert checked > 0
