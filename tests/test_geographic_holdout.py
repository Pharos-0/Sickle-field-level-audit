"""Pin the no-leakage property of the geographic holdout.

A reviewer asked specifically whether the held-out region's own fields
influence the threshold that judges them. These tests make the answer
checkable instead of a matter of reading the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.evaluation.regression_metrics import geographic_holdout  # noqa: E402


def test_held_out_region_cannot_move_its_own_threshold():
    """The decisive test: change ONLY the held-out region's signal values to
    something extreme. If they fed the threshold, the threshold would move."""
    region = {1.0: "A", 2.0: "A", 3.0: "B", 4.0: "B", 5.0: "B", 6.0: "B"}
    bad = {1.0, 3.0}
    base = {1.0: 0.9, 2.0: 0.1, 3.0: 0.8, 4.0: 0.2, 5.0: 0.3, 6.0: 0.4}
    spiked = dict(base); spiked[1.0] = 99.0; spiked[2.0] = -99.0  # region A only

    a = {r["region"]: r for r in geographic_holdout(base, bad, region)}
    b = {r["region"]: r for r in geographic_holdout(spiked, bad, region)}
    assert a["A"]["threshold"] == b["A"]["threshold"], \
        "region A's own values changed the threshold used to judge region A"


def test_threshold_fields_exclude_the_held_out_region():
    region = {float(i): ("A" if i < 3 else "B") for i in range(6)}
    out = {r["region"]: r for r in geographic_holdout(
        {float(i): i / 10 for i in range(6)}, set(), region)}
    assert out["A"]["n_threshold_fields"] == 3   # the three B fields
    assert out["B"]["n_threshold_fields"] == 3   # the three A fields


def test_recall_is_none_not_zero_when_region_has_no_bad_field():
    """A recall over zero cases is undefined, not 0%. Reporting 0 would
    make a region with nothing to find look like a failure."""
    # enough fields in each region that neither is skipped by the
    # len(rest) < 2 guard, which needs at least two fields for a quantile
    region = {1.0: "A", 2.0: "A", 3.0: "B", 4.0: "B", 5.0: "B"}
    out = {r["region"]: r for r in geographic_holdout(
        {1.0: 0.5, 2.0: 0.45, 3.0: 0.4, 4.0: 0.6, 5.0: 0.7}, {3.0}, region)}
    assert out["A"]["recall"] is None
    assert out["B"]["recall"] is not None


def test_unlabelled_regions_are_counted_and_never_held_out():
    region = {1.0: "A", 2.0: "A", 3.0: None, 4.0: float("nan"), 5.0: "B",
              6.0: "B"}
    out = geographic_holdout({float(i): i / 10 for i in range(1, 7)},
                             {1.0}, region)
    assert {r["region"] for r in out} == {"A", "B"}
    assert all(r["n_unlabelled_in_rest"] == 2 for r in out)


@pytest.mark.parametrize("workload", [0.25, 0.5, 0.75])
def test_flagged_count_tracks_workload(workload):
    region = {float(i): ("A" if i < 4 else "B") for i in range(12)}
    sig = {float(i): i / 12 for i in range(12)}
    out = {r["region"]: r for r in geographic_holdout(sig, set(), region,
                                                      workload)}
    assert out["B"]["n_flagged"] <= out["B"]["n_fields"]
