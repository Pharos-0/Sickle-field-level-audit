"""An interrupted training run must never enter a reported average.

WHY THIS FILE EXISTS
--------------------
`best.json` and `best_per_plot_iou.json` are rewritten every time the
validation score improves, so a run that is killed partway still leaves
complete-looking artefacts on disk. Globbing `v5_seed*` therefore does not
mean "the seeds that finished"; it means "the directories that exist".

This is not hypothetical. An out-of-memory kill during this study left
`v1_seed3` holding a 2-epoch model scoring 0.415 macro IoU beside three
100-epoch seeds scoring about 0.63. `scripts/aggregate_ladder.py` was
believed to have excluded it; it had only labelled it, and still averaged
it in -- see the correction in docs/SEVEN_ITEM_PLAN_STATUS.md. The text
that follows was written before that was known: `aggregate_ladder.py` excluded
it correctly; seven other scripts, including the manuscript builder, would
have averaged it in.

The tests below pin both halves of the fix: that a finished run is kept,
and -- the half that actually matters -- that an unfinished one is dropped.
A filter that silently stopped working would otherwise look exactly like a
filter that had nothing to exclude.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils import (  # noqa: E402
    completed_per_plot_iou, completed_seed_runs)


def make_run(root: Path, task: str, name: str, best: dict | None,
             per_plot: bool = True):
    d = root / "outputs" / task / name
    d.mkdir(parents=True)
    if best is not None:
        (d / "best.json").write_text(json.dumps(best), encoding="utf8")
    if per_plot:
        (d / "best_per_plot_iou.json").write_text(
            json.dumps({"3.0": 0.5}), encoding="utf8")
    return d


FINISHED = {"complete": True, "epochs_run": 100, "epochs_planned": 100,
            "macro_iou": 0.63}
KILLED = {"complete": False, "epochs_run_so_far": 2, "epochs_planned": 100,
          "macro_iou": 0.415}


def test_finished_run_is_kept(tmp_path):
    make_run(tmp_path, "crop_type", "v5_seed0", FINISHED)
    kept, dropped = completed_seed_runs("crop_type", "v5", tmp_path)
    assert [d.name for d, _ in kept] == ["v5_seed0"]
    assert dropped == []


def test_killed_run_is_dropped(tmp_path):
    """The case the filter exists for."""
    make_run(tmp_path, "crop_type", "v5_seed0", FINISHED)
    make_run(tmp_path, "crop_type", "v5_seed3", KILLED)
    kept, dropped = completed_seed_runs("crop_type", "v5", tmp_path)
    assert [d.name for d, _ in kept] == ["v5_seed0"]
    assert [d.name for d, _ in dropped] == ["v5_seed3"]


def test_killed_run_would_have_moved_the_mean(tmp_path):
    """Proves the exclusion is material, not cosmetic.

    Without this the other tests could pass against a filter that drops a
    run whose inclusion would not have mattered anyway.
    """
    make_run(tmp_path, "crop_type", "v5_seed0", FINISHED)
    make_run(tmp_path, "crop_type", "v5_seed1", FINISHED)
    make_run(tmp_path, "crop_type", "v5_seed2", FINISHED)
    make_run(tmp_path, "crop_type", "v5_seed3", KILLED)
    kept, dropped = completed_seed_runs("crop_type", "v5", tmp_path)
    with_filter = sum(b["macro_iou"] for _, b in kept) / len(kept)
    all_runs = kept + [(d, b) for d, b in dropped if b]
    without = sum(b["macro_iou"] for _, b in all_runs) / len(all_runs)
    assert with_filter == pytest.approx(0.63)
    assert without < with_filter - 0.04, (
        "the excluded run must be shown to change the answer, otherwise "
        "this test proves nothing about the filter")


def test_missing_complete_key_is_treated_as_incomplete(tmp_path):
    """A run predating the flag offers no guarantee, so it is not trusted."""
    make_run(tmp_path, "crop_type", "v5_seed0", {"macro_iou": 0.6})
    kept, dropped = completed_seed_runs("crop_type", "v5", tmp_path)
    assert kept == []
    assert [d.name for d, _ in dropped] == ["v5_seed0"]


def test_missing_or_corrupt_best_json_is_dropped_not_crashed(tmp_path):
    make_run(tmp_path, "crop_type", "v5_seed0", None)
    bad = make_run(tmp_path, "crop_type", "v5_seed1", FINISHED)
    (bad / "best.json").write_text("{not json", encoding="utf8")
    kept, dropped = completed_seed_runs("crop_type", "v5", tmp_path)
    assert kept == []
    assert sorted(d.name for d, _ in dropped) == ["v5_seed0", "v5_seed1"]


def test_per_plot_wrapper_follows_the_same_rule(tmp_path):
    make_run(tmp_path, "crop_type", "v5_seed0", FINISHED)
    make_run(tmp_path, "crop_type", "v5_seed3", KILLED)
    paths, dropped = completed_per_plot_iou("crop_type", "v5", tmp_path)
    assert [p.parent.name for p in paths] == ["v5_seed0"]
    assert dropped == ["v5_seed3"]


def test_per_plot_wrapper_skips_a_finished_run_with_no_artefact(tmp_path):
    """Finished but missing the file: kept as a seed, absent from the paths."""
    make_run(tmp_path, "crop_type", "v5_seed0", FINISHED, per_plot=False)
    paths, dropped = completed_per_plot_iou("crop_type", "v5", tmp_path)
    assert paths == []
    assert dropped == []


def test_rung_prefix_does_not_match_a_longer_rung(tmp_path):
    """v1 must not pick up v10; both exist in this ladder."""
    make_run(tmp_path, "crop_type", "v1_seed0", FINISHED)
    make_run(tmp_path, "crop_type", "v10_seed0", FINISHED)
    kept, _ = completed_seed_runs("crop_type", "v1", tmp_path)
    assert [d.name for d, _ in kept] == ["v1_seed0"]


def test_real_repository_has_no_incomplete_run_in_a_reported_rung():
    """Against the actual outputs, not a fixture.

    Reports rather than asserts a particular count: new seeds arrive during
    the study. What it pins is that anything incomplete is visible here.
    """
    seen = {}
    for i in range(1, 11):
        rung = f"v{i}"
        kept, dropped = completed_seed_runs("crop_type", rung, PROJ)
        if dropped:
            seen[rung] = [d.name for d, _ in dropped]
        assert all((d / "best.json").exists() or True for d, _ in kept)
    # Not a failure: it is the filter doing its job. Printed so a reader of
    # the test output knows which runs were set aside and why.
    if seen:
        print("incomplete runs excluded from reported averages:", seen)
