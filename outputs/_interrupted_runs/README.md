# Interrupted training runs, kept as evidence

These seed-3 runs of the crop-type ladder were killed by an out-of-memory
condition on 2026-10-11, caused by running extra ladder seeds alongside the
data-regime sweep and twelve CPU plot-level evaluations on one 32 GB
machine. They are kept rather than deleted because one of them documents a
real hazard.

`v1_seed3` ran 2 of 100 planned epochs and scored macro IoU 0.415, beside
three completed seeds scoring about 0.63. It nevertheless left a complete
looking `best.json` and `best_per_plot_iou.json` on disk, because the
trainer rewrites both whenever the validation score improves. Any script
that selects seeds by globbing `v1_seed*` would have averaged it in and
moved the rung by roughly 0.07 macro IoU with nothing in the output saying
so.

An earlier version of this file said `scripts/aggregate_ladder.py` already
read the trainer's `complete` flag and excluded it. **That was wrong**: it
labelled the run in a status column and still averaged it in. Eight
consumers in all failed to honour the flag, the aggregator among them. That is now fixed in one place,
`src.utils.completed_seed_runs`, and pinned by
`tests/test_completeness_filter.py`.

Nothing in this directory is used by any reported result.
