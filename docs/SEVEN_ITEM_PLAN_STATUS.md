# Seven-item plan — what was done, 2026-10-11

Against the plan for raising the audit paper's research quality. Each item
says what ran, what it produced, and what is still open. Nothing below is a
statement of intent: where something did not run, it says so.

**Bottom line.** Items 1, 2, 3, 5, 6 and 7 are done. Item 4 is now
running: on your instruction the data-regime sweep is paused
(`docs/SWEEP_PAUSED.md`) and the GPU is training v3 and v4 at seeds 3 to 7.
The analysis below is what decided those two rungs, and it also says the
extra-seed job I had queued earlier would have been spent on the wrong
ones.

---

## The incident, stated first because it cost the user time

I launched twelve extra ladder-seed trainings and, later, twelve CPU
plot-level evaluations, while the data-regime sweep was already running on
a 32 GB machine. Free memory fell to 2.1 GB and the following broke:

| what | state |
|---|---|
| `plots_200_seed002/v2_radar_seed2` | truncated at 2 of 100 epochs |
| `plots_250_seed002/v1_seed2` | truncated at 4 of 100 epochs |
| `plots_315_seed002/v1_seed2` | truncated at 2 of 100 epochs |
| `plots_315_seed002/v2_seed2` | deadlocked; the sweep made no progress for 35 minutes |
| my own `v1_seed3`, `v2_seed3`, `v4_seed3`, `v5_seed3` | killed |
| one plot-level evaluation (`[S2,L8,S1]_utae`) | out-of-memory, re-run successfully afterwards |

Repair: the extra-seeds job was stopped, the deadlocked sweep was
restarted, and it is now running alone at 76 % GPU. `run_sweep.py` skips
only cells marked `complete: true`, so the three truncated cells are being
re-trained from scratch automatically; `src/train.py` has no resume logic,
so they restart at epoch 1 rather than continuing from a partial state.
**Nothing in the sweep's results is corrupted — the truncated cells are
being redone.** The cost is wall-clock time, roughly an hour.

My four killed runs are in `outputs/_interrupted_runs/` with a README
rather than deleted, because one of them exposed a real defect.

### The defect it exposed

`best.json` and `best_per_plot_iou.json` are rewritten every time the
validation score improves, so a killed run leaves complete-looking
artefacts behind. `v1_seed3` ran 2 of 100 epochs, scored 0.415 macro IoU
and left both files on disk, beside three finished seeds scoring about
0.63.

**Correction, 2026-10-12.** An earlier version of this paragraph said
`scripts/aggregate_ladder.py` already read the trainer's `complete` flag
and excluded such runs. That was wrong. It *labelled* an unfinished run in
its status column and still averaged it into the rung's mean, SD and seed
count. It did not affect the published three-seed table, where every rung
is `ok` at three finished seeds, but it did affect an interim rebuild while
v3 was being re-run: a seed at 19 of 100 epochs sat inside the v3 mean and
the manuscript rendered "seven seeds at v3" when six had finished. Fixed in
the aggregator itself, and `scripts/verify_all.py` now checks that every
rung's seed count in the table equals its finished-run count.

So **eight scripts**, not seven, selected or averaged seeds without
honouring the flag, including the manuscript builder and both figure
builders, which globbed `v5_seed*`. Averaging that run in would have moved the rung by roughly 0.07
macro IoU with nothing in any output saying so.

Fixed in one place, `src.utils.completed_seed_runs`, used by every
consumer, and pinned by `tests/test_completeness_filter.py` — including a
test that the excluded run would have changed the answer, so the filter
cannot pass by having nothing to exclude.

---

## 1. Freeze the research question

The abstract opened with ninety words of YES-TECH policy before reaching
the benchmark. It now opens with the question the paper answers — what a
pooled benchmark number says about an individual field — and keeps
YES-TECH as two sentences of motivation, ending with an explicit statement
that no claim is made about any deployed system. Section VII already
carried the same limit and is unchanged.

## 2. Independently verify every headline result

`scripts/minimal_repro.py`. One short block per defect, each printing the
published value beside the measured value, the sentence connecting them,
the checkpoint identifiers, the artefact read, and the library versions.
All four reproduce.

| | as published | as measured |
|---|---|---|
| D1 batch size | macro IoU 0.8011 for `[S1]_unet3d` | 0.4616 to 0.7888 across batch sizes 1 to 64, spread 32.7 IoU points, same frozen weights |
| D3 degenerate baselines | reported as crop-type baselines with a macro IoU | `[L8]_utae` and `[S2]_utae`: macro IoU 0.3812, paddy IoU 0.0000, non-paddy recall 1.000 |
| D4 phenology MAPE | `val_mape` 0.093895, reads as 9.39 % | `val_mae` / 183 = 0.093895, matching for 24 of 24 runs, worst disagreement 1.05e-08 |
| yield vs constant | 24 checkpoints with MAE, RMSE, MAPE and no constant baseline | constant 1841.9 kg/acre scores MAE 435.9; best checkpoint 497.0; 24 of 24 worse on point estimates (2026-10-13: no plot-clustered interval excludes zero, so the claim is "none shown to beat a constant"; `results/yield_vs_constant.json`) |

Every block reads a cached evaluation rather than re-running a model, and
the output says so: it shows the conclusions follow from the recorded
evaluations, and is not a substitute for re-running them.

**This also found an error in the manuscript.** Section V-A quoted three
batch-size values and "a swing of 6.5 IoU points" as literals. The saved
cells give 0.0662, which is 6.6, and the checkpoint quoted is the mildest
of the four evaluated at more than one batch size — another moves 32.7
points. Both the abstract and Section V-A now compute these from the
artefacts and report the range, so the paper no longer understates its own
finding by a factor of five.

## 3. Strengthen the field-level statistics

`scripts/field_size.py` and `scripts/field_distributions.py`.

> **Second correction, 2026-10-12.** The figures in this section used IoU
> averaged over both classes, which caps any imperfect single-crop field at
> 0.5 (35 of the 36 validation fields hold one crop). Scored on the crop each
> field holds, the quartiles are 0.861 / 0.957 / 1.000 and 6 %, 11 % and 17 %
> of fields fall below 0.30, 0.50 and 0.70. The current figures are in
> `results/field_distributions.json`; this record is left as written.

The paper contrasted a pooled 0.8665 against a per-field median of 0.672.
That is one aggregate against another. It now reports the distribution:
quartiles 0.469 / 0.672 / 1.000, range 0.000 to 1.000, and the fraction
below each of three stated thresholds with intervals — 11 % below 0.30
(95 % CI 3–22), 33 % below 0.50 (19–50), 53 % below 0.70 (36–69). None of
the three is an official criterion and the text says so.

The mechanism is now exact rather than asserted. The pooled confusion
matrix is the sum of the per-field matrices, so removing a field and
recomputing is arithmetic. `scripts/field_size.py` reconstructs each
field's weight from the masks alone and **cross-checks the total against
the released evaluations: 9,860 scored pixels, an exact match**. One field
holds 47.3 % of them; the nine largest hold 82.9 %; an equally weighted
field would hold 2.8 %.

Removing that single field from the twelve released crop-type checkpoints
moves the pooled macro IoU by a median of 0.066 and **by as much as 0.261**
(`[S2]_convlstm`: 0.4190 to 0.6798). A benchmark number one field of 36 can
move that far is not a field-level statement.

Two wrong readings are ruled out in the text:

- *Large fields are easier.* They are not. Spearman between field size and
  per-field IoU is −0.043, 95 % CI −0.384 to +0.294. The pooled figure is
  dominated by a few fields in **weight**, not because those fields score
  better.
- *The gap is a fixed optimism that could be subtracted.* It is not. Across
  the released checkpoints it ranges from −0.1962 to +0.3047, and
  `[S2]_convlstm` scores **higher** per-field than pooled.

For the regression tasks, the per-field-season error distribution with
plot-grouped intervals (41 field-seasons at only 14 plots, so field-seasons
are not independent): yield median 341.5 kg/acre, 95 % CI 215.5–706.6, p90
1232.2, and the worst tenth of field-seasons carries 28 % of the summed
error against 10 % for an even spread; the worst quarter carries 55 %.

## 4. More ablation seeds — RUNNING, on the rungs the analysis identifies

The sweep is paused on your instruction and the GPU is on **v3 and v4 at
seeds 3, 4, 5, 6 and 7**, ten runs, seed-major so that an interruption
leaves complete pairs. Log: `logs_ladder_seeds.log`. The sweep's state at
the pause and the single command to resume it are in
`docs/SWEEP_PAUSED.md`; nothing of the sweep was lost.

Measured pace: 87 s per epoch, so about 2.4 h per 100-epoch run and
roughly **24 h for all ten**, finishing around Monday evening.

The analysis below is what chose those two rungs, and it is also the part
of item 4 that stands on its own regardless of how the seeds come out.

### It finishes without anyone present

`scripts/after_seeds.sh` is watching, and when the seeds land it will
aggregate the ladder, re-run the power analysis, record what drifted, adopt
the new seed-dependent values, rebuild and re-check everything, and resume
the sweep. Log: `logs_after_seeds.log`.

Three things about it are deliberate:

- **It does not wait for ever.** A plain wait-for-the-marker loop would
  hang silently if the training job died, which is exactly how this morning
  would have ended. It gives up after 90 minutes with no training process
  and no log growth, says which happened, and carries on with whatever
  seeds did finish — the completeness filter keeps the ladder correct, and
  the manuscript now renders its seed counts from the CSV, so the paper
  will state the real number rather than an intended one.
- **Adopting a value is not agreeing with it.** Seed-dependent headline
  numbers will move, and `drift_check.py` is built to stop the build when
  they do. The script records the before state to
  `results/drift_check_before_new_seeds.json`, adopts the new values so the
  rebuild can proceed, and writes an amendment into
  `configs/tolerances.yaml` marked `reviewed: false` with a paragraph
  saying a person still has to confirm each change. That flag is the thing
  to look for when you come back.
- **It runs from a copy.** Editing a bash script while bash is executing it
  can resume at the wrong byte offset. Both the seed loop and the watcher
  run from copies in the scratchpad, so `scripts/` can be edited freely
  while they work.

### Preparing for a ladder that is no longer uniform

Two rungs will carry eight seeds and the rest three. Nine places in the
manuscript said "three seeds" as a literal, and `verify_all.py` asserted
three seeds per rung. An unattended rebuild would have produced a paper
stating a seed count it did not have. All nine now render from
`results/ladder_crop_type.csv`, and when the ladder stops being uniform the
text says so and explains why those rungs have more. The check now requires
at least three per rung and prints the actual counts.

`scripts/ladder_power.py` answers what more seeds would buy. Seeds are
**paired** across rungs — seed k of one rung and seed k of the next share
an initialisation and a data order — so each step is a paired difference,
which removes the shared seed effect and is materially more powerful than
comparing two means.

| | steps |
|---|---|
| resolved at 3 seeds (paired interval excludes zero) | v1→v2, v4→v5, v6→v7 |
| small (interval lies within ±σmax, a large effect is ruled out) | v5→v6, v7→v8, v8→v9, v9→v10 |
| underpowered (interval still admits an effect larger than σmax) | v2→v3, v3→v4 |

Two findings follow, and both are now in Section V-B:

1. **σmax classifies v3→v4 as exceeding the spread, but its paired interval
   [−0.1687, +0.0719] includes zero.** Larger than the measured noise and
   distinguishable from zero are different claims. The paper previously let
   the first stand for the second.
2. The manuscript's five inconclusive rungs are not alike. Four are
   inconclusive because the effect is small; only v2→v3 and v3→v4 are
   inconclusive for want of seeds. Of those, v3→v4 needs about 8 seeds and
   v2→v3 about 152 — so **one open question is worth re-running and one is
   not.**

The extra-seeds job I had queued earlier would not have helped: it covered
v1, v2, v4, v5, v6, v7 and **omitted v3**, the one rung the analysis says
to run. Had it finished, it would have spent roughly a day of GPU time
tightening three steps that are already resolved and left the open question
untouched. That is the concrete value of having run the power analysis
before the seeds rather than after.

Required-n figures come from a mean and a standard deviation each estimated
from three numbers and are an order-of-magnitude guide, not a plan. The
three raw per-seed differences are recorded for every step in
`results/ladder_power.json` so the intervals can be judged rather than
taken.

## 5. Validate the reliability safeguard

The yield screen was already pre-registered, with baselines, a workload
sweep and plot-bootstrap intervals, and it returns "not demonstrated". The
**crop-type** screen was never held to that standard: it reported recall
and precision at several workloads with no comparison against random
referral, no interval on AUROC, and no label-free alternative.

`scripts/screen_validate.py` supplies those, re-derived from the stored
per-field signals so no model is re-run.

- Random referral is treated as a **permutation null**, not one draw: over
  10,000 permutations the mean AUROC is 0.502 and the upper 97.5th
  percentile is 0.701. A single lucky ordering of 36 fields reaches that,
  so comparing against one draw would turn on which was taken.
- Seed disagreement scores AUROC 0.806 (95 % CI 0.639–0.946) and confidence
  0.823 (0.675–0.944). Both exclude chance and both exceed the null's
  97.5th percentile.
- Against random referral, the paired recall excess excludes zero at the
  33 %, 50 % and 75 % workloads — but not at 25 %.
- Held out by region, with the threshold set from the other regions only
  and pooled across them: recall 0.83 (0.60–1.00) at a 50 % workload
  against 0.50 for random.
- Against label-free baselines that need no model — smallest field first,
  largest first, fewest seasons first — both screens win on the point
  estimate by about 0.26 AUROC, but **the paired interval includes zero for
  two of the three**. Those baselines are not themselves informative, so
  this is a limit of 36 fields, not evidence that field size works as well.

Reported as: ranks unreliable fields better than chance and better than
random referral; **not** separable at this sample size from every
label-free alternative; not demonstrated. A cost is stated in the same
breath: seed disagreement needs several independently seeded models, so a
deployment that trains once cannot compute it.

The yield result stays in its own file and the text says the two must not
be combined into one reliability claim.

## 6. One-command reproduction with drift checks that fail

`configs/tolerances.yaml` declares 31 headline numbers, each with its
source, its tolerance, and a written reason for that tolerance.
`scripts/drift_check.py` recomputes every one from primary artefacts and
fails the build when any moves further than declared. It also checks that
the value the manuscript **prints** still rounds from the value the
artefacts hold.

Tolerances are mostly zero, because most values are deterministic
arithmetic over stored artefacts; the classes and their reasons are in the
file. Seed-dependent entries are flagged separately as SEEDS-CHANGED rather
than DRIFT, so adding a seed reads as new evidence and still stops the
build until the new values are written down deliberately.

**It caught me twice.** I entered full-precision expected values that I had
only computed to three or four decimals, inventing the remaining digits —
once for `ladder_sigma_max` and `field_spearman_size_vs_iou`, then again
for the two batch-size entries. Both are recorded as amendments in the
file, with the second noting that the same mistake was made twice and the
working rule that follows from it. No measurement changed; the registry was
wrong.

`scripts/build_all.py` now runs fifteen steps in dependency order and exits
non-zero if any check fails: yield screen, field size, field distributions,
screen validation, ablation power, figures, yield figure, manuscript,
reference table, PDF, verify_all, independent_check, drift_check,
minimal_repro, pytest, then a hash manifest. The idempotence probe still
confirms a second run changes nothing.

`scripts/plot_level_eval.py` is deliberately **not** a build step: it needs
the released weights, which are not in the repository, and takes ten
minutes on CPU. Anything missing its analysis is reported as NOT RUN rather
than averaged over silently.

### A second defect found while doing this

`scripts/plot_level_eval.py` had been broken since `src/utils.py` was
added: it put `src` on `sys.path`, so `from utils.dataset import
SICKLE_Dataset` resolved to this study's `src/utils.py` instead of the
authors' package and raised `ModuleNotFoundError`. The artefacts in
`results/plot_level/` predated that change, so nothing had re-run it. Fixed
by importing through the `src.` package, as `phase5_gate.py` and
`regression_gate.py` already did, with the reason written at the import.

## 7. Finish the manuscript audit

- All 20 references are cited; no citation points at a missing reference;
  the file is clean UTF-8 with no replacement characters. The SICKLE page
  range stays at the CVF proceedings value with the Crossref discrepancy
  documented, as before.
- Figure numbering had a **duplicate Fig. 4** after the new distribution
  figure was inserted. Renumbered to 1–8, and `figure()` now parses the
  caption number and refuses to emit two figures with the same one, so the
  build fails rather than producing a document with two Figure 4s.
- Section V-A no longer hard-codes its numbers (item 2 above).
- `verify_all.py` went from 19 checks to 31, covering the field
  distributions, the leave-fields-out result, the screen validation, the
  ablation power analysis, the completeness filter and the drift check.
- The provenance check was misleading: it passed on `len(manifests) > 0`
  while printing a clean-tree fraction beside it, which read as though the
  fraction had been verified. It now checks what it says — that every run
  recorded a commit and a dirty flag, 74 of 74 — and reports the clean
  fraction (53 %) as a fact rather than a pass condition.
- The test suite is 80 tests, all passing.

---

## Still open, and needing the user

1. **Author name, email and target venue.** The title block still reads
   `[Author name(s) — to be completed by the author]` and
   `[email — to be completed]`. I will not invent these.
2. **Measured yields.** Nothing here can separate model error from
   farmer-recall error. This bounds every yield conclusion in the paper and
   is not fixable from this dataset.
3. **Independent reproduction by another person.** `build_all.py` confirms
   the manuscript agrees with the artefacts in this repository. It cannot
   confirm that someone else, running the released code and data on their
   own machine, gets the same numbers.
4. **The sweep is paused and must be resumed** when the ladder seeds
   finish — one command, in `docs/SWEEP_PAUSED.md`. 34 of its 90 runs are
   complete and will be skipped; 6 are incomplete and will be re-trained.
5. **v2→v3 stays open and should stay open.** At about 152 seeds for 80 %
   power it is not worth attempting, and the paper says so rather than
   implying the question is settled.
6. **The PDF is now 9 pages.** It was 8 before the material added today.
   No venue has been named, so no page limit is known and nothing was cut
   to hit one. If an 8-page limit applies, the cheapest reduction that
   loses no finding is Fig. 4, a three-panel column-width figure: panels
   (a) and (b) show the same distribution two ways, and dropping (b) would
   recover most of a page.
