You are Claude Code acting as a senior ML research engineer and reproducibility
engineer.

I am building an M.Tech research project on the SICKLE agricultural
remote-sensing dataset.

**Title:** When Does Temporal Complexity Pay?
**Subtitle:** Independent spatial diversity and the marginal value of temporal
and multi-sensor modelling in smallholder crop mapping.

============================================================
0. ABSOLUTE RULE: DO NOT HALLUCINATE
============================================================

You MUST NOT invent: dataset paths, directory structures, filenames, band names,
sensor names, label encodings, crop IDs, class names, spatial or temporal
dimensions, preprocessing rules, splits, API behaviour, model APIs, metric
implementations, published numbers, hyperparameters, missing-value conventions,
cloud-mask fields, modality availability, coordinates, sample counts or plot IDs.

If something is not known, DO NOT GUESS. Instead: inspect the repository,
inspect the actual dataset files, inspect existing code, consult the paper,
determine the answer from evidence, document the evidence, then implement.

If an essential fact cannot be verified, STOP that part and report:

    MISSING VERIFIED INFORMATION — CANNOT IMPLEMENT SAFELY

Never silently substitute an assumption.

============================================================
1. VERIFIED FACTS — these override any contrary assumption
============================================================

Measured directly from the downloaded artefacts on 2026-10-05. If your own
inspection contradicts any of these, STOP and report the discrepancy rather than
resolving it silently.

**Locations**

    Dataset zip   D:\FPD\sickle_dataset-002.zip        6.34 GB, 7.0 GB unpacked
    Toy dataset   D:\FPD\WACV 2024\sickle_toy_dataset.zip
    Paper         D:\FPD\2312.00069v1.pdf
    Released runs D:\FPD\runs\wacv_2024\               3.4 GB — SEE SECTION 2
    Extracted CSV C:\ds\sickle\sickle_dataset_tabular.csv

**Zip structure** (3,428,250 entries)

    images/L8, images/S1, images/S2       209,039 .npz files
    masks/3m, masks/10m, masks/30m        2,995,937 .tif files
    sickle_dataset_tabular.csv            the label/metadata table

**Tabular columns**

    UNIQUE_ID, PLOT_ID, RIVER_PART, STANDARD_SEASON, YEAR, AREA, CROP, VARIETY,
    PADDY_BIN, SOWING_DATE, TRANSPLANTING_DATE, HARVESTING_DATE, SOWING_DAY,
    TRANSPLANTING_DAY, HARVESTING_DAY, YIELD, SPLIT

**Counts — the paper says 388 plots; the released tabular has 351.**

    samples total      2,370
    unique PLOT_ID       351      <-- NOT 388
    train            1,937 samples / 315 plots
    val                227 samples /  36 plots
    test               206 samples /   0 plots

**THE TEST SPLIT IS UNLABELLED.** Test rows have blank PLOT_ID, blank CROP and
blank PADDY_BIN. There is no usable held-out test set.

    => The evaluation set for this entire study is the 36 VALIDATION plots.
    => Every reported number is a validation number and must be labelled as such.
    => Numbers are NOT comparable to the paper's published table.
    => Do NOT write assertions about "test plots". There are none.

Plot-level split integrity was checked: **0 plots straddle splits.** The
released splits are clean; use them, do not re-split.

**Crop distribution (plots; val in brackets)**

    Paddy 262 [29] | Banana 37 [4] | Coconut 19 [3] | Banyan 11 [0]
    Sugarcane 11 [0] | Groundnut 10 [0] | Cotton 5 [0] | Eucalyptus 3 [0]
    Brinjal 3 [0] | + 12 further classes with <=2 plots, 0-1 in val

Several "crops" are trees (Banyan, Eucalyptus, Teak, Palm, Tamarind).

**Binary task:** `num_classes: 2`, confirmed from the authors' released configs.

PADDY_BIN counted at plot level gives 262 paddy and 105 non-paddy. **That sums
to 367, not 351.** The discrepancy is not an error and must not be "corrected":

    PADDY_BIN is a PER-SAMPLE label, and plots rotate crops between seasons.
    Exactly 16 plots carry both a paddy and a non-paddy label in different
    seasons:  262 + 105 = 367 = 351 + 16.  VERIFIED.
    24 plots grow more than one crop overall, e.g. plot 25 grows Bhendi,
    Brinjal and Tomato; plot 75 grows Paddy and Sesame.

**Methodological consequence, and it matters for the sweep:** a plot is a unit
of spatial independence, NOT a unit of class. Selecting N plots yields a
variable number of season-samples and a variable class mix. Record, for every
subset manifest, both the plot count and the resulting sample and class counts.

Recompute all of these in PHASE 0 from the CSV and report any disagreement.

**Yield:** 319 samples from 131 plots.

**Authors' training configuration** (from released conf.json)

    img_size [32,32] | epochs 100 | batch_size 32 | lr 0.1 | seed 0
    N_params ~3.1M (unet3d) | use_augmentation true
    S2 bands: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 | mask_res 10

============================================================
2. THE RELEASED RUNS — INSPECT THESE FIRST
============================================================

`D:\FPD\runs\wacv_2024\` contains the authors' own checkpoints, configs and
training logs across 5 tasks: crop_type, crop_yield, sowing_date,
transplanting_date, harvesting_date.

**Run count: 78, obtained by enumerating the directories, not by arithmetic.**
Note that 5 x 4 sensors x 3 architectures = 60, which does NOT match. The
difference is real and must be understood, not reconciled away:

    crop_type            13   (12 + one `_unet3d_debug` run)
    crop_yield           26   (12 regular + 12 `_season` variants + 2 debug)
    sowing_date          13
    transplanting_date   13
    harvesting_date      13
                         --
                         78

**Re-enumerate the directories in PHASE 0 and report any disagreement with 78.**
Exclude `*_debug` runs from any benchmark comparison.

Each run directory holds:

    checkpoint_best.pth.tar   checkpoint_last.pth.tar   conf.json   trainlog.json

`trainlog.json` holds **100 epochs of train AND val metrics** with per-class
breakdowns (paddy / non-paddy).

**This is the most valuable artefact in the project.** It supplies the
reproduction baseline without a single GPU hour. Inspect it in PHASE 0.

Already extracted from these logs (crop_type, seed 0, best-val epoch):

    sensors      convlstm   unet3d     utae     paper(reported)
    S2             45.35%   71.73%   38.12%           78.12%
    S1             41.29%   80.11%   60.10%           81.77%
    L8             76.63%   70.61%   38.12%           56.04%
    S2,L8,S1       67.99%   81.90%   51.55%           81.07%

Three observations to verify independently and then build on:

1. **U-Net3D overfits substantially.** At the FINAL epoch, train-minus-val IoU
   is +24.2 (L8), +21.1 (S2), +17.4 (fusion), +10.5 (S1). This is direct
   evidence for the study's central premise.
   NOTE: comparing train-vs-val at the *best-val* epoch biases val upward.
   Always compare at a fixed epoch.
2. **Some released baselines are degenerate.** `[L8]_utae` and `[S2]_utae` both
   finish at 38.1% val with **0.0% paddy IoU** — they predict everything as
   non-paddy.
3. **L8 diverges from the paper by +14.6 points.** Investigate; do not assume.

============================================================
3. RESEARCH QUESTION
============================================================

> How much independent spatial data does temporal and multi-sensor complexity
> need before it starts paying?

The contribution is a controlled empirical study, NOT a new architecture.
No architecture is proposed and none may be claimed.

The independent spatial unit is the **PLOT**. Never treat the ~209k satellite
observations as independent samples. Correct phrasing:

> 351 unique plots producing 2,370 season-level samples and ~209k satellite
> observations; the independent spatial unit for generalisation is the plot.

============================================================
4. TASK ARM A — PRIMARY (and currently the only viable arm)
============================================================

Binary semantic segmentation: **Paddy vs Non-Paddy**, using the official
formulation and `PADDY_BIN`.

Headline metric: **Macro IoU**. Also report Macro F1, Paddy IoU, Non-Paddy IoU.

Do not alter label semantics. Do not fabricate a 21-class benchmark.

============================================================
5. TASK ARM B — CONDITIONAL, PROBABLY NOT VIABLE
============================================================

Arm B was intended as a reduced K-class task to test whether the data
requirement scales with task difficulty.

**The measured distribution says it is almost certainly not viable.** In the 36
validation plots there are 29 Paddy, 4 Banana, 3 Coconut and essentially
nothing else. No sensible `MIN_PLOTS_PER_CLASS` yields a usable K.

**Therefore:**

1. In PHASE 1, compute the per-class plot counts per split and write
   `artifacts/arm_b_viability.json` stating, for each candidate
   `MIN_PLOTS_PER_CLASS` in {3,5,10,15,20}, the resulting K and the minimum
   number of validation plots in any retained class.
2. **Pre-specified feasibility criterion for this study** (not a general
   statistical sufficiency claim): Arm B is viable only if at least 3 retained
   classes have >= 5 validation plots each. Otherwise write
   `arm_b_status: NOT_VIABLE`, **skip Arm B entirely**, and record the reason in
   the experiment matrix.
3. Do not quietly lower the threshold to manufacture viability.

**If Arm B is not viable, the difficulty axis must be replaced, not dropped.**
The tabular file supports two alternatives, both verified present:

    YEAR            2018 / 2019 / 2020, all present in train and val
    RIVER_PART      5 sub-basins (Upper Cauvery, Middle Vennar, Lower Cauvery,
                    Coastal Cauvery, Coastal Vennar)

A held-out-year or held-out-region regime is a legitimate and arguably more
realistic difficulty axis. **But implementing it requires building your own
splits, which forfeits the authors' adjacent-plot masking** — so it must
re-implement that masking explicitly and say so. Propose this to me before
implementing; do not choose unilaterally.

============================================================
6. EXPERIMENTAL LADDER
============================================================

Do not change the scientific meaning of a variant.

    V1   S2, single image, U-Net2D                     spatial baseline
    V2   V1 + ImageNet-pretrained encoder              transfer; SWEEP BASELINE
    V3   S2, temporal, U-Net3D                         temporal information
    V4   S2, temporal, UTAE                            temporal attention
    V5   S2, temporal, ConvLSTM                        temporal recurrence
    V6   **U-Net3D**, S2 temporal — DECLARED REFERENCE see below
    V7   V6 + S1 + L8                                  multi-sensor information
    V8a  V7 + verified cloud/quality masking           observation quality
    V8b  V7 + modality dropout                         sensor robustness
    V9a  V7 + cross entropy                            loss reference
    V9b  V7 + class-weighted CE (weights from TRAIN only)
    V9c  V7 + focal loss (gamma/alpha configured, never tuned on eval)
    V10  Pre-specified integrated configuration        combination effect

**V6 IS U-NET3D, NOT UTAE.** A correction to an earlier plan.

**Primary rationale — benchmark continuity.** The SICKLE paper states that
results are reported "using the same benchmarking model (U-Net 3D for
time-series and U-Net 2D for single image) for a fair comparison". U-Net3D is
therefore the benchmark-compatible temporal reference, and V2 (single-image)
pairs with U-Net2D by the same logic.

**The authors' released runs are used to verify that their configuration is
reproducible — NOT to select the reference architecture on observed
performance.** Selecting an architecture because it scored well on validation,
even someone else's validation, is selection on the evaluation set and a
reviewer would be right to object.

(For completeness, and not as the selection reason: those runs show U-Net3D at
70.6-81.9% val IoU against UTAE at 38.1-60.1%, with two UTAE runs degenerate at
0% paddy IoU. This is consistent with, but does not justify, the choice.)

V6 is declared **before** any run of this study. V3/V4/V5 remain an
equal-footing three-way comparison that feeds **no selection**.

**V10 is pre-specified and must be run even if it performs worse.** Define it as
U-Net3D + S1/S2/L8 + modality dropout + focal loss. It is NOT "the
best-performing combination".

============================================================
7. PRIMARY EXPERIMENT — DATA-REGIME SWEEP
============================================================

**This is the paper. It must not be dropped for compute reasons.**

Plot counts — **the training split has 315 plots, so 388 does not exist**:

    50, 100, 150, 200, 250, 315

Configurations: **V2** (single-image), **V6** (temporal), **V7** (multi-sensor).

Seeds: 5.

    6 plot counts x 3 configurations x 5 seeds = 90 runs   (Arm A)
    + 90 more only if Arm B is declared viable

Evaluation is on the 36 validation plots, every time.

**Run this sweep BEFORE the full ladder.** If the curves do not separate beyond
their own noise, the study needs rethinking and that must surface in week one.

Pre-registered, falsifiable hypothesis:

> If temporal complexity requires independent spatial diversity to generalise,
> its advantage over single-date modelling should increase with plot count. If
> it does not, the data-quantity explanation is weakened.

Do not assume a crossover exists. All outcomes are reportable, including
"NO CROSSOVER OBSERVED" and curves that are flat and parallel — the latter
falsifies the premise and sends the investigation to observation quality or
label noise instead.

============================================================
8. SAMPLING AND LEAKAGE
============================================================

Sweep subsetting operates at **PLOT** level. Never subsample pixels,
observations or patches in a way that splits a plot.

Deterministic, reusable plot selection. Store manifests:

    artifacts/subsets/plots_050_seed001.json  ...

Nested subsets are preferred (the 50-plot set is a subset of the 100-plot set)
so that curves are not confounded by disjoint sampling; state which you chose.

Before training, print TRAIN PLOTS and VALIDATION PLOTS and assert their
intersection is empty. **Do not assert on test plots — there are none.**

Also assert, and fail loudly on violation:
- normalisation statistics computed from TRAIN only
- class weights computed from TRAIN only
- no validation-derived threshold, model or hyperparameter selection

============================================================
9. PROTOCOL SELECTION — PRE-DECLARED
============================================================

With a 36-plot evaluation set, repeated selection against it will fit the
protocol to the evaluation data. Therefore:

- V6 is fixed a priori (Section 6).
- V10 is fixed a priori.
- Any further selection rule must be written into `configs/study.yaml` before
  the runs it governs, and recorded in `docs/STATISTICAL_PROTOCOL.md`.

============================================================
10. METRICS AND STATISTICS
============================================================

### 10.1 Primary metric

Headline metric is **Macro IoU (mIoU)**, matching the SICKLE crop-type
evaluation. For the binary task report, in this order:

    1. Macro IoU            [PRIMARY]
    2. Paddy IoU
    3. Non-Paddy IoU
    4. Macro F1             [SECONDARY]
    5. Paddy precision      6. Paddy recall
    7. Non-Paddy precision  8. Non-Paddy recall

**Do NOT use pixel accuracy as a primary or headline metric.** The classes are
imbalanced and accuracy hides collapse: the released `[S2]_utae` run reaches
38.1% val IoU with **0.0% paddy IoU** — a model predicting a single class
everywhere — and accuracy would not reveal that.

Be explicit about ignored/invalid pixels (`ignore_index: -999` in the authors'
config). Define and document the behaviour when a class has no valid
ground-truth pixels in the evaluation region; do not silently emit 0 or NaN.

### 10.2 Plot-level generalisation (secondary)

The independent spatial unit of this study is the plot, so also compute
**Plot-Macro IoU**:

    for each validation plot: compute IoU over that plot's pixels
    Plot-Macro IoU = unweighted mean of per-plot IoU

Every plot receives equal weight, which pixel-pooled mIoU does not give.

Two rules that must be fixed before computing it, because the data forces the
question (Section 1: plots rotate crops across seasons):

- **Aggregation unit.** A plot contributes several season-samples. Pool all of a
  plot's valid pixels across its samples into one confusion matrix, then compute
  that plot's IoU. Do not average per-sample IoU. State this in the docs.
- **Undefined classes.** If a plot contains no ground-truth pixels of a class,
  that class's IoU is undefined for that plot. Exclude it from that plot's macro
  average and record the exclusion count. Never substitute 0.

Report plot-level variability, and **95% bootstrap confidence intervals obtained
by resampling PLOTS, not pixels.**

**Plot-Macro IoU is an additional spatial-generalisation analysis. It does NOT
replace benchmark mIoU**, which remains the primary comparable metric.

### 10.3 Paired comparisons

**Primary comparisons are PAIRED seed-wise differences**, not two independent
means:

    for each seed: delta = metric(V6) - metric(V2)
    report mean(delta), sd(delta), CI where appropriate

Report the paired delta for **Macro IoU, Paddy IoU, Non-Paddy IoU, Macro F1**
and Plot-Macro IoU.

Write `results/deltas.csv` with: task_arm, plot_count, seed, baseline, variant,
metric, baseline_value, variant_value, delta.

Where |mean Delta| < sd(Delta), report the effect as **small relative to
observed seed-to-seed variability and therefore inconclusive**. Do not state
that the variant "has no effect" — a small mean relative to variance is not
evidence of a zero effect, and the distinction matters in a thesis.

### 10.4 Statistical unit — two levels, never confused

- **Model-to-model comparison:** the five seeds are the repeated experimental
  runs. n = 5.
- **Spatial generalisation uncertainty:** the validation plots are the
  resampling units. n = 36.
- **Never treat individual pixels or satellite observations as independent
  statistical samples.** With ~209k observations from 351 plots, a pixel-level
  interval would be wrong by orders of magnitude.

Five seeds is a small sample. Do not call a difference "significant" because one
mean is larger. If a test is used, document test, assumptions, exact n, p-value
and effect size.

============================================================
11. COMPUTE, REPORTED AS A RESULT
============================================================

Record per run: parameter count, trainable parameters, training time, inference
time, peak GPU memory. Write `results/compute.csv` and report performance
improvement per unit cost, with the cost metric defined explicitly.

Target hardware: **RTX 5060 8 GB, 32 GB RAM, i7-14700HX.**

- AMP by default; configurable batch size; gradient accumulation.
- 8 GB will bite at V7/V10 (three sensors, long sequences). Fail gracefully on
  OOM and name the parameter to change.
- **Profile data-loading throughput during the first representative run.** Do
  not assume whether GPU compute, CPU preprocessing, RAM pressure or disk I/O
  dominates. If repeated loading proves to be a bottleneck across ~155 runs,
  implement a caching or preloading strategy within available RAM, and report
  the measurement that motivated it.
- **Checkpoint and resume every run.** This machine has lost a multi-hour
  experiment to a shutdown, and Windows Smart App Control intermittently blocks
  SciPy DLLs — the runner must retry a failed run rather than abort the batch.

============================================================
12. RUN ACCOUNTING
============================================================

    Ladder   13 variants x 5 seeds                    =  65
    Sweep     6 plot counts x 3 configs x 5 seeds      =  90   (Arm A)
    ------------------------------------------------------------
    Total planned                                     = 155
    (+90 only if Arm B is declared viable in PHASE 1)

Generate `experiments/experiment_matrix.csv` programmatically and validate the
count. If a run is intentionally skipped, the matrix must state why. Never
silently omit a run.

============================================================
13. ENGINEERING REQUIREMENTS
============================================================

Config-driven, one YAML per variant plus `configs/base.yaml` and
`configs/study.yaml`. No experimental parameter hidden in Python.

Single training engine: `python -m src.train --config configs/v6.yaml --seed 1`.
Supports checkpoint, resume, best/last, AMP, gradient accumulation, clipping,
deterministic seeding, logging, failure recovery.

**Never continue silently after NaN loss** — save diagnostics, identify the
batch, stop the run, mark it FAILED, and do not report it as a success.

Every run writes `run_manifest.json`: experiment_id, variant, task_arm, seed,
plot_count, dataset_version, git_commit, config_hash, model, loss, modalities,
training parameters, normalisation id, checkpoint path, start/end time, status,
hardware, software environment.

Before any large experiment: `python scripts/smoke_test.py`, verifying dataset
loading, labels, tensor shapes, model construction, forward, loss, backward,
optimiser step, checkpoint save/load, evaluation, metrics. It must end with
SMOKE TEST PASSED or SMOKE TEST FAILED, and must never claim success it did not
execute.

Unit tests (pytest) for dataset indexing, split integrity, class mapping,
normalisation, metrics, losses, model output shapes, modality dropout,
checkpoint restore, seed reproducibility. Do not tell me tests pass unless they
actually passed.

============================================================
14. SCIENTIFIC CLAIMS
============================================================

- Never say "causal". Use **controlled component attribution**.
- Never claim "first study", "no previous work" or "novel architecture".
- Do not call the data-regime relationship a "law". Use **data-regime
  relationship**.
- Related work must be worded safely. Six works use SICKLE (STF-GatedCropNet,
  SAR2Agri, SPEAR, AnySat, Time2Agri, Temporal-Aware Spatial Interaction
  Transformer); all propose methods. Say that a controlled component-attribution
  study "remains insufficiently explored", not that "none ablates".
- Do not claim self-supervised pretraining, multimodal transformer fusion or
  adaptive sensor weighting as contributions — SPEAR and Time2Agri occupy them.

============================================================
15. NO FAKE RESULTS — ABSOLUTE
============================================================

Never generate fake accuracy, IoU, F1, loss, runtime, GPU memory, parameter
count, crossover point, significance or improvement percentage.

If not run: **NOT RUN**. If failed: **FAILED**. If unavailable:
**NOT AVAILABLE**. Use `[RESULT NOT YET RUN]` in the README.

Never say "dataset loaded successfully" unless verified; "training completed"
unless it did; "UTAE implementation" if it is an unverified approximation;
"cloud masking" if the dataset lacks the required field.

============================================================
16. IMPLEMENTATION ORDER — STOP AND VERIFY AT EACH PHASE
============================================================

    PHASE 0   Repository audit + inspect D:\FPD\runs\wacv_2024 released logs
    PHASE 1   Dataset inspection; per-class plot counts; ARM B VIABILITY DECISION
    PHASE 2   Dataset loader (verify one complete sample end to end)
    PHASE 3   Plot-level split system + leakage assertions
    PHASE 4   Arm A binary task
    PHASE 5   Reproduce ONE released cell from the authors' checkpoint
              ([S2]_unet3d crop_type). THIS IS THE GATE — nothing proceeds until
              a released number is recovered.
    PHASE 6   V1/V2 spatial models
    PHASE 7   V3/V4/V5 temporal models
    PHASE 8   V6 declared reference (U-Net3D)
    PHASE 9   V7 multi-sensor
    PHASE 10  V8a/V8b   PHASE 11  V9a/b/c   PHASE 12  V10
    PHASE 13  Smoke + unit tests
    PHASE 14  ONE seed of ONE variant, timed mid-batch (the laptop throttles)
    PHASE 15  DATA-REGIME SWEEP (90 runs) — the paper
    PHASE 16  Full ladder (65 runs)
    PHASE 17  Aggregate   PHASE 18  Statistics   PHASE 19  Figures
    PHASE 20  Reproducibility audit

Note the order: **the sweep runs before the ladder**, because it is the result.

============================================================
17. DELIVERABLES
============================================================

Repository tree; files created/modified; explanation of each config; exact
commands for smoke test, each variant, the sweep, the ladder, aggregation,
statistics, figures and resume; dataset configuration requirements; everything
that could NOT be verified; implementation limitations; tests actually executed
with their real results; current git commit hash; and a final REPRODUCIBILITY
STATUS section.

Main figure: x = training plots (50..315), y = Macro IoU, three curves
(V2 / V6 / V7), with uncertainty. Save PNG and PDF.

============================================================
18. MOST IMPORTANT OPERATING RULE
============================================================

Optimise for CORRECT, VERIFIED, RUNNABLE, REPRODUCIBLE, RESEARCH-VALID code —
not for volume of code.

Given a choice between "write something plausible" and "stop because the
evidence is insufficient", **always stop and report**.

============================================================
START NOW — PHASE 0
============================================================

Do not implement models yet.

1. Inspect the repository.
2. **Inspect `D:\FPD\runs\wacv_2024\` — the authors' released configs, logs and
   checkpoints. Confirm or refute the three observations in Section 2.**
3. Inspect `C:\ds\sickle\sickle_dataset_tabular.csv` and confirm or refute every
   count in Section 1.
4. Inspect the dataset zip structure.
5. Produce `docs/IMPLEMENTATION_AUDIT.md`.
6. List every fact in Section 1 you could NOT confirm, and every discrepancy.
7. Identify reusable components and exact blockers.

Then show me the audit before any large-scale implementation.
