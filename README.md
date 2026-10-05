# When Does Temporal Complexity Pay?

*Independent spatial diversity and the marginal value of temporal and
multi-sensor modelling in smallholder crop mapping*

A controlled empirical study on the **SICKLE** dataset (Cauvery Delta, Tamil
Nadu, India). The contribution is a measurement, not an architecture — **no new
model is proposed and none is claimed**.

> **Status: PHASE 0–5 complete. No study training runs have been executed yet.**
> Every number in this repository is either measured from the released
> artefacts or marked `[RESULT NOT YET RUN]`.

---

## The question

Temporal modelling *should* beat a single snapshot — crops change through a
season. **On SICKLE it doesn't.** From the authors' own published results:

| Observation | Implication |
|---|---|
| Yield: single-image Landsat-8 **46.74%** MAPE beats every time-series variant (54.00–73.59%) | more time makes it worse |
| Crop type: Sentinel-1 alone **81.77%** IoU beats three-sensor fusion **81.07%** | more sensors make it no better |

With only **351 plots**, the suspect is that complexity carries a data cost
this dataset cannot pay. The study measures where that cost turns.

---

## What has been established so far

### Dataset facts — 51 checks, 0 refuted

`scripts/phase0_verify.py` → `artifacts/phase0_verification.csv`

| | |
|---|---|
| Samples / plots | 2,370 season-level samples from **351 plots** (the paper says 388) |
| Splits | train 1,937 / 315 plots · val 227 / **36 plots** · test 206 / **0 plots** |
| **Test split** | **Unlabelled.** All 206 rows blank in PLOT_ID, CROP and PADDY_BIN, and no masks exist for them. The 36 validation plots are the entire evaluation set. |
| Split integrity | 0 plots straddle splits — the released splits are clean, use them |
| Task | **Binary paddy vs non-paddy**, `num_classes: 2` (21 crop labels exist but the benchmark is binary) |

### The six mask bands — decoded from evidence

Undocumented anywhere available; derived empirically and then independently
confirmed against the authors' code.

| Band | Meaning | Evidence |
|---|---|---|
| **0** | `PLOT_ID` | present in **600/600** sampled masks |
| **1** | crop code | on own-plot pixels, exactly one value per crop over 800 samples |
| 2 / 3 / 4 | sowing / transplanting / harvesting day | 45/45, 100/100, 45/45 |
| **5** | **yield ÷ pixel count** | sums to the tabular YIELD in **200/200** |

Codes: `1 Paddy · 2 Banana · 3 Coconut · 4 Banyan · 5 Sugarcane · 6 Palm ·
7 Groundnut · 8 Eucalyptus · 9 Tamarind · 10 Mixed · 11 Cotton · 15 Betelwine`

**Band 5 is a trap.** It is total plot yield spread over the plot's pixels, so
a per-pixel regression target fits something that scales inversely with plot
area. The target must be the **sum over the plot**.

**Band 0 is what makes Plot-Macro IoU possible** — without it, per-plot metrics
could not be computed at all.

### Arm B (reduced multi-class): NOT VIABLE

Tested against a criterion fixed before computing it (≥3 classes, each with ≥5
validation plots). **Every threshold fails** — beyond Paddy (29 val plots) only
Banana (4) and Coconut (3) have any validation support.
`artifacts/arm_b_viability.json`. **Run count is 155, Arm A only.**

### Reproduction gate — 12 cells, 9 PASS, and it produced a finding

`docs/REPRODUCTION_GATE.md`. Tolerance **±0.01 declared before any run** and
never widened. Single sequential sweep at `batch_size = 32`, recorded in every
output file.

    PASS at +/-0.01 : 9/12
    abs delta       : median 0.0012   min 0.0000   max 0.0345

Exact or near-exact: `[S2]_utae` 0.0000 · `[L8]_utae` 0.0000 ·
`[S1]_convlstm` 0.0000 · `[L8]_convlstm` 0.0002 · `[L8]_unet3d` 0.0009 ·
`[S2,L8,S1]_convlstm` 0.0005. Trainable parameter counts match the released
configs exactly.

**Three released baselines never trained.** `[S2]_utae`, `[L8]_utae` and
`[S2,L8,S1]_utae` are all best at **epoch 1**; the first two score an identical
0.3812 with **paddy IoU exactly 0.000** -- they predict one class everywhere.
Three of the four UTAE cells in the published benchmark are degenerate.

**Paddy is the harder class in all twelve cells**, often by 20-30 IoU points
(best cell 0.731 vs 0.912). Macro IoU alone hides this.

**The finding: evaluation results depend on batch size.** Same checkpoint, same
data, same code — only batch size changed:

| batch_size | `[L8]_unet3d` | Δ |
|---:|---:|---:|
| 32 | 0.7071 | 0.0009 |
| 16 | 0.6561 | 0.0500 |
| 8 | 0.6409 | 0.0652 |

**A 6.5 IoU-point swing from batch size alone, at evaluation, with frozen
weights.** Mechanism: `pad_collate` pads the temporal axis to the *per-batch
maximum*, and in `pastis_unet3d.forward` the pad mask is realigned to the
pooled temporal length by truncation (`pad_mask[:, :final.shape[-1]]`) rather
than by pooling. Whether masking engages therefore depends on batch
composition.

**Consequence: any IoU reported on this benchmark without its evaluation batch
size is not reproducible.** This study fixes `batch_size = 32` everywhere and
records it in every run manifest.

---

## Repository

```
configs/          per-variant YAML                      [NOT YET WRITTEN]
src/
  evaluation/metrics.py   macro IoU, per-class IoU/F1, Plot-Macro IoU,
                          plot-level bootstrap, paired seed deltas
scripts/
  phase0_verify.py        verify every claimed fact
  phase1_inspect.py       Arm B viability + on-disk format
  phase1_mask_semantics.py  decode the six mask bands
  phase5_gate.py          THE REPRODUCTION GATE
  extract_dataset.py      archive -> disk (skips the redundant raw tif tree)
  make_subsets.py         nested plot-level subsets for the sweep
  make_previews.py        human-viewable imagery
shims/rasterio.py         tifffile-backed; Smart App Control blocks real GDAL
reference_impl/           official SICKLE code (cloned, unmodified)
tests/                    20 passing unit tests
docs/
  IMPLEMENTATION_AUDIT.md   PHASE 0 + 1 findings
  REPRODUCTION_GATE.md      PHASE 5 and the batch-size finding
  SICKLE_MASTER_PROMPT.md   the study specification
artifacts/                verification, viability, subsets, dataset format
results/previews/         contact sheet, per-plot overviews, time series
results/gate/             one JSON per reproduced cell
```

## Data

| | |
|---|---|
| Archive | `D:\FPD\sickle_dataset-002.zip` (6.34 GB) |
| Extracted | `D:\FPD\sickle_data` — **215,532 files, 3.0 GB** |
| Released runs | `D:\FPD\runs\wacv_2024` — 78 runs with configs, 100-epoch logs, checkpoints |

The archive carries **two copies of the imagery**: `images/{SAT}/npy/` (209,039
`.npz`, what the dataloader reads) and `images/{SAT}/tif/` (**2.99M raw
per-band GeoTIFFs the reference code never touches**). Extraction skips the
latter by default — 215k files instead of 3.2M.

## Running it

```bash
# verification and inspection (no GPU)
python scripts/phase0_verify.py
python scripts/phase1_inspect.py
python scripts/phase1_mask_semantics.py --n 300

# the gate -- nothing proceeds until this passes
python scripts/phase5_gate.py --run "[S2]_unet3d" --batch-size 32

# sweep subsets and previews
python scripts/make_subsets.py
python scripts/make_previews.py

pytest tests -q
```

`scripts/phase5_gate.py` sets `PYTHONPATH` for the rasterio shim internally.

## Environment notes specific to this machine

- **rasterio is blocked by Windows Smart App Control** (`DLL load failed ...
  An Application Control policy has blocked this file`), consistently. The
  shim at `shims/rasterio.py` supplies the one API the reference code uses and
  raises on anything else rather than faking it.
- **SciPy DLLs are intermittently blocked** by the same policy — a suite can
  fail and then pass unchanged. Retry before debugging.
- **The discrete GPU intermittently powers down.** CUDA was available early in
  a session and gone later with no code change. The gate runs on CPU.

## Limitations, stated up front

- **Validation-set numbers only.** The test split is unlabelled; results are
  not comparable to the paper's published table.
- **351 plots**, phrased precisely: *351 unique plots producing 2,370
  season-level samples and ~209k satellite observations; the independent
  spatial unit for generalisation is the plot.* Never "209k independent
  samples".
- One region, one cropping system, paddy-dominated.
- Attribution is **within this experimental system** — the term used is
  *controlled component attribution*, never "causal".
- No architecture is proposed.

## Next

1. Dataloader + config system
2. Time one training run, budget the 155
3. **Data-regime sweep (90 runs) — this is the paper**
4. Ladder (65 runs), aggregation, statistics, figures
