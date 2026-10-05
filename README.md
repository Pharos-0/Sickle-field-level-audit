# When Does Temporal Complexity Pay?

*Independent spatial diversity and the marginal value of temporal and
multi-sensor modelling in smallholder crop mapping*

A controlled empirical study on the **SICKLE** dataset (Cauvery Delta, Tamil
Nadu, India). The contribution is a measurement, not an architecture — **no new
model is proposed and none is claimed**.

> **Status: PHASE 0–6 complete. v1 training is RUNNING. v2–v10 NOT RUN.**
> Every number here is either measured from released artefacts or marked
> `NOT RUN`.

---

## The question

Temporal modelling *should* beat a single snapshot — crops change through a
season. **On SICKLE it doesn't.** From the authors' own published results:

| Observation | Implication |
|---|---|
| Yield: single-image Landsat-8 **46.74%** MAPE beats every time-series variant (54.00–73.59%) | more time makes it worse |
| Crop type: Sentinel-1 alone **81.77%** IoU beats three-sensor fusion **81.07%** | more sensors make it no better |

With only **351 plots**, the suspect is that complexity carries a data cost
this dataset cannot pay. The ladder measures where that cost turns.

---

## The ladder

Ten rungs, each inheriting from the previous one and changing **exactly one
thing**, so any adjacent difference is attributable to that change *within this
experimental system*. This is **controlled component attribution** — never
"causal".

| rung | adds | isolates |
|---|---|---|
| **v1** | single-acquisition S2, unweighted CE, no aug | the floor |
| **v2** | full temporal sequence | **temporal effect** |
| v3 | augmentation | augmentation effect |
| v4 | class-weighted loss | class imbalance effect |
| v5 | Sentinel-1 radar | radar fusion effect |
| v6 | Landsat-8 | three-sensor fusion effect |
| v7 | UTAE temporal attention | temporal architecture effect |
| v8 | wider encoder | model capacity effect |
| v9 | cosine LR + warmup | optimisation effect |
| v10 | plot-balanced sampling | spatial unit effect |

`v2 − v1` is the headline measurement.

**v1 is "single-acquisition", not "a 2D baseline".** U-Net3D halves the temporal
axis twice, so T=1 collapses to T=0 and the forward pass is impossible. The one
chosen frame is therefore repeated to 4 **constant** timesteps: zero temporal
information, identical architecture, parameter count and collate path to v2.

```bash
python scripts/check_configs.py       # resolve all 10 rungs, assert invariants
bash scripts/smoke_test.sh v1 v2      # 2 batches each; NOT a result
bash scripts/train.sh v1 0            # one rung, one seed
SEEDS="0 1 2" bash scripts/run_ladder.sh
python scripts/aggregate_ladder.py    # the table, with NOT RUN / DEGENERATE flags
```

---

## What has been established

### Dataset facts — 51 checks, 0 refuted

| | |
|---|---|
| Samples / plots | 2,370 season-level samples from **351 plots** (the paper says 388) |
| Splits | train 1,937 / 315 plots · val 227 / **36 plots** · test 206 / **0 plots** |
| **Test split** | **Unlabelled.** All 206 rows blank in PLOT_ID, CROP and PADDY_BIN, and no masks exist. The 36 validation plots are the entire evaluation set. |
| Task | **Binary paddy vs non-paddy**, `num_classes: 2`, **paddy = class 0** |

### The six mask bands — decoded from evidence

| Band | Meaning | Evidence |
|---|---|---|
| **0** | `PLOT_ID` | present in **600/600** sampled masks |
| **1** | crop code | exactly one value per crop over 800 samples |
| 2 / 3 / 4 | sowing / transplanting / harvesting day | 45/45, 100/100, 45/45 |
| **5** | **yield ÷ pixel count** | sums to the tabular YIELD in **200/200** |

**Band 5 is a trap** — per-pixel yield regression fits something that scales
inversely with plot area; the target must be the sum over the plot.
**Band 0 is what makes Plot-Macro IoU possible.**

### Arm B (reduced multi-class): NOT VIABLE

Tested against a criterion fixed before computing it (≥3 classes, each with ≥5
validation plots). **Every threshold fails.** Run count is 155, Arm A only.

### Reproduction gate — 12 cells, 9 PASS

Tolerance **±0.01 declared before any run**, never widened.
`PASS 9/12 · median |Δ| 0.0012 · max 0.0345`. Trainable parameter counts match
the released configs exactly. Details in `docs/REPRODUCTION_GATE.md`.

---

## Three findings

### 1. Evaluation depends on batch size

Same checkpoint, same data, same code, frozen weights:

| batch_size | `[L8]_unet3d` | Δ |
|---:|---:|---:|
| 32 | 0.7071 | 0.0009 |
| 16 | 0.6561 | 0.0500 |
| 8 | 0.6409 | 0.0652 |

**6.5 IoU points from batch size alone.** `pad_collate` pads the temporal axis
to the per-batch maximum, and the pad mask is realigned by **truncation**
instead of pooling. **Any IoU reported without its evaluation batch size is not
reproducible.** This study pins `batch_size = 32` everywhere.

### 2. The released U-Net3D produces NaN, and nothing guards it

The same truncated mask makes the masked-mean denominator **zero** for a sample
whose early acquisitions are empty:

| split | samples hitting 0/0 | share |
|---|---:|---:|
| train | **13 / 1937** | 0.7% |
| val | **0 / 227** | 0.0% |

`grep -rn "isnan\|isfinite\|nan_to_num" reference_impl/` returns **nothing**, so
NaN gradients from one batch destroy every weight. Not an optimiser problem —
the NaN count is identical at lr 0.1 / 0.01 / 0.001 / 0.0001, and appears with
freshly initialised weights and zero optimiser steps.

`src/patches.py` clamps the denominator, verified **bit-identical on 9/9 batches
that were already finite, 0 differing, 3 NaN→finite**, so it cannot change any
previously valid number. `reference_impl/` is never edited.
See `docs/MODEL_DEFECTS.md`.

### 3. Pixel-pooled and plot-weighted IoU disagree, and reorder the benchmark

Across **all 12** released cells:

| | |
|---|---|
| median gap (pixel − plot) | **+0.1812** |
| range | +0.0104 to +0.3047, **negative in one cell** (−0.1962) |
| best cell, pixel-pooled | `[S2,L8,S1]_unet3d` **0.8217** |
| best cell, **plot-weighted** | `[L8]_convlstm` **0.7557** |

The three-sensor fusion model that wins the published metric **loses to a
single sensor** on the plot-weighted one, with non-overlapping 95% CIs
([0.422, 0.620] vs [0.629, 0.876]). Both metrics are reported everywhere.

**Paddy is the harder class in all twelve cells**, often by 20–30 IoU points.

---

## Repository

```
configs/        base.yaml + v1..v10.yaml   the ladder, one change per rung
src/
  utils.py      config resolution, seeding, run manifests
  dataset.py    SICKLE wrapper: single-date mode, subsets, plot-balanced sampling
  model.py      nested YAML -> the reference's flat CFG
  loss.py       CE, optional inverse-frequency weights
  patches.py    the UNet3D NaN fix, installed at runtime
  train.py      training engine
  eval.py       both metric families side by side
  predict.py    qualitative prediction maps
  evaluation/metrics.py   macro IoU, Plot-Macro IoU, plot bootstrap, paired deltas
scripts/
  check_configs.py    validate the ladder (no torch, no data)
  train.sh/.ps1       run one rung
  smoke_test.sh/.ps1  prove the pipeline moves
  run_ladder.sh       the full ladder, SEQUENTIALLY
  aggregate_ladder.py the results table (flags NOT RUN / DEGENERATE / smoke)
  phase0_verify.py    verify every claimed fact
  phase5_gate.py      the reproduction gate
  plot_level_eval.py  Plot-Macro IoU on released checkpoints
notebooks/      explore_sickle.ipynb
shims/          tifffile-backed rasterio (Smart App Control blocks real GDAL)
reference_impl/ official SICKLE code, cloned, UNMODIFIED
tests/          29 passing
docs/           MODEL_DEFECTS · REPRODUCTION_GATE · IMPLEMENTATION_AUDIT
TRAINING_GUIDE.md
```

## Data

| | | |
|---|---|---|
| **Canonical** | `D:\FPD\sickledataset extracted` | 209,039 npz · 6,492 masks · csv md5 `e804c1c1…` |
| Archive | `D:\FPD\sickle_dataset-002.zip` | 6.34 GB |
| **Toy — do not use** | `D:\FPD\WACV 2024` | 19,859 npz · 651 masks · **different csv** `0cb87468…` |
| ~~`D:\FPD\sickle_data`~~ | **deleted** | redundant npz-only copy; 2.63 GB reclaimed |
| Released runs | `D:\FPD\runs\wacv_2024` | 78 runs over 5 tasks |

Pointing `data.root` at the toy folder loads without complaint and silently
changes the evaluation set.

## Environment notes specific to this machine

- **rasterio is blocked by Windows Smart App Control** (3/3 attempts). The shim
  supplies the one API the reference uses and raises on anything else.
- **SciPy DLLs are intermittently blocked** by the same policy — retry before
  debugging.
- **The discrete GPU powers down when idle** and `nvidia-smi` needs admin.
  CUDA was unavailable for the v1 launch; it runs on CPU at ~4.2 min/epoch.

## Limitations, stated up front

- **Validation-set numbers only.** The test split is unlabelled.
- **351 plots**: *351 unique plots producing 2,370 season-level samples and
  ~209k satellite observations; the independent spatial unit for generalisation
  is the plot.* Never "209k independent samples".
- One region, one cropping system, paddy-dominated.
- Attribution is **within this experimental system** — *controlled component
  attribution*, never "causal".
- The ladder uses `lr 1e-3`, not the released `0.1`, because the scheduler the
  authors always pair with 0.1 is itself a rung (v9). The ladder is internally
  controlled but **not lr-matched to the released baselines**; its results are
  not reproductions of them.
- No architecture is proposed.

## Next

1. v1 finishes (~7 h), then v2 — `v2 − v1` is the headline number
2. v3–v10, then seeds 1 and 2
3. Data-regime sweep (90 runs)
4. **Open decision:** Arm B's replacement difficulty axis — held-out `YEAR`,
   held-out `RIVER_PART`, or drop it
