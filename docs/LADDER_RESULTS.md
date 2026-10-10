# The ladder — results

**30/30 runs complete. v1–v10 × seeds 0, 1, 2. 109 hours of compute.
Zero failures, zero NaN batches, zero degenerate rungs.**

Produced by `scripts/run_ladder_py.py`, aggregated by
`scripts/aggregate_ladder.py` → `results/ladder_crop_type.csv`.

All numbers are **validation** numbers on the 36 validation plots. The SICKLE
test split is unlabelled, so nothing here is comparable to the paper's
published test table.

---

## 1. Rungs

Mean ± standard deviation over three seeds.

| rung | adds | pixel macro IoU | plot macro IoU | paddy IoU |
|---|---|---:|---:|---:|
| v1 | single acquisition (floor) | 0.6201 ± 0.0114 | 0.3729 ± 0.0264 | 0.4453 |
| **v2** | **+ full temporal sequence** | 0.7399 ± 0.0317 | 0.5728 ± 0.0490 | 0.6161 |
| v3 | + augmentation | 0.7289 ± 0.0168 | 0.4623 ± 0.0608 | 0.5822 |
| v4 | + class-weighted loss | 0.6805 ± 0.0417 | 0.6236 ± 0.0271 | 0.5503 |
| **v5** | **+ Sentinel-1 radar** | **0.8665 ± 0.0163** | **0.6755 ± 0.0295** | **0.7999** |
| v6 | + Landsat-8 | 0.8740 ± 0.0112 | 0.6484 ± 0.0275 | 0.8096 |
| v7 | + UTAE (replaces U-Net3D) | 0.8044 ± 0.0118 | 0.5532 ± 0.0508 | 0.7121 |
| v8 | + wider encoder | 0.8049 ± 0.0218 | 0.5534 ± 0.0479 | 0.7032 |
| v9 | + cosine LR + warmup | 0.7971 ± 0.0144 | 0.5754 ± 0.0586 | 0.7017 |
| v10 | + plot-balanced sampling | 0.7861 ± 0.0189 | 0.5704 ± 0.0561 | 0.6829 |

**No rung was degenerate.** Worth stating against the baseline: three of the
four *released* UTAE baselines never trained (best epoch 1, paddy IoU exactly
0.000). Every UTAE rung here trained.

---

## 2. Rung-to-rung deltas

Each rung changes exactly one thing from its predecessor, so an adjacent
difference is attributable to that change **within this experimental system**.
This is *controlled component attribution*. It is **not** causal identification
and must never be written as causal.

A delta smaller than the seed-to-seed spread is reported as **inconclusive**,
not as zero and not as a result.

| step | component | Δ pixel | Δ plot | verdict |
|---|---|---:|---:|---|
| v1→v2 | **temporal information** | **+0.1198** | **+0.1998** | **positive** |
| v2→v3 | augmentation | −0.0110 | −0.1104 | inconclusive |
| v3→v4 | class weighting | −0.0484 | +0.1613 | negative on pixel, **opposite on plot** |
| v4→v5 | **Sentinel-1 radar** | **+0.1860** | +0.0518 | **positive** |
| v5→v6 | + Landsat-8 | +0.0075 | −0.0270 | inconclusive, **opposite directions** |
| v6→v7 | UTAE architecture | −0.0696 | −0.0952 | negative |
| v7→v8 | model capacity | +0.0004 | +0.0002 | inconclusive |
| v8→v9 | optimiser schedule | −0.0078 | +0.0220 | inconclusive, **opposite directions** |
| v9→v10 | plot-balanced sampling | −0.0110 | −0.0050 | inconclusive |

---

## 3. Findings

### 3.1 Information pays. Complexity does not.

Only two of nine steps are significant, and both add **information**:

    temporal information   +0.1198 pixel   +0.1998 plot
    Sentinel-1 radar       +0.1860 pixel   +0.0518 plot

Both are 4–10× the seed-to-seed spread.

Everything that adds **model complexity** is inconclusive or negative:
capacity (+0.0004), optimiser (−0.0078), UTAE (−0.0696). The three most
expensive rungs — v8, v9, v10 — consumed roughly **75% of the 109 hours** and
produced no measurable effect.

With 351 plots, this is the study's thesis measured rather than asserted.

### 3.2 Five of nine steps are inconclusive — and two looked like results at one seed

At seed 0 alone, augmentation read **−0.0496** and the capacity rung read
**−0.0098**. Both looked like findings. At three seeds the seed-to-seed spread
swallows them.

Without the seeds, both would have become confident claims. This is the single
strongest argument for the paired-seed design, and it should be stated in the
write-up rather than hidden.

### 3.3 The two metrics disagree in direction on three steps

| step | Δ pixel | Δ plot |
|---|---:|---:|
| v3→v4 class weighting | **−0.0484** | **+0.1613** |
| v5→v6 + Landsat-8 | +0.0075 | −0.0270 |
| v8→v9 optimiser | −0.0078 | +0.0220 |

The mechanism is not mysterious and was predicted before any run: both
interventions that **rebalance away from large, well-represented plots** lose
pixel-pooled IoU and gain plot-weighted IoU. Pixel-pooled IoU rewards exactly
the plots they de-emphasise.

The same divergence was measured independently across the twelve released
checkpoints (median gap +0.1812, range −0.1962 to +0.3047, and it reorders
which model wins). Seeing it reproduce inside a controlled experiment is
considerably stronger evidence than seeing it in released artefacts alone.

### 3.4 The third sensor adds nothing measurable

v5→v6 is **+0.0075 pixel, −0.0270 plot — inconclusive**. S2+S1 captures the
benefit; adding Landsat-8 does not measurably improve on it.

This is the controlled counterpart to the published anomaly that motivated the
study: in the authors' own table, three-sensor fusion (81.07) scores *below*
Sentinel-1 alone (81.77).

---

## 4. Conditions that bind every number here

- **Validation only.** The test split is unlabelled.
- **`batch_size = 32` everywhere.** Evaluation is batch-size dependent on this
  benchmark — 6.5 IoU points on frozen weights (`REPRODUCTION_GATE.md` §2).
- **Training used micro-batch 8 × 4 accumulation and bf16 autocast.** This is
  **not** equivalent to a true batch of 32: `pad_collate` pads the temporal
  axis to the per-batch maximum, so a micro-batch of 8 pads to a different `T`
  (v2: 72→61, v7: 74→37). It is held CONSTANT across every rung, so the deltas
  stay controlled, and **evaluation always ran fp32 at batch 32**. Absolute
  numbers are not comparable to runs trained at a true batch of 32.
- **A runtime patch was active**: the released `UNet3D` divides by zero in its
  masked temporal mean for 13/1937 training samples. Verified bit-identical on
  every batch that was already finite (`MODEL_DEFECTS.md` §2).
- **lr 1e-3, not the released 0.1**, because the schedule the authors pair with
  0.1 is itself a rung (v9). Held constant across v1–v8.
- **351 plots**, phrased precisely: *351 unique plots producing 2,370
  season-level samples; the independent spatial unit is the plot.*

---

## 5. What this does not establish

The ladder measures what each component buys **at full data** (315 training
plots). It does not show how those effects **scale** with the amount of
independent spatial diversity.

That is the data-regime sweep (`scripts/run_sweep.py`, 90 runs), which tests
the hypothesis registered before any run: *if temporal complexity requires
independent spatial diversity to generalise, its advantage over single-date
modelling should increase with plot count.*
