# Overnight session — 2026-10-05

What was built, what was found, what did not happen, and what to do next.

**Everything is at `D:\FPD\sickle-project`. The dataset is extracted and kept at
`D:\FPD\sickle_data`.**

---

## 1. What did NOT happen, stated first

- **No study training runs.** Zero of the 155 planned runs were executed.
- **No models, configs or training engine written.** PHASE 6 onward is untouched.
- **The GPU became unavailable mid-session** — CUDA was working early (the
  earlier detection experiment used it) and reported `cuda available: False`
  later with no code change. The discrete GPU appears to power down when the
  machine idles. All overnight work ran on CPU, which was sufficient.

What *was* done is the verification and infrastructure that was blocking
everything else.

---

## 2. The dataset is extracted and viewable

`D:\FPD\sickle_data` — **215,532 files, 3.0 GB**, counts verified exactly:

    images/S2/npy  126,503      masks/10m  2,164
    images/S1/npy   37,657      masks/30m  2,164
    images/L8/npy   44,879      masks/3m   2,164
    sickle_dataset_tabular.csv

**A 15x saving was found while doing this.** The archive carries two copies of
the imagery: `images/{SAT}/npy/` (209,039 `.npz`, what the dataloader reads)
and `images/{SAT}/tif/` (**2.99M raw per-band GeoTIFFs the reference code never
touches**). The first extraction attempt was writing all 3.2M files; skipping
the redundant tree cut it to 215k and the backfill then finished in 0.4 min.

### Images to look at — `results/previews/`

| File | What it shows |
|---|---|
| `contact_sheet.png` | 24 plots, paddy green / non-paddy red |
| `overview_<uid>.png` | one sample: S2 RGB across its season + all six mask bands decoded, with the plot outlined in red |
| `timeseries_<uid>.png` | the full S2 sequence for one plot |
| `modalities_<uid>.png` | S2 optical vs S1 radar vs L8, same plot |

Two things are immediately visible and both matter: **a plot occupies only a
tiny fraction of its 33x33 chip** (the "~15 pixels per plot" problem made
concrete), and **roughly 40% of mid-season frames are cloud-covered**.

---

## 3. Findings

### 3.1 Every claimed fact verified — 51 checks, 0 refuted

`scripts/phase0_verify.py`. Includes the 351-plot count (the paper says 388),
the unlabelled test split, clean plot-level splits, and the authors' training
config.

### 3.2 The six mask bands, decoded from evidence

Not documented anywhere available. Derived empirically, then independently
confirmed against the authors' own code.

| Band | Meaning | Evidence |
|---|---|---|
| 0 | `PLOT_ID` | 600/600 masks contain the sample's own plot id |
| 1 | crop code | one value per crop on own-plot pixels, 800 samples |
| 2/3/4 | sowing / transplanting / harvesting day | 45/45, 100/100, 45/45 |
| 5 | **yield / pixel count** | sums to tabular YIELD in **200/200** |

**Band 5 is a trap** — it is plot yield spread across pixels, so a per-pixel
regression target fits something that scales inversely with plot area. The
authors' `get_combined_yield` sums it per plot, confirming this.

**Band 0 is what makes Plot-Macro IoU implementable at all.**

### 3.3 Arm B is NOT VIABLE — decided, not deferred

Against a criterion fixed before computing it (>=3 classes, each >=5 validation
plots), **every threshold fails**. Beyond Paddy (29 val plots) only Banana (4)
and Coconut (3) have validation support. `artifacts/arm_b_viability.json`.

**Run count is 155, Arm A only.** The difficulty axis must be replaced —
`YEAR` and `RIVER_PART` are both present and viable, but using them means
building custom splits and re-implementing the adjacent-plot masking (now
possible via band 0). **That is your decision; the prompt forbids me choosing.**

### 3.4 The reproduction gate — and the finding it produced

> **Note:** a first twelve-cell sweep was discarded because two loops ran
> concurrently at different batch sizes and overwrote the same files, which at
> the time did not record the batch size. See `docs/REPRODUCTION_GATE.md`
> section 4a. The individual numbers below were each observed directly with the
> batch size passed explicitly, and stand. The full sweep was re-run cleanly.

Tolerance **±0.01 declared before any run, never widened.**

Final clean sweep: **12 cells, 9 PASS, median |Δ| 0.0012, max 0.0345.**

Best reproductions:

    [S2]_utae         0.3812 -> 0.3812   |Δ| 0.0000
    [L8]_utae         0.3812 -> 0.3812   |Δ| 0.0000
    [L8]_unet3d       0.7061 -> 0.7071   |Δ| 0.0009
    [S2]_unet3d       0.7173 -> 0.7192   |Δ| 0.0019  (at batch 16)
    [S2,L8,S1]_unet3d 0.8190 -> 0.8217   |Δ| 0.0027
    [S1]_utae         0.6010 -> 0.5983   |Δ| 0.0027
    [S1]_convlstm     0.4129 -> 0.4129   |Δ| 0.0000
    [L8]_convlstm     0.7663 -> 0.7661   |Δ| 0.0002

Trainable parameter counts match the released configs exactly. The harness is
correct.

**But results depend on batch size.** Same checkpoint, same data, same code:

| batch_size | `[L8]_unet3d` | Δ |
|---:|---:|---:|
| 32 | 0.7071 | 0.0009 |
| 16 | 0.6561 | 0.0500 |
| 8 | 0.6409 | 0.0652 |

**6.5 IoU points from batch size alone, at evaluation, with frozen weights.**

Mechanism, traced in the code: `pad_collate` pads the temporal axis to the
**per-batch maximum**; in `pastis_unet3d.forward` the pad mask is realigned to
the pooled temporal length by *truncation*
(`pad_mask[:, :final.shape[-1]]`) rather than by pooling. Whether masking
engages therefore depends on batch composition.

**The cleanest evidence:** the two models that reproduce *exactly* (Δ 0.0000)
are `[S2]_utae` and `[L8]_utae` — both degenerate, both best at **epoch 1** with
IoU 0.3812, both predicting one class everywhere. A model that ignores its
input is immune to padding. The models that actually use temporal structure are
the ones that move.

**Consequences:** any IoU reported on this benchmark without its evaluation
batch size is not reproducible; this study fixes `batch_size = 32` and records
it in every manifest; and all variant comparisons must hold it constant.

This was only found because the tolerance was declared in advance. Widening it
to make the first attempt "pass" would have hidden a 6.5-point sensitivity.

---

## 4. Built and tested

| Component | Status |
|---|---|
| `src/evaluation/metrics.py` | macro IoU, per-class IoU/F1/precision/recall, **Plot-Macro IoU**, plot-level bootstrap CI, paired seed deltas |
| `tests/` | **20 passing unit tests** |
| `scripts/make_subsets.py` | 30 nested plot subsets, nesting verified |
| `shims/rasterio.py` | tifffile-backed; Smart App Control blocks real GDAL |
| `reference_impl/` | official SICKLE code, cloned unmodified |
| `requirements.txt`, `.gitignore`, `README.md` | written |

**Subset composition is stable**: paddy fraction stays 0.398–0.426 across
50→315 plots, so the data-regime curve will not be confounded by class-balance
drift. That was worth checking before trusting the sweep.

---

## 5. Environment gotchas that cost time

- **rasterio is blocked by Windows Smart App Control**, consistently
  (3/3 attempts). Shimmed.
- **SciPy DLLs are intermittently blocked** by the same policy — a suite can
  fail then pass unchanged. Retry before debugging.
- **The discrete GPU powers down when idle.** Check `torch.cuda.is_available()`
  before long runs.

---

## 6. Next, in order

1. **Your decision:** the Arm B replacement axis — held-out YEAR, held-out
   RIVER_PART, or drop the difficulty axis entirely.
2. Dataloader + config system (the reference `SICKLE_Dataset` works; it needs
   wrapping with subset manifests and run manifests).
3. Time one training run on GPU, then budget the 155.
4. **Data-regime sweep (90 runs) — this is the paper.** Run it before the
   ladder; if the curves do not separate, that must surface in week one.
5. Ladder (65 runs), aggregation, statistics, figures.

The gate is cleared, so step 2 onward is unblocked.
