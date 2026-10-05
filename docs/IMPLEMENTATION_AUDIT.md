# IMPLEMENTATION AUDIT — PHASE 0 and PHASE 1

**Date:** 2026-10-05 · **Project:** `D:\FPD\sickle-project`
**Status:** PHASE 0 and PHASE 1 complete. No models implemented yet, by design.

Every statement below was derived from the artefacts by a script in `scripts/`
and is reproducible. Nothing here is assumed.

---

## 1. Verification of the master prompt's "verified facts"

`scripts/phase0_verify.py` → `artifacts/phase0_verification.csv`

    CONFIRMED 51   REFUTED 0   RECORDED 9

**Every claimed fact was confirmed**, including: 2,370 samples; 351 unique
PLOT_IDs; train 1,937 samples / 315 plots; val 227 / 36; test 206 / **0**; all
206 test rows blank in PLOT_ID, CROP and PADDY_BIN; 0 plots straddling splits;
the crop distribution; the 262 + 105 = 351 + 16 rotation reconciliation; 78
released run directories with the per-task breakdown; and the authors' config
(`num_classes 2`, `img_size [32,32]`, 100 epochs, batch 32, lr 0.1, seed 0,
`ignore_index -999`).

**Newly recorded**

    npz per modality   L8 44,879 | S1 37,657 | S2 126,503   (= 209,039)
    [S2]_unet3d        best val IoU 0.7173 | final val 0.6601
                       final train 0.8709  | FINAL-EPOCH GAP 0.2108

---

## 2. Dataset layout — verified

    sickle_dataset-002.zip   6.34 GB, 3,428,250 entries, 7.0 GB unpacked
      images/{L8,S1,S2}/...  209,039 .npz
      masks/{3m,10m,30m}/*.tif  2,995,937 .tif
      sickle_dataset_tabular.csv

**Masks are indexed by `UNIQUE_ID`.** `masks/10m/` holds 2,164 files; 0 have no
tabular row, and **exactly 206 tabular rows have no mask — the test split.**
The test split has neither labels nor masks. This independently confirms that
the 36 validation plots are the entire evaluation set.

Mask grid sizes: **3m → 108x108, 10m → 33x33, 30m → 12x12**, float64,
Deflate-compressed GeoTIFF with 6 bands.

---

## 3. THE SIX MASK BANDS — decoded from evidence

This was not documented anywhere available; it was derived empirically and each
line carries its test result.

| Band | Meaning | Evidence |
|---|---|---|
| **0** | **`PLOT_ID`** | contains the sample's own PLOT_ID in **600/600** sampled masks |
| **1** | **crop code** | restricted to own-plot pixels (band0 == PLOT_ID), exactly **one** value per crop across 800 samples |
| **2** | `SOWING_DAY` | 45/45 |
| **3** | `TRANSPLANTING_DAY` | 100/100 |
| **4** | `HARVESTING_DAY` | 45/45 |
| **5** | **`YIELD` spread per pixel** | **sum over own-plot pixels == tabular YIELD in 200/200** |

**Crop codes** (own-plot pixels, 800 samples):

    1 Paddy · 2 Banana · 3 Coconut · 4 Banyan · 5 Sugarcane · 6 Palm tree
    7 Groundnut · 8 Eucalyptus · 9 Tamarind tree · 10 Mixed plantation
    11 Cotton · 15 Betelwine

### Two consequences that affect implementation

**(a) A chip contains neighbouring plots.** Band 0 is what makes the authors'
adjacent-plot masking possible, and it is what makes **per-plot IoU
computable** for the Plot-Macro IoU metric in Section 10.2 of the prompt.
Without band 0 that metric could not be implemented at all.

**(b) Band 5 is NOT a per-pixel yield.** It is total plot yield divided by the
plot's pixel count — it *sums* to the plot yield. Training a regressor on band 5
directly would fit a quantity that scales inversely with plot area. The yield
target must be the **sum over the plot**, not the per-pixel value.

---

## 4. ARM B — DECIDED: NOT VIABLE

`scripts/phase1_inspect.py` → `artifacts/arm_b_viability.json`

Pre-specified criterion, fixed before computing and not lowered afterwards:
**>= 3 retained classes, each with >= 5 validation plots.**

| MIN_PLOTS_PER_CLASS | K | min val plots in a retained class | Meets? |
|---:|---:|---:|---|
| 3 | 9 | 0 | fails |
| 5 | 7 | 0 | fails |
| 10 | 6 | 0 | fails |
| 15 | 3 | 3 | fails |
| 20 | 2 | 4 | fails |

**No threshold passes.** Beyond Paddy (29 val plots), only Banana (4) and
Coconut (3) have any validation support. `arm_b_status: NOT_VIABLE`.

**Consequence for the study:** the difficulty axis must be replaced, not
dropped. `YEAR` (2018/2019/2020) and `RIVER_PART` (5 sub-basins) are both
present in train and val and are candidates. **Both require building custom
splits, which forfeits the authors' adjacent-plot masking — that masking would
have to be re-implemented using band 0, which is now possible.** This decision
is deferred to the user per prompt Section 5.

**Run accounting is therefore: 65 ladder + 90 sweep = 155 runs, Arm A only.**

---

## 5. Environment

- `tifffile` + `imagecodecs` **were required and installed**. Neither PIL nor
  OpenCV can read these masks: PIL fails on the compression, and OpenCV refuses
  >4 channels (`Unsupported number of channels: 6`). This must go in
  `requirements.txt`.
- Working interpreter: **`.venv` inside this project** (torch 2.11.0+cu128).
  It previously borrowed `D:\Final Project M.Tech\.venv`, which belongs to a
  **different** project -- deleting or moving that one would have broken this
  one, and the dependency was hard-coded in 19 places. The project is now
  self-contained; `scripts/*.sh` and `scripts/*.ps1` resolve the interpreter as
  `$SICKLE_PY` -> `<project>/.venv` -> `python` on PATH, and the VS Code config
  uses `${workspaceFolder}/.venv`.
- Disk: D: has 128 GB free; the zip is unpacked lazily (read from the archive)
  so no extraction has been performed beyond the tabular CSV.

---

## 6. What could NOT be verified

| Item | Status |
|---|---|
| The paper's claim of **388 plots** | **Contradicted** by the released tabular (351). Unresolved — the released artefact is what the code must use |
| L8 released val IoU vs the paper (70.6% vs 56.04%) | **Unexplained**, +14.6 points. Not investigated yet |
| Whether a cloud/quality field exists (prompt V8a) | **Not yet determined.** The `.npz` image contents have not been inspected; V8a remains blocked until they are |
| Image `.npz` internal structure | **Not yet inspected** — next task before any dataloader |
| Whether the authors' reported numbers are best-epoch or final-epoch | **Unknown**, and it matters: for `[S2]_unet3d` the two differ by 5.7 IoU points |

---

## 7. Blockers

1. **V8a (cloud/quality masking) is blocked** until the `.npz` contents are
   inspected. If no quality field exists, the prompt requires reporting
   "cannot be implemented from the verified dataset representation" rather than
   fabricating one.
2. **The Arm B replacement axis is a user decision** (prompt Section 5
   explicitly forbids choosing unilaterally).

---

## 8. Next steps, in order

1. Inspect `.npz` image structure — keys, shapes, dtypes, temporal length,
   band order, missing-observation representation. **Decides V8a.**
2. Reproduce one released cell from the authors' checkpoint
   (`[S2]_unet3d crop_type`) — **PHASE 5, the gate.** Nothing proceeds until a
   released number is recovered within a tolerance declared in advance.
3. Build the metrics module, including Plot-Macro IoU using band 0, with unit
   tests.
4. Build plot-level subset manifests for the sweep (nested subsets).
5. Only then implement models.
