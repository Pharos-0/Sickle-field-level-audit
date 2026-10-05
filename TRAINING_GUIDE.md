# Training guide

Everything needed to run the ladder, and the things that will bite you if you
don't read it.

---

## 0. Before anything

```bash
python scripts/check_configs.py
```

Resolves all ten rungs, prints what each one changes, and asserts the ladder's
invariants (batch size 32 everywhere, `num_classes` 2, eval split is `val`,
every rung differs from its predecessor, `data.root` exists). It touches
neither torch nor the dataset, so it runs anywhere and costs seconds. A config
error caught here is free; the same error caught in epoch 40 is not.

Then prove the pipeline moves:

```bash
bash scripts/smoke_test.sh v1 v2
```

Two training batches and a full validation pass per rung. **A smoke run is not
a result** and its metrics are meaningless — it exists to prove data loads,
the model steps, both metric families compute and a checkpoint writes.

---

## 1. Data

| | |
|---|---|
| `data.root` | `D:/FPD/sickledataset extracted` |
| Contents | 209,039 `.npz` · 6,492 masks · `sickle_dataset_tabular.csv` |
| CSV md5 | `e804c1c165e3f55d03da8305a380ff77` |

**Do not point `data.root` at `D:/FPD/WACV 2024`.** That is the *toy* subset —
19,859 npz, 651 masks — with a **different** CSV (md5 `0cb87468...`). It will
load without complaint and silently evaluate on a different set.

`D:/FPD/sickle_data` is a redundant npz-only copy and can be deleted to
reclaim ~3 GB.

---

## 2. Running

```bash
bash scripts/train.sh v1 0                 # one rung, one seed
bash scripts/train.sh v1 0 --epochs 10     # override
SEEDS="0 1 2" bash scripts/run_ladder.sh   # the whole ladder, sequentially
```

PowerShell equivalents: `.\scripts\train.ps1 -Config v1 -Seed 0`.

Outputs land in `outputs/<task>/<version>_seed<N>/`:

| file | what |
|---|---|
| `manifest.json` | config hash, git commit **and dirty flag**, device, batch size, param count |
| `config_resolved.json` | the fully merged config, so a run is reconstructible |
| `trainlog.json` | per-epoch train loss, LR, and every val metric |
| `best.json` | selected epoch and its full metric set |
| `checkpoint_best.pth.tar` | weights at the selected epoch |
| `best_per_plot_iou.json` | per-plot IoU at the selected epoch |

### Run them one at a time

`scripts/run_ladder.sh` is sequential on purpose. The first reproduction-gate
sweep had to be **thrown away** because two loops ran concurrently at different
batch sizes and overwrote the same filenames
(`docs/REPRODUCTION_GATE.md` §4a). Don't recreate that.

---

## 3. Things that will bite you

### Batch size is not a free knob

**Fixed at 32 everywhere. Do not vary it.** Evaluation on this benchmark is
batch-size dependent: `pad_collate` pads the temporal axis to the *per-batch
maximum*, and `pastis_unet3d.forward` realigns the pad mask by truncation
(`pad_mask[:, :final.shape[-1]]`) rather than by pooling, so whether masking
engages depends on batch composition. Measured on **frozen weights**:

| batch_size | `[L8]_unet3d` val IoU |
|---:|---:|
| 32 | 0.7071 |
| 16 | 0.6561 |
| 8 | 0.6409 |

6.5 IoU points from batch size alone. `check_configs.py` enforces 32.

### U-Net3D cannot take a one-frame sequence

It applies two `MaxPool3d(2, 2)` over time, so **T must be ≥ 4**; T=1 gives
`Calculated output size: (32x0x16x16)` and dies. This is why v1's single
acquisition is *repeated* to 4 constant timesteps rather than sliced to one.
A constant sequence carries zero temporal information while keeping
architecture, parameter count and collate identical to v2 — so `v2 − v1` is
the value of temporal information, not the cost of swapping models.

Describe v1 as **single-acquisition**, never as "a 2D baseline".

### Watch for degenerate runs

If a run ends with **paddy IoU 0.000**, it is predicting one class everywhere.
Three of the four *released* UTAE baselines fail this way (best at epoch 1;
`[S2]_utae` and `[L8]_utae` both score exactly 0.3812 with paddy IoU 0). The
trainer prints `DEGENERATE` and `WARNING: best epoch is 1` for exactly this.
**Report a degenerate run as degenerate, not as a result.**

### Two metrics, and they disagree

Every epoch records both:

* **pixel-pooled macro IoU** — comparable to the published benchmark
* **Plot-Macro IoU** — every plot weighted equally, 95% CI from resampling
  *plots*, which is the study's independent spatial unit

On the released checkpoints the gap (pixel minus plot) has a median of **+0.18**
across the eleven cells measured, ranges from +0.01 to +0.30, goes *negative*
in one cell, and **reorders which model wins**. Report both. Always.

**Selection is on pixel-pooled macro IoU**, declared in `src/train.py` before
any run, so the ladder stays comparable to the released baselines and
Plot-Macro stays an out-of-selection measurement.

### The test split is unlabelled

All 206 test rows are blank in `PLOT_ID`/`CROP`/`PADDY_BIN` and no masks exist.
`split_frame()` raises if you ask for it. **Every number this project produces
is a validation number** and is not comparable to the paper's published test
table.

### Environment

* **rasterio is blocked by Windows Smart App Control** on the dev machine. The
  tifffile-backed shim in `shims/rasterio.py` supplies the one API the
  reference uses and raises on anything else rather than faking it.
* **SciPy DLLs are intermittently blocked** by the same policy — a suite can
  fail then pass unchanged. Retry before debugging.
* **The discrete GPU powers down when idle**, and `nvidia-smi` needs admin on
  this machine. `utils.pick_device()` checks at run time and the manifest
  records what it got. CPU is viable for v1 but slow for the full-sequence
  rungs.
* `num_workers: 0` on Windows. Raise it only on Linux.

---

## 4. Cost

CPU timings from this machine, Sentinel-2 only:

| rung | sequence | relative cost |
|---|---|---|
| v1 | 4 constant frames | baseline |
| v2–v4 | full S2 sequence | several× v1 |
| v5–v10 | 2–3 sensors, full sequence | substantially more |

**Time one epoch before budgeting the ladder.** 10 rungs × 3 seeds × 100 epochs
is not achievable on CPU; either get CUDA working or cut seeds and epochs and
say so in the write-up.

---

## 5. Honesty rules for anything written from these runs

Carried over from the study specification and not negotiable:

* Attribution is **within this experimental system** — "controlled component
  attribution", never "causal".
* No architecture is proposed. Don't claim one.
* Never write "first study", "no previous work" or "novel architecture".
* **If a run was not executed, say NOT RUN. If it failed, say FAILED.** Do not
  interpolate, and do not quietly drop a rung that broke.
* 351 plots, phrased precisely: *351 unique plots producing 2,370 season-level
  samples and ~209k satellite observations; the independent spatial unit for
  generalisation is the plot.* Never "209k independent samples".
