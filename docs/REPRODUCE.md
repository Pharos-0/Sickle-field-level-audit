# Reproducing this study

Exact commands, in dependency order. Every number in the manuscript comes
from a file one of these produces.

Prefix all commands with the project's own interpreter:

```
cd D:/FPD/sickle-project
.venv/Scripts/python.exe <script> [args]
```

On Linux/macOS use `.venv/bin/python`. The project must not borrow an
interpreter from another project — see the header of `requirements.txt`.

---

## 0. Environment

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

torch **must** be installed first, from the CUDA index, pinned. Installing
`timm` afterwards once replaced a working `2.11.0+cu128` with `2.14.1+cpu`
silently. The pin in `requirements.txt` was verified in a clean environment
on 2026-10-11 to survive the full install.

**Known environment failure.** On a machine with Windows Smart App Control
active, a *freshly installed* torch raises
`OSError: [WinError 4551] An Application Control policy has blocked this
file` on `torch/lib/c10.dll`. This was reproduced both under
`AppData\Local\Temp` and on `D:`, so it is the policy reacting to newly
written binaries, not a path problem. Everything that does not import torch
still runs; `tests/test_patches.py` skips with a clear message.

| Need | Requires torch? |
|---|---|
| `verify_all.py`, `independent_check.py` | no (stdlib / numpy) |
| manuscript, figures, reference table | no |
| `regression_gate.py`, `phase5_gate.py`, `triage_eval.py`, `yield_screen.py` inputs | **yes** |

**Recorded environment (2026-10-14).** The exact package versions are in
`requirements-lock.txt`; every checkpoint, with its SHA-256, is in
`docs/CHECKPOINT_INVENTORY.csv` (`scripts/checkpoint_inventory.py`).

| | |
|---|---|
| OS | Windows 11, build 26200 |
| Python | 3.12.10 |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| GPU | NVIDIA RTX 5060 Laptop, 8 GB, compute capability 12.0, driver 617.14 |
| RAM | 32 GB |

Known limits on this hardware: run one GPU training job at a time (two
concurrent jobs exhausted memory on 2026-10-11, see
`outputs/_interrupted_runs/README.md`); PDF export needs Microsoft Word;
absolute paths live in `configs/base.yaml` (`data.root`) and, for the
released weights, the `SICKLE_RUNS` environment variable (see `.env.example`) read by
the inventory script (also read by `scripts/plot_level_eval.py` and
`scripts/regression_gate.py`). The prototype is expected beside this folder as
`../sickle-guard`, or wherever `SICKLE_GUARD_ROOT` points.

---

## 1. Checks that need no GPU and no torch

```
.venv/Scripts/python.exe scripts/verify_all.py
.venv/Scripts/python.exe scripts/independent_check.py
.venv/Scripts/python.exe -m pytest -q
```

`independent_check.py` imports nothing from `src/` and re-derives the
manuscript's headline numbers from raw artefacts with its own numpy
implementations. If it and `verify_all.py` ever disagree, trust neither until
the difference is explained — they share no helper code.

## 2. Rebuild every deliverable from one run

```
.venv/Scripts/python.exe scripts/build_all.py
```

Figures → manuscript → reference table → PDF → idempotence probe → all three
check layers → hash manifest in `docs/deliverable_hashes.json`. Stops at the
first failure and refuses to report success if any check fails. Add
`--skip-pdf` where MS Word is unavailable.

## 3. Regenerating experiment artefacts

These need torch and the SICKLE data at the `data.root` in
`configs/base.yaml`. They are slow and the repository already contains their
outputs; only re-run them to verify.

**Crop-type reproduction gate** (one released cell):

```
.venv/Scripts/python.exe scripts/phase5_gate.py --run "[S1]_unet3d" --task crop_type --batch-size 32 --device cpu
```

**Regression gate**, which also writes the per-field-season predictions the
yield experiment consumes:

```
.venv/Scripts/python.exe scripts/regression_gate.py --task crop_yield --run "[L8]_unet3d" --batch-size 32 --device cpu
```

Repeat for every non-`_debug` run. `_debug` runs are 2-epoch smoke tests and
are excluded from every count in the paper.

**Crop-type screening signals** (3 seeds of v5; writes
`results/triage/per_field_signals_v5.csv`):

```
.venv/Scripts/python.exe scripts/triage_eval.py --config v5 --seeds 0 1 2 --device cpu --num-workers 0
```

Use `--num-workers 0` when anything else is running: an 8-worker run once
exhausted memory and killed a concurrent job.

**Batch-size diagnosis of the unreproduced crop-type cells:**

```
.venv/Scripts/python.exe scripts/diagnose_gate_misses.py --device cpu --batch-sizes 8 16 64
```

Batch sizes 1 and 4 are omitted by default. On the first cell both moved the
score far *away* from the released value, and each cost roughly 45 minutes of
CPU on ConvLSTM, so they cannot answer whether any batch size closes the gap.

## 4. The yield-reliability screening experiment

The protocol is frozen in `configs/yield_screen_protocol.yaml`, written
before any result was computed. `yield_screen.py` loads it and aborts if the
expected checkpoint count is not met, rather than quietly running on a
partial ensemble.

```
.venv/Scripts/python.exe scripts/yield_screen.py
.venv/Scripts/python.exe scripts/make_figure_yield_screen.py
```

Outputs `results/triage/yield_screen.json`,
`results/triage/yield_per_field.csv`, and
`results/figures_ieee/fig_yield_screen.png`.

Independent unit is the **plot**, not the field-season: 41 field-seasons come
from 14 plots, up to four seasons each. Thresholds are leave-one-plot-out and
every interval bootstraps plots.

---

## Determinism

`src/utils.set_seed` fixes Python, numpy and torch RNGs per run. Every run
writes a `manifest.json` recording the resolved config and its content hash,
the git commit and dirty flag, the device, the effective and micro batch
sizes, the seed, and the trainable parameter count. Evaluation is full
precision at batch size 32 throughout; evaluation on this benchmark is
batch-size dependent (Section V-A), so a result without its batch size is not
reproducible.

Bootstrap procedures take an explicit seed and default to 0.

## What reproduction here does and does not establish

Running the above confirms that the manuscript agrees with the artefacts in
this repository. It is **not** independent reproduction: no second person and
no second machine has rebuilt the trained checkpoints from the released code
and data. That remains outstanding and the manuscript says so.
