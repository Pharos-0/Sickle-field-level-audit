# Defects in the released SICKLE implementation

Three defects found while building this study, all in the handling of the
temporal padding mask. Each is stated with how it was found, what was measured,
and what this project does about it. Nothing here is inferred from reading the
code alone; every claim has a number attached or is explicitly marked as not
measured.

`reference_impl/` is **never edited**. Fixes are installed at runtime from
`src/patches.py` and recorded in every run manifest.

---

## The common root

All three come from the same place. `utae_utils.pad_collate` pads the temporal
axis to the **maximum sequence length within each batch**, and a timestep that
is entirely `pad_value` is subsequently treated as padding. Satellite revisit
counts differ per sample, so batch composition determines how much padding each
sample receives. Downstream, each architecture realigns or consumes that mask
in a way that is not safe for every input it can actually receive.

---

## 1. Evaluation results depend on batch size

**Severity: invalidates any reported number that omits its batch size.**

In `pastis_unet3d.forward` the pad mask is realigned to the pooled temporal
length by **truncation**:

```python
pad_mask = pad_mask[:, :final.shape[-1]]   # "match new temporal length"
```

Truncation is not the inverse of pooling. Whether masking engages therefore
depends on which samples share a batch.

Measured on a **frozen** released checkpoint — same weights, same data, same
code, only batch size changed:

| batch_size | `[L8]_unet3d` val macro IoU | abs delta vs released |
|---:|---:|---:|
| **32** (the authors' value) | **0.7071** | 0.0009 |
| 16 | 0.6561 | 0.0500 |
| 8 | 0.6409 | 0.0652 |

**A 6.5 IoU-point swing from batch size alone, at evaluation, with no training.**

Corroborating detail: the two cells that reproduce *exactly* (|Δ| = 0.0000) are
`[S2]_utae` and `[L8]_utae`, both degenerate and predicting one class
everywhere. A model that ignores its input is immune to how the input is
padded.

**What this project does:** fixes `batch_size = 32` everywhere, enforces it in
`scripts/check_configs.py`, and writes it into every output filename and
payload. Full detail in `REPRODUCTION_GATE.md` §2.

---

## 2. NaN from a zero denominator in the masked temporal mean

**Severity: destroys training. Silent.**

Same function, the final reduction:

```python
pad_mask = pad_mask[:, :final.shape[-1]]
pad_mask = ~pad_mask
out = (final * pad_mask).sum(dim=-1) / pad_mask.sum(dim=-1)
```

`pad_mask` is computed over the **full** sequence and then truncated to the
pooled length. A sample whose **early** acquisitions are empty — common here,
because acquisitions start late in a season and missing or cloudy frames are
filled with zeros — can have every entry of the *truncated* mask marked as
padding. The denominator is then **0**, and the output is NaN.

### Measured (Sentinel-2, batch_size 32)

| split | samples hitting 0/0 | share |
|---|---:|---:|
| train | **13 / 1937** | 0.7% |
| val | **0 / 227** | 0.0% |

Two consequences follow directly from that table:

* **Training is affected and evaluation is not.** This is why the twelve-cell
  reproduction gate came back clean while the first training attempt produced
  NaN loss on roughly five batches in every twelve.
* **The reproduction gate's numbers are unaffected by the fix**, because no
  validation sample ever reaches the bad branch.

### It is not an optimiser problem

The NaN count was **identical (5 batches) at learning rates 0.1, 0.01, 0.001
and 0.0001**, which rules out divergence. Hooking every module showed non-finite
output appearing with **freshly initialised weights and zero optimiser steps**,
on specific batches only, with inputs verified free of NaN and Inf.

### There is no guard anywhere

`grep -rn "isnan\|isfinite\|nan_to_num" reference_impl/` returns **nothing**. An
unpatched training run takes a NaN loss on those samples, and NaN gradients from
a single batch are enough to destroy every weight in the model.

### The fix

`src/patches.py` clamps the denominator to a minimum of 1. Verified on 12 real
batches against the released arithmetic with identical weights:

```
batches bit-identical where released was finite : 9
batches that DIFFER where released was finite   : 0
batches NaN before, finite after (fixed)        : 3
```

**The patch cannot change a result that was not already NaN.** Pinned by
`tests/test_patches.py`, which fails if bit-identity is ever lost.

**What is deliberately NOT fixed:** the truncation itself. Pooling the pad mask
to the output length is the correct realignment, but it would change the
behaviour of the released checkpoints and break comparability with the
published benchmark. That belongs in the write-up as a finding, not in a silent
patch.

---

## 3. ConvLSTM indexes the last timestep when a sample is fully padded

**Severity: silently wrong output. NOT MEASURED.**

`convlstm.py` line 169:

```python
last_positions = (~pad_mask).sum(dim=1) - 1
```

For a fully padded sample this is `-1`, which in Python indexing selects the
**last** element rather than raising. The model returns a hidden state derived
from padding and reports no error.

**This has not been measured on this dataset and is not patched.** It is
recorded because it is the same class of defect, and because anyone extending
this study to ConvLSTM rungs needs to check it first.

---

## Scope: what is *not* affected

`utae.py` contains a structurally identical unguarded division:

```python
out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
```

but it sits inside `Temporal_Aggregator` under `mode == "mean"`. This study runs
`agg_mode: att_group`, so **that branch is never reached** and UTAE is not
affected by defect 2.

This matters for an attractive but wrong inference. Three of the four released
UTAE baselines never trained (best at epoch 1; `[S2]_utae` and `[L8]_utae` both
score exactly 0.3812 with paddy IoU exactly 0.000). It would be tidy to explain
that with NaN poisoning — **but the code path does not support it**, and the
cause of the UTAE degeneracy remains **unexplained**.

---

## Summary

| # | Defect | Effect | Measured | Patched |
|---|---|---|---|---|
| 1 | pad mask truncated, not pooled | up to **6.5 IoU** swing from batch size | yes | no — pinned at 32 instead |
| 2 | zero denominator in masked mean | **NaN**, destroys training | yes — 13/1937 train, 0/227 val | yes, bit-identity verified |
| 3 | ConvLSTM `-1` index | silently wrong hidden state | **no** | no |

Defects 1 and 2 are the same line of code seen from two directions.
