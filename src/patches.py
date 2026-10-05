"""A correctness patch for `pastis_unet3d.UNet3D.forward`, applied at runtime.

WHY THIS EXISTS
---------------
The released `UNet3D.forward` ends with a masked temporal mean:

    pad_mask = pad_mask[:, :final.shape[-1]]   # truncate to the pooled length
    pad_mask = ~pad_mask
    out = (final * pad_mask).sum(dim=-1) / pad_mask.sum(dim=-1)

`pad_mask` is computed over the FULL sequence length, then realigned to the
pooled temporal length by **truncation** rather than by pooling. A timestep
that is entirely `pad_value` is treated as padding, so a sample whose EARLY
acquisitions are empty -- common here, because acquisitions start late in a
season and missing/cloudy frames are filled with zeros -- can have every entry
of the truncated mask marked "pad". The denominator is then **0** and the
output is NaN.

MEASURED ON THIS DATASET (batch_size 32, Sentinel-2):

    train:  13 / 1937 samples  (0.7%)  hit the 0/0
    val:     0 /  227 samples  (0.0%)

There is **no NaN guard anywhere in the reference implementation** (no
`isnan`, `isfinite` or `nan_to_num` in any file), so an unpatched training run
takes NaN loss on those samples, and NaN gradients from a single batch are
enough to destroy every weight in the model.

This is the same truncation-instead-of-pooling flaw that makes evaluation
batch-size dependent (docs/REPRODUCTION_GATE.md section 2). There it changes a
number by up to 6.5 IoU points; here it produces NaN outright.

Scope: this affects `model: unet3d` only. UTAE's structurally similar division
in `Temporal_Aggregator` sits under `mode == "mean"`, and this study runs
`agg_mode: att_group`, so that branch is not reached. ConvLSTM does
`(~pad_mask).sum(dim=1) - 1`, which for a fully padded sample yields -1 and
silently indexes the LAST timestep rather than failing -- wrong, but not NaN.
Neither is patched here, and neither should be described as fixed.

WHAT THE PATCH DOES
-------------------
`clamp` (the default and the only mode applied): the denominator is clamped to
a minimum of 1. For every sample where at least one timestep survives
truncation -- which is 99.3% of train and 100% of val -- the arithmetic is
**bit-identical** to the released code, so this cannot change any number that
was not already NaN. It therefore does not invalidate the reproduction gate.
For a fully-padded sample it yields 0 instead of NaN.

What this patch deliberately does NOT do: fix the truncation itself. Pooling
the pad mask to the output length would be the correct realignment, but it
would change the behaviour of the released checkpoints and break comparability
with the published benchmark. That belongs in the write-up as a finding, not in
a silent patch.

`reference_impl/` is never edited. The patch is installed at runtime and
`assert_patched()` records it in the run manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))

PATCH_ID = "unet3d_masked_mean_clamp_v1"
_INSTALLED = False


def _patched_forward(self, x, batch_positions=None):
    """Byte-for-byte the released forward, with one clamped denominator."""
    out = x.permute(0, 2, 1, 3, 4)
    pad_mask = None
    if self.pad_value is not None:
        pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)
        if self.zero_pad:
            out[out == self.pad_value] = 0

    en3 = self.en3(out)
    pool_3 = self.pool_3(en3)
    en4 = self.en4(pool_3)
    pool_4 = self.pool_4(en4)
    center_in = self.center_in(pool_4)
    center_out = self.center_out(center_in)
    concat4 = torch.cat([center_out, en4[:, :, :center_out.shape[2], :, :]], dim=1)
    dc4 = self.dc4(concat4)
    trans3 = self.trans3(dc4)
    concat3 = torch.cat([trans3, en3[:, :, :trans3.shape[2], :, :]], dim=1)
    dc3 = self.dc3(concat3)
    final = self.final(dc3)
    final = final.permute(0, 1, 3, 4, 2)          # B x C x H x W x T

    if self.pad_value is not None:
        if pad_mask.any():
            pad_mask = pad_mask[:, :final.shape[-1]]
            pad_mask = ~pad_mask
            denom = pad_mask.sum(dim=-1)
            # THE FIX. Released code divides by `denom`, which is 0 whenever a
            # sample has no surviving timestep after truncation -> NaN. Clamping
            # to 1 leaves every other sample's arithmetic unchanged.
            denom = denom.clamp(min=1)
            out = (final.permute(1, 2, 3, 0, 4)
                   * pad_mask[None, None, None, :, :]).sum(dim=-1) \
                / denom[None, None, None, :]
            out = out.permute(3, 0, 1, 2)
        else:
            out = final.mean(dim=-1)
    else:
        out = final.mean(dim=-1)
    return out


def install() -> str:
    """Install the patch. Idempotent. Returns the patch id."""
    global _INSTALLED
    if _INSTALLED:
        return PATCH_ID
    from models import pastis_unet3d
    pastis_unet3d.UNet3D.forward = _patched_forward
    _INSTALLED = True
    return PATCH_ID


def is_installed() -> bool:
    return _INSTALLED


def assert_patched() -> str:
    if not _INSTALLED:
        raise RuntimeError("patches.install() was not called before training")
    return PATCH_ID
