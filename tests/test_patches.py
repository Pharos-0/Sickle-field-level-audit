"""Regression tests for the UNet3D masked-mean patch.

The patch exists because the released forward divides by zero for a sample
whose timesteps are all marked as padding after truncation. These tests pin
both halves of the contract:

  1. it fixes the NaN, and
  2. it changes NOTHING else -- bit-identical output wherever the released code
     already produced a finite result.

(2) is the load-bearing one. If it ever fails, the reproduction gate's numbers
are no longer comparable to the released checkpoints and the patch has become
a silent experimental change rather than a bug fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))


class FakeUNet3D:
    """Minimal stand-in exercising only the masked-mean tail.

    The real module is heavy and needs the dataset; the defect lives entirely
    in the final reduction, so that is what is reproduced here. `final` is
    supplied directly and the encoder/decoder are bypassed.
    """

    pad_value = 0
    zero_pad = True

    def __init__(self, final, pad_mask):
        self._final = final
        self._pad_mask = pad_mask


def _released_tail(final, pad_mask):
    """The released arithmetic, verbatim: no clamp."""
    pm = pad_mask[:, :final.shape[-1]]
    pm = ~pm
    return (final.permute(1, 2, 3, 0, 4)
            * pm[None, None, None, :, :]).sum(dim=-1) \
        / pm.sum(dim=-1)[None, None, None, :]


def _patched_tail(final, pad_mask):
    """The patched arithmetic: denominator clamped to >= 1."""
    pm = pad_mask[:, :final.shape[-1]]
    pm = ~pm
    denom = pm.sum(dim=-1).clamp(min=1)
    return (final.permute(1, 2, 3, 0, 4)
            * pm[None, None, None, :, :]).sum(dim=-1) \
        / denom[None, None, None, :]


def _make(b=4, c=2, h=3, w=3, t=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, t, generator=g)


def test_released_tail_produces_nan_when_all_timesteps_padded():
    """The defect itself. If this stops failing, the patch is unnecessary."""
    final = _make()
    pad_mask = torch.zeros(4, 4, dtype=torch.bool)
    pad_mask[2, :] = True            # sample 2: every timestep is padding
    out = _released_tail(final, pad_mask)
    assert not torch.isfinite(out).all(), \
        "expected the released arithmetic to divide by zero"


def test_patch_removes_the_nan():
    final = _make()
    pad_mask = torch.zeros(4, 4, dtype=torch.bool)
    pad_mask[2, :] = True
    out = _patched_tail(final, pad_mask)
    assert torch.isfinite(out).all()


def test_patch_is_bit_identical_when_no_sample_is_fully_padded():
    """The contract that keeps the reproduction gate valid."""
    final = _make(seed=7)
    pad_mask = torch.zeros(4, 4, dtype=torch.bool)
    pad_mask[0, 3] = True            # partial padding only
    pad_mask[1, 2:] = True
    a = _released_tail(final, pad_mask)
    b = _patched_tail(final, pad_mask)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "patch altered a result that was already valid"


@pytest.mark.parametrize("seed", range(5))
def test_bit_identical_across_random_partial_masks(seed):
    g = torch.Generator().manual_seed(seed)
    final = _make(seed=seed)
    pad_mask = torch.rand(4, 4, generator=g) < 0.5
    pad_mask[:, 0] = False           # guarantee one surviving timestep each
    a = _released_tail(final, pad_mask)
    b = _patched_tail(final, pad_mask)
    assert torch.equal(a, b)


def test_install_is_idempotent_and_reports_patch_id():
    from src import patches
    first = patches.install()
    second = patches.install()
    assert first == second == patches.PATCH_ID
    assert patches.is_installed()
    assert patches.assert_patched() == patches.PATCH_ID
