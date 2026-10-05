"""Losses.

Only what the ladder actually uses. `ignore_index` matters more than usual
here: pixels belonging to plots from another split are set to the ignore value
by the reference dataset (adjacent-plot masking), and counting them would leak
neighbouring-split labels into the training signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_loss(cfg: dict, class_weights: torch.Tensor | None, device: str):
    kind = cfg["loss"]["type"]
    ignore = cfg["data"]["ignore_index"]

    if kind != "cross_entropy":
        raise ValueError(f"loss.type {kind!r} not supported")

    w = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=w, ignore_index=ignore)
