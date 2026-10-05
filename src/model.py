"""Model construction.

The architectures are the authors' own (`reference_impl/models/`), reached
through their `Fusion_model`. This study proposes no architecture and must not
be read as proposing one; the contribution is a measurement.

`Fusion_model` expects a flat `argparse.Namespace` with the keys the released
`conf.json` files carry. The ladder's YAML is nested, so the single job here is
to flatten it faithfully and to fail loudly on anything the reference would
otherwise silently default.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "shims"))
sys.path.insert(0, str(PROJ / "reference_impl"))

SUPPORTED = ("unet3d", "utae", "convlstm")


def to_namespace(cfg: dict, device: str) -> Namespace:
    """Flatten a resolved ladder config into the reference's CFG object.

    Key names are taken from the released `conf.json` verbatim so that a config
    built here is interchangeable with one the authors shipped.
    """
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    sats = {k: dict(v) for k, v in cfg["satellites"].items()}
    for v in sats.values():
        v["img_size"] = list(v["img_size"])

    primary = list(sats.keys())[0]
    ns = Namespace(
        model=m["name"],
        encoder_widths=list(m["encoder_widths"]),
        decoder_widths=list(m["decoder_widths"]),
        out_conv=list(m["out_conv"]),
        str_conv_k=m["str_conv_k"],
        str_conv_s=m["str_conv_s"],
        str_conv_p=m["str_conv_p"],
        agg_mode=m["agg_mode"],
        encoder_norm=m["encoder_norm"],
        n_head=m["n_head"],
        d_model=m["d_model"],
        d_k=m["d_k"],
        padding_mode=m["padding_mode"],
        device=device,
        num_workers=t["num_workers"],
        batch_size=t["batch_size"],
        lr=t["lr"],
        epochs=t["epochs"],
        num_classes=d["num_classes"],
        ignore_index=d["ignore_index"],
        pad_value=d["pad_value"],
        satellites=sats,
        primary_sat=primary,
        img_size=list(sats[primary]["img_size"]),
        task=d["task"],
        exp_name=d["task"],
        actual_season=d.get("actual_season", False),
        use_augmentation=bool(cfg["augmentation"]["enabled"]),
        cache=False,
        debug=False,
        resume="",
        run_id="",
        data_dir=str(d["root"]),
    )
    return ns


def build_model(cfg: dict, device: str):
    name = cfg["model"]["name"]
    if name not in SUPPORTED:
        raise ValueError(f"model.name {name!r} not supported; "
                         f"expected one of {SUPPORTED}")
    from utils import model_utils
    ns = to_namespace(cfg, device)
    model = model_utils.Fusion_model(ns).to(device)
    return model, ns


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
