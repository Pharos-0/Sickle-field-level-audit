"""Dataset construction for the ladder.

The authors' `SICKLE_Dataset` is used unmodified and wrapped, rather than
forked. Everything this study needs that the reference does not provide is
added around it:

  * `temporal.mode == "single"`  collapse the time axis to one acquisition,
                                which is what makes v1 a genuine no-temporal
                                floor rather than a short sequence
  * `data.subset`               restrict the TRAIN split to a plot manifest,
                                for the data-regime sweep
  * `sampling.plot_balanced`    sample plots, not pixels, so training agrees
                                with how the study measures

Labels, for the avoidance of doubt: the reference does `mask[1] -= 1` and then
`mask[1][mask[1] > 1] = 1`, so **paddy is class 0** and non-paddy is class 1.
`src/evaluation/metrics.py` encodes the same convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "shims"))        # tifffile-backed rasterio
sys.path.insert(0, str(PROJ / "reference_impl"))

from utils.dataset import SICKLE_Dataset          # noqa: E402
from utils import utae_utils                      # noqa: E402


class SingleDateWrapper(tdata.Dataset):
    """Reduce each sample to the information content of ONE acquisition.

    v1 has to be a real floor for the temporal question. The obvious
    implementation -- slice the time axis to length 1 -- does not work and the
    reason is architectural, not incidental: `pastis_unet3d` applies two
    `MaxPool3d(kernel=2, stride=2)` over the temporal axis, so T=1 collapses to
    T=0 and the forward pass raises
    `Given input size: (32x1x32x32). Calculated output size: (32x0x16x16)`.
    The architecture cannot represent a one-frame sequence at all.

    So instead the chosen acquisition is REPEATED to `repeat_to` timesteps.
    The sequence is constant, therefore it carries zero temporal information --
    every temporal difference, gradient and attention weight over it is
    identically zero -- while architecture, parameter count, collate behaviour
    and tensor rank stay exactly as in v2. `v2 - v1` is then the value of
    temporal *information*, not the cost of swapping models, which is what the
    study is asking.

    This is a deliberate design decision with a visible cost: v1 is not
    literally a 2D model, and it should be described as "single-acquisition"
    rather than "no temporal model". Stating it that way is accurate; calling
    v1 a 2D baseline would not be.

    The frame is chosen deterministically. `median_index` takes the middle
    valid acquisition, which is near peak season for a paddy cycle and does not
    depend on the RNG, so the choice is not a hidden source of seed variance.
    """

    STRATEGIES = ("median_index", "first", "last")
    MIN_TIMESTEPS = 4          # two temporal halvings: 4 -> 2 -> 1

    def __init__(self, base: tdata.Dataset, strategy: str = "median_index",
                 repeat_to: int = 4):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}, "
                             f"expected one of {self.STRATEGIES}")
        if repeat_to < self.MIN_TIMESTEPS:
            raise ValueError(
                f"temporal.repeat_to={repeat_to} is below the architectural "
                f"minimum {self.MIN_TIMESTEPS}: pastis_unet3d halves the "
                f"temporal axis twice, so anything shorter collapses to zero "
                f"length and the forward pass fails.")
        self.base = base
        self.strategy = strategy
        self.repeat_to = repeat_to

    def __len__(self) -> int:
        return len(self.base)

    def _pick(self, sample: np.ndarray, dates: np.ndarray) -> int:
        """Index of the frame to keep.

        A frame that is entirely zero is padding the reference inserted for a
        missing acquisition, so those are excluded before choosing. If every
        frame is empty there is nothing to choose and index 0 is returned; the
        sample is degenerate either way and the model sees zeros.
        """
        arr = np.asarray(sample)
        nonempty = [i for i in range(arr.shape[0]) if np.any(arr[i] != 0)]
        if not nonempty:
            return 0
        if self.strategy == "first":
            return nonempty[0]
        if self.strategy == "last":
            return nonempty[-1]
        return nonempty[len(nonempty) // 2]

    def __getitem__(self, idx):
        data, mask = self.base[idx]
        out = {}
        for sat, (sample, dates) in data.items():
            keep = self._pick(sample, dates)
            n = self.repeat_to
            s = np.repeat(np.asarray(sample)[keep:keep + 1], n, axis=0)
            if dates is None:
                d = None
            else:
                # The positional encoding is fed these day offsets. Repeating
                # the DATE as well as the image keeps the sequence genuinely
                # constant; incrementing it would leak an ordering signal into
                # a baseline that is supposed to carry none.
                d = np.repeat(np.asarray(dates)[keep:keep + 1], n, axis=0)
            out[sat] = (s, d)
        return out, mask


def _load_subset_plots(name: str) -> set[int]:
    path = PROJ / "artifacts" / "subsets" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"subset manifest {path} not found. "
            f"Run `python scripts/make_subsets.py` first.")
    obj = json.loads(path.read_text(encoding="utf8"))
    plots = obj.get("plot_ids", obj) if isinstance(obj, dict) else obj
    return {int(p) for p in plots}


def load_tabular(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    csv = root / cfg["data"]["tabular"]
    if not csv.exists():
        raise FileNotFoundError(f"tabular csv not found at {csv}")
    return pd.read_csv(csv)


def split_frame(cfg: dict, split: str) -> pd.DataFrame:
    """Rows for one split, with the study's filters applied.

    The TEST split is unlabelled -- every row is blank in PLOT_ID, CROP and
    PADDY_BIN and no masks exist for it -- so asking for it is a mistake worth
    failing loudly on rather than silently evaluating on empty labels.
    """
    df = load_tabular(cfg)
    if split == "test":
        raise ValueError(
            "The SICKLE test split is UNLABELLED (all 206 rows blank in "
            "PLOT_ID/CROP/PADDY_BIN, no masks on disk). This study reports "
            "validation numbers only; see README 'Limitations'.")

    sub = df[df.SPLIT == split].reset_index(drop=True)
    if cfg["data"]["task"] != "crop_type":
        sub = sub[sub.YIELD > 0].reset_index(drop=True)

    name = cfg["data"].get("subset")
    if name and split == cfg["data"]["train_split"]:
        keep = _load_subset_plots(name)
        sub = sub[sub.PLOT_ID.isin(keep)].reset_index(drop=True)
    return sub


def build_dataset(cfg: dict, split: str, phase: str) -> tdata.Dataset:
    df = split_frame(cfg, split)
    if len(df) == 0:
        raise RuntimeError(f"split {split!r} is empty after filtering")

    sats = {k: dict(v) for k, v in cfg["satellites"].items()}
    for v in sats.values():
        v["img_size"] = tuple(v["img_size"])

    ds = SICKLE_Dataset(
        df=df,
        data_dir=str(cfg["data"]["root"]),
        satellites=sats,
        ignore_index=cfg["data"]["ignore_index"],
        transform=bool(cfg["augmentation"]["enabled"]) if phase == "train" else None,
        actual_season=cfg["data"].get("actual_season", False),
        phase=phase,
    )

    if cfg["temporal"]["mode"] == "single":
        ds = SingleDateWrapper(ds, cfg["temporal"]["single_date_strategy"],
                               repeat_to=cfg["temporal"].get("repeat_to", 4))
    elif cfg["temporal"]["mode"] != "full":
        raise ValueError(f"temporal.mode must be 'single' or 'full', "
                         f"got {cfg['temporal']['mode']!r}")
    return ds, df


class PadCollate:
    """Picklable wrapper around the reference `pad_collate`.

    A lambda cannot be pickled, and Windows DataLoader workers use `spawn`, so
    a lambda collate silently forces `num_workers=0`. That matters a great deal
    here: measured on this machine, ~204 s of a ~250 s epoch is data loading on
    a single thread while 27 others idle. Making this a module-level callable
    is what allows workers to be used at all.
    """

    def __init__(self, pad_value):
        self.pad_value = pad_value

    def __call__(self, batch):
        return utae_utils.pad_collate(batch, pad_value=self.pad_value)


def _plot_balanced_weights(df: pd.DataFrame) -> torch.Tensor:
    """One weight per SAMPLE, inversely proportional to its plot's frequency.

    Plots contribute wildly different numbers of season-level samples. Uniform
    sampling therefore trains disproportionately on a few plots, while the
    study's statistical unit is the plot. This equalises expected plot exposure.
    """
    counts = df.PLOT_ID.value_counts()
    w = df.PLOT_ID.map(lambda p: 1.0 / float(counts[p])).to_numpy()
    return torch.as_tensor(w, dtype=torch.double)


def build_loader(cfg: dict, split: str, phase: str, seed: int = 0):
    ds, df = build_dataset(cfg, split, phase)
    bs = cfg["training"]["batch_size"]

    sampler = None
    shuffle = phase == "train"
    if phase == "train" and cfg["sampling"]["plot_balanced"]:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = tdata.WeightedRandomSampler(
            weights=_plot_balanced_weights(df),
            num_samples=len(df), replacement=True, generator=g)
        shuffle = False

    nw = int(cfg["training"]["num_workers"])
    loader = tdata.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
        num_workers=nw,
        collate_fn=PadCollate(cfg["data"]["pad_value"]),
        # Workers are respawned every epoch by default, and spawn is expensive
        # on Windows. Keeping them alive matters because loading dominates.
        persistent_workers=bool(nw),
        prefetch_factor=(4 if nw else None),
        drop_last=False,
    )
    return loader, df


def class_weights_from_train(cfg: dict, df: pd.DataFrame) -> torch.Tensor | None:
    """Inverse-frequency weights over the binary label, from TRAIN rows only.

    Computed from the tabular PADDY_BIN column rather than by scanning masks:
    it is the same quantity at plot level and avoids a full pass over 209k
    arrays just to set two numbers. Never computed from val.
    """
    spec = cfg["loss"].get("class_weights")
    if not spec or spec == "null":
        return None
    if spec != "inverse_frequency":
        raise ValueError(f"unknown loss.class_weights {spec!r}")

    col = "PADDY_BIN" if "PADDY_BIN" in df.columns else None
    if col is None:
        raise RuntimeError("PADDY_BIN not in the tabular csv; cannot derive "
                           "class weights without scanning masks")
    # paddy == class 0
    n_paddy = int((df[col] == 1).sum())
    n_other = int((df[col] != 1).sum())
    if n_paddy == 0 or n_other == 0:
        return None
    total = n_paddy + n_other
    w = torch.tensor([total / (2.0 * n_paddy), total / (2.0 * n_other)],
                     dtype=torch.float32)
    return w
