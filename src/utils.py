"""Config resolution, seeding, device selection and run manifests.

The ladder in `configs/` is cumulative: each `vN.yaml` declares
`defaults: [- vN-1]` and overrides exactly one block. `load_config` walks that
chain and deep-merges it, so a rung's YAML shows only what it changes while the
resolved config carries everything.

Nothing here reads the dataset. Keeping config resolution importable without
torch or the reference implementation means `scripts/check_configs.py` can
validate the whole ladder on any machine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJ = Path(__file__).resolve().parents[1]
CONFIGS = PROJ / "configs"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge. `over` wins. Lists replace, they do not append.

    Lists replace deliberately: `satellites` and `encoder_widths` are both
    lists-or-dicts whose meaning is positional, so appending would silently
    build a config nobody wrote.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config {path}")
    return yaml.safe_load(path.read_text(encoding="utf8")) or {}


def load_config(name: str, _seen: tuple[str, ...] = ()) -> dict:
    """Resolve `configs/<name>.yaml` through its `defaults:` chain."""
    name = name.replace(".yaml", "")
    if name in _seen:
        raise ValueError(f"circular defaults: {' -> '.join(_seen + (name,))}")

    raw = _read_yaml(name)
    parents = raw.pop("defaults", []) or []
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for p in parents:
        p = p.lstrip("- ").strip() if isinstance(p, str) else p
        merged = _deep_merge(merged, load_config(p, _seen + (name,)))

    cfg = _deep_merge(merged, raw)
    cfg["_resolved_from"] = name
    return cfg


def data_root() -> Path:
    """The canonical dataset root, from configs/base.yaml.

    Single source of truth. The standalone scripts (the reproduction gate,
    plot-level evaluation, subset construction) used to carry their own
    hard-coded copy of this path, which is how a project ends up silently
    reading two different datasets.
    """
    root = Path(load_config("base")["data"]["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"data.root does not exist: {root}\n"
            f"Set it in configs/base.yaml. It must be the FULL extraction "
            f"(209,039 npz, 6,492 masks, csv md5 e804c1c165e3f55d03da8305a380ff77) "
            f"-- not 'WACV 2024', which is the toy subset with a different csv.")
    return root


def config_hash(cfg: dict) -> str:
    """Stable short hash of a resolved config, for run identity."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG this project can reach, and ask cuDNN to be deterministic.

    Determinism is not guaranteed on CUDA for every op, so run manifests record
    the seed and the device rather than claiming bit-exact reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def pick_device(requested: str = "auto") -> str:
    """Resolve the device. 'auto' prefers CUDA when it is actually available.

    The discrete GPU on the development machine powers down when idle, so CUDA
    can be present at import and gone later. Always call this rather than
    assuming, and record what it returned.
    """
    import torch
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# run manifest
# --------------------------------------------------------------------------

def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=PROJ,
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def build_manifest(cfg: dict, seed: int, device: str, extra: dict | None = None) -> dict:
    """Everything needed to say what produced a number.

    `git_dirty` is recorded honestly: a run made against uncommitted changes is
    not reproducible from its commit hash alone and the manifest must say so
    rather than implying a clean tree.
    """
    man = {
        "version": cfg.get("project", {}).get("version"),
        "description": cfg.get("project", {}).get("description"),
        "resolved_from": cfg.get("_resolved_from"),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "device": device,
        "batch_size": cfg["training"]["batch_size"],
        "batch_size_note": (
            "Evaluation on this benchmark is batch-size dependent: pad_collate "
            "pads the temporal axis to the per-batch maximum. A result without "
            "its batch size is not reproducible. See docs/REPRODUCTION_GATE.md"
        ),
        "data_root": cfg["data"]["root"],
        "task": cfg["data"]["task"],
        "satellites": sorted(cfg["satellites"].keys()),
        "temporal_mode": cfg["temporal"]["mode"],
        "model": cfg["model"]["name"],
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        man["torch"] = torch.__version__
        man["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    if extra:
        man.update(extra)
    return man


def run_dir(cfg: dict, seed: int) -> Path:
    """Where a run writes. A subset run goes somewhere else entirely.

    The data-regime sweep trains the SAME rung on different plot subsets. If
    those shared a directory with the full-data ladder run they would silently
    overwrite it -- 109 hours of completed work -- and nothing would say so.
    Sweep runs therefore live under `sweep/<subset>/`.
    """
    v = cfg.get("project", {}).get("version", "unknown")
    base = PROJ / cfg["runtime"]["out_dir"] / cfg["data"]["task"]
    subset = cfg["data"].get("subset")
    d = (base / "sweep" / str(subset) / f"{v}_seed{seed}" if subset
         else base / f"{v}_seed{seed}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf8")


def completed_seed_runs(task: str, rung: str, proj: Path | None = None):
    """Seed directories for one rung whose training actually finished.

    An interrupted run still leaves `best.json` and `best_per_plot_iou.json`
    on disk, because both are rewritten every time the validation score
    improves. Those artefacts are real but they describe a model that was
    stopped early, and averaging one into a rung alongside completed seeds
    moves the rung without anything in the output saying so. This happened
    in practice: an out-of-memory kill left a 2-epoch v1 seed scoring 0.415
    macro IoU beside three 100-epoch seeds scoring about 0.63.

    `scripts/aggregate_ladder.py` was believed to guard against this and did
    not: it labelled an unfinished run in its status column while still
    averaging it in, until 2026-10-12. It now uses the same rule. Every
    consumer of these directories must apply it, which is why it lives
    here rather than being repeated.

    Returns (kept, dropped) as lists of (seed_dir, best_json_dict). A run
    with no `complete` key predates the flag and is treated as NOT complete,
    because the safe reading of a missing guarantee is that it is absent.
    """
    proj = proj or PROJ
    kept, dropped = [], []
    for d in sorted((proj / "outputs" / task).glob(f"{rung}_seed*")):
        f = d / "best.json"
        if not f.exists():
            dropped.append((d, None))
            continue
        try:
            b = json.loads(f.read_text(encoding="utf8"))
        except (OSError, ValueError):
            dropped.append((d, None))
            continue
        (kept if b.get("complete") is True else dropped).append((d, b))
    return kept, dropped


def completed_per_plot_iou(task: str, rung: str, proj: Path | None = None):
    """Paths to `best_per_plot_iou.json` for seeds that finished training.

    Thin wrapper over `completed_seed_runs` so that every figure, table and
    statistic in the paper selects seeds by the same rule. Returns
    (paths, dropped_names).
    """
    kept, dropped = completed_seed_runs(task, rung, proj)
    paths = [d / "best_per_plot_iou.json" for d, _ in kept]
    return [p for p in paths if p.exists()], [d.name for d, _ in dropped]


def completed_field_iou(rung: str, proj: Path | None = None,
                        task: str = "crop_type"):
    """Per-seed field IoU for a rung's completed seeds.

    Field IoU is the IoU of the crop(s) a field holds
    (src/evaluation/metrics.py, own_class_iou_from_confusion), read from the
    per-field confusion matrices written by scripts/field_confusions.py. It
    replaces the trainer's best_per_plot_iou.json (macro IoU) for every
    per-field statistic in the papers; see the metrics module docstring for
    why. Raises if the confusion file is missing or covers different seeds
    from the ones that finished training, so a figure cannot silently mix
    seed sets. Returns (list of {plot_id: field_iou} per seed, seeds).
    """
    import json as _json
    import numpy as _np
    from src.evaluation.metrics import own_class_iou_from_confusion
    proj = proj or Path(__file__).resolve().parents[1]
    f = proj / "results" / "field_confusions" / f"{rung}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run scripts/field_confusions.py "
                                f"--rung {rung}")
    fc = _json.loads(f.read_text(encoding="utf8"))
    kept, _ = completed_seed_runs(task, rung, proj)
    want = sorted(int(d.name.rsplit("seed", 1)[1]) for d, _ in kept)
    have = sorted(int(s) for s in fc["per_seed"])
    if have != want:
        raise RuntimeError(f"{f.name} covers seeds {have}; completed seeds "
                           f"are {want}. Re-run scripts/field_confusions.py.")
    out = []
    for s in have:
        rows = fc["per_seed"][str(s)]
        out.append({float(k): own_class_iou_from_confusion(_np.asarray(r["cm"]))
                    for k, r in rows.items()})
    return out, have
