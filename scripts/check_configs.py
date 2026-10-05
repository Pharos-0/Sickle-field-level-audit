"""Validate the whole ladder without touching the dataset or torch.

Resolves every vN.yaml through its defaults chain and asserts the ladder's
invariants. Run this before any training: a config error found here costs
seconds, the same error found in epoch 40 costs hours.

    python scripts/check_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils import load_config, config_hash   # noqa: E402

LADDER = [f"v{i}" for i in range(1, 11)]
FAIL = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAIL.append(msg)


def main() -> int:
    cfgs = {}
    for name in LADDER:
        try:
            cfgs[name] = load_config(name)
        except Exception as e:
            FAIL.append(f"{name}: failed to resolve -- {e}")
    if FAIL:
        for f in FAIL:
            print("FAIL", f)
        return 1

    print(f"{'rung':6s}{'sensors':16s}{'temporal':10s}{'model':10s}"
          f"{'aug':5s}{'loss':18s}{'bs':>4s}  hash")
    print("-" * 82)
    for n in LADDER:
        c = cfgs[n]
        print(f"{n:6s}{','.join(sorted(c['satellites'])):16s}"
              f"{c['temporal']['mode']:10s}{c['model']['name']:10s}"
              f"{str(c['augmentation']['enabled'])[:4]:5s}"
              f"{str(c['loss']['class_weights'] or 'none'):18s}"
              f"{c['training']['batch_size']:4d}  {config_hash(c)}")

    # --- invariants the ladder depends on -----------------------------------
    for n, c in cfgs.items():
        check(c["training"]["batch_size"] == 32,
              f"{n}: batch_size is {c['training']['batch_size']}, must be 32 "
              f"(evaluation is batch-size dependent; see REPRODUCTION_GATE.md)")
        check(c["data"]["num_classes"] == 2, f"{n}: num_classes must be 2")
        check(c["data"]["eval_split"] == "val",
              f"{n}: eval_split must be val, the test split is unlabelled")
        check(c["project"]["version"] == n,
              f"{n}: project.version is {c['project']['version']!r}")
        check(Path(c["data"]["root"]).exists(),
              f"{n}: data.root does not exist: {c['data']['root']}")

    # each rung must differ from its predecessor, or it measures nothing
    for a, b in zip(LADDER, LADDER[1:]):
        check(config_hash(cfgs[a]) != config_hash(cfgs[b]),
              f"{b} is identical to {a} -- the rung changes nothing")

    # v1 must be the no-temporal floor; v2 must restore time and change ONLY that
    check(cfgs["v1"]["temporal"]["mode"] == "single", "v1 must be single-date")
    check(cfgs["v2"]["temporal"]["mode"] == "full", "v2 must be full temporal")
    for k in ("satellites", "model", "augmentation", "loss", "sampling"):
        check(cfgs["v1"][k] == cfgs["v2"][k],
              f"v2 changes {k} as well as temporal mode -- "
              f"the temporal effect would not be isolated")

    print("-" * 82)
    if FAIL:
        for f in FAIL:
            print("FAIL", f)
        print(f"\n{len(FAIL)} FAILURES")
        return 1
    print(f"all {len(LADDER)} rungs resolve; invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
