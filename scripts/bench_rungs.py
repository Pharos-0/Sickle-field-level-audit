"""Measure the real per-rung cost, so the schedule is budgeted not guessed.

For each rung: the temporal length it actually receives, one forward+backward,
and the data-loading cost at the configured num_workers. Epoch time is then
    61 train batches x (io + compute) + validation
measured on THIS machine while anything else is also running, so treat the
numbers as an upper bound rather than a best case.

    python scripts/bench_rungs.py
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import torch
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src import utils, patches
from src.dataset import build_loader
from src.model import build_model, count_params
from src.loss import build_loss

NB_TRAIN = 61
ROWS = []

def bench(rung: str, nw: int, device: str = "cpu"):
    cfg = utils.load_config(rung)
    cfg["training"]["num_workers"] = nw
    utils.set_seed(0)
    loader, df = build_loader(cfg, "train", "train", seed=0)
    model, _ = build_model(cfg, device); model.train()
    crit = build_loss(cfg, None, "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    it = iter(loader)
    inp, mask = next(it)                      # warm up / pay worker startup
    T = max(v[0].shape[1] for v in inp.values())

    # data loading only
    t0 = time.time(); k = 0
    for _ in range(3):
        try: next(it); k += 1
        except StopIteration: break
    io = (time.time() - t0) / max(k, 1)

    # compute only
    def _to(x):
        if torch.is_tensor(x): return x.to(device)
        if isinstance(x, dict): return {k: _to(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return type(x)(_to(c) for c in x)
        return x
    inp = _to(inp)
    y = mask[cfg["data"]["task"]].long().to(device)

    # WARM UP BEFORE TIMING. Without this the first rung measured absorbs CUDA
    # context creation and cuDNN autotuning and reports a GPU slower than the
    # CPU -- v1 measured 8.32 s on GPU against 0.52 s on CPU purely from this.
    N_WARMUP, N_TIMED = 3, 5
    for _ in range(N_WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(inp), y); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N_TIMED):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(inp), y); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()   # CUDA is async; timing without this lies
    comp = (time.time() - t0) / N_TIMED

    ep = NB_TRAIN * (io + comp)
    ROWS.append(dict(rung=rung, device=device, sensors=",".join(sorted(cfg["satellites"])),
                     model=cfg["model"]["name"], T=int(T),
                     params=count_params(model),
                     io_s=round(io, 2), compute_s=round(comp, 2),
                     epoch_s=round(ep, 0), hours_100ep=round(ep * 100 / 3600, 1)))
    print(f"{rung:5s} {ROWS[-1]['sensors']:10s} {cfg['model']['name']:9s} "
          f"T={T:3d} params={count_params(model):>9,} io={io:5.2f}s "
          f"compute={comp:5.2f}s -> {ep/60:5.1f} min/epoch, "
          f"{ep*100/3600:5.1f} h/100ep", flush=True)

if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    patches.install()
    print(f"device={device}  num_workers={nw}  threads={torch.get_num_threads()}  "
          f"cuda={torch.cuda.is_available()}\n", flush=True)
    for r in [f"v{i}" for i in range(1, 11)]:
        try:
            bench(r, nw, device)
        except Exception as e:
            print(f"{r:5s} FAILED {type(e).__name__}: {e}", flush=True)
            ROWS.append(dict(rung=r, status=f"FAILED {type(e).__name__}"))
    (PROJ / "results").mkdir(exist_ok=True)
    (PROJ / "results" / f"bench_rungs_nw{nw}_{device}.json").write_text(
        json.dumps(ROWS, indent=2), encoding="utf8")
    tot = sum(r.get("hours_100ep", 0) for r in ROWS)
    print(f"\nTOTAL one seed, 100 epochs, all 10 rungs: {tot:.0f} h "
          f"({tot/24:.1f} days).  Three seeds: {tot*3/24:.1f} days.")
