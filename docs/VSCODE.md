# Running this project in VS Code — start to finish

VS Code is **not faster**. It is the same interpreter on the same CPU; an
editor does not change arithmetic. Use it because it is easier to read and
because the task menu removes the need to remember commands.

What actually changes the runtime is the GPU — see §7.

---

## 1. Install (once)

1. **VS Code** — <https://code.visualstudio.com/>
2. Open the folder: **File → Open Folder → `D:\FPD\sickle-project`**
3. VS Code will offer the recommended extensions (from `.vscode/extensions.json`).
   Accept, or install manually:
   - **Python** (`ms-python.python`) — required
   - **Jupyter** (`ms-toolsai.jupyter`) — for the EDA notebook
   - **YAML** (`redhat.vscode-yaml`) — for the ladder configs

---

## 2. Point it at the interpreter (once)

The venv lives **outside** this folder, at

    D:\Final Project M.Tech\.venv\Scripts\python.exe

so VS Code cannot find it automatically. `.vscode/settings.json` already sets
it. To confirm:

`Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** →
paste the path above.

Bottom-right of the window should then show that interpreter. If it shows a
different Python, nothing will import — `torch` is only installed in this venv.

---

## 3. Prove the setup works (30 seconds)

`Ctrl+Shift+P` → **Tasks: Run Task** → pick in order:

| task | what it proves | time |
|---|---|---|
| **0. Check configs** | all 10 rungs resolve, invariants hold, data root exists | instant |
| **1. Run tests** | 29 tests pass | ~2 s |
| **2. Smoke test** | data loads, model steps, metrics compute, checkpoint writes | ~3 min |

If task 0 fails with *"data.root does not exist"*, fix the path in
`configs/base.yaml`. It must be the **full** extraction:

    D:/FPD/sickledataset extracted

**Not** `D:/FPD/WACV 2024` — that is the toy subset with a different CSV. It
loads silently and changes the evaluation set.

A smoke run is **not a result**. It trains on two batches to prove the pipeline
moves, and `aggregate_ladder.py` refuses to report it.

---

## 4. Get the timing table before committing hours

**Tasks: Run Task → 3. Benchmark all rungs**

Measures every rung's real tensor shape, forward+backward cost and loading cost,
then prints minutes-per-epoch and hours-per-100-epochs, and writes
`results/bench_rungs_nw8.json`. Takes a few minutes and tells you what the full
ladder would actually cost on this machine. Do this before step 5.

---

## 5. Train

**Tasks: Run Task → 4. Train one rung** → pick the rung and seed.

Or the whole ladder: **5. Run the whole ladder** (sequential; see §6 for why).

To step through the code with breakpoints instead, use **Run and Debug**
(`Ctrl+Shift+D`) → *Train: pick a rung*. Debugging is slower; use it to inspect,
not to produce results.

Monitor a run from a second terminal (``Ctrl+` ``):

```powershell
& 'D:\Final Project M.Tech\.venv\Scripts\python.exe' scripts/watch.py v1 0
```

or **Tasks: Run Task → 7. Watch a running rung**.

### Closing VS Code

A task runs **inside** VS Code's terminal and **dies when VS Code closes**.
For a multi-hour run that must survive, start it detached from PowerShell
instead:

```powershell
cd 'D:\FPD\sickle-project'
$py = 'D:\Final Project M.Tech\.venv\Scripts\python.exe'
Start-Process $py -ArgumentList '-u','-m','src.train','--config','v1','--seed','0' `
  -RedirectStandardOutput 'outputs\logs\v1_seed0.log' `
  -RedirectStandardError  'outputs\logs\v1_seed0.err' `
  -WindowStyle Hidden
```

That process is independent of VS Code, this chat, and any terminal.

---

## 6. Rules that are not optional

- **One run at a time.** Two multi-hour CPU jobs halve each other. The first
  reproduction-gate sweep in this project was thrown away because concurrent
  loops overwrote each other's files (`REPRODUCTION_GATE.md` §4a).
- **Never change `batch_size`.** Fixed at 32. Evaluation on this benchmark is
  batch-size dependent — 6.5 IoU points on frozen weights.
  `check_configs.py` enforces it.
- **Check for `DEGENERATE`.** Paddy IoU 0 means one class predicted everywhere.
  Three of the four *released* UTAE baselines fail this way. Report it as
  degenerate, not as a result.
- **Lower `num_workers` to 0** in `configs/base.yaml` if workers crash. It costs
  ~6.5× loading speed but always works.

---

## 7. The thing that would actually make this fast

    NVIDIA GeForce RTX 5060 Laptop GPU
    status: "this hardware device is not connected to the computer" (Code 45)

The card is in the machine and the NVIDIA services run, but Windows cannot see
the device, so `torch.cuda.is_available()` is `False` and everything runs on
CPU. Usual causes: the discrete GPU disabled in BIOS, graphics forced to
"Integrated only" / a MUX setting, or a driver state needing reinstall.

Check with:

```powershell
Get-PnpDevice -Class Display | Format-List FriendlyName, Status, ProblemDescription
```

Fixing this is worth far more than any software change here. Afterwards verify:

```powershell
& 'D:\Final Project M.Tech\.venv\Scripts\python.exe' -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, set `device: cuda` in `configs/base.yaml` (or leave
`auto`) and raise `batch_size`? **No** — leave batch size at 32. Raise epochs
or seeds instead.

---

## 8. Results

**Tasks: Run Task → 6. Aggregate results table**

Prints every rung under both metrics, the rung-to-rung deltas, and flags
`NOT RUN`, `INCOMPLETE (n/100 epochs)` and `DEGENERATE`. It computes no delta
across an incomplete or degenerate rung, and it excludes smoke runs. Writes
`results/ladder_crop_type.csv`.
