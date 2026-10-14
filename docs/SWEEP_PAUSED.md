# Data-regime sweep — paused 2026-10-11 17:35, how to resume

Paused on your instruction so the GPU could go to the ablation seeds.
Nothing was deleted. `scripts/run_sweep.py` is resumable: it skips any cell
whose `best.json` says `complete: true`, so resuming picks up exactly where
this left off.

## Resume with

```bash
cd "D:/FPD/sickle-project" && nohup .venv/Scripts/python.exe -u scripts/run_sweep.py > logs_sweep_restart.log 2>&1 &
```

Do not run it while the ladder seeds are still going. Running both at once
is what exhausted memory earlier today.

## State at the pause

90 runs planned (6 plot counts x 3 configs x 5 seeds).

| | count |
|---|---:|
| complete, will be skipped on resume | 34 |
| incomplete, will be re-trained from scratch | 6 |
| not yet started | 50 |

The six incomplete cells:

| cell | epochs reached |
|---|---|
| `plots_200_seed002/v2_radar_seed2` | 2 of 100 |
| `plots_250_seed002/v1_seed2` | 4 of 100 |
| `plots_250_seed002/v2_radar_seed2` | not started |
| `plots_250_seed002/v2_seed2` | not started |
| `plots_315_seed002/v1_seed2` | 2 of 100 |
| `plots_315_seed002/v2_seed2` | not started |

Three of those were truncated earlier today when I ran extra trainings and
twelve CPU evaluations alongside the sweep and ran the machine out of
memory. `src/train.py` has no resume logic, so each restarts at epoch 1
rather than continuing from a partial state — the results are not
corrupted, the time is simply spent again.

`plots_200_seed002/v2_radar_seed2` had reached 12 epochs of its restart
when this pause took effect, so it starts over once more.

## What is running instead

`scripts/ladder_power.py` paired the ablation seeds across rungs and found
that only two steps are still open at three seeds: v2 to v3, which would
need about 152 seeds, and v3 to v4, which would need about 8. So the GPU is
on **v3 and v4 at seeds 3, 4, 5, 6 and 7** — the only extra seeds that
change an answer.

They run seed-major, v3 then v4 at each seed, because the analysis is
paired: a v3 seed without its matching v4 seed adds nothing. Stopping part
way therefore leaves complete pairs.

Log: `logs_ladder_seeds.log`.
