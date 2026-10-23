# SICKLE field-level audit

Reproduction and field-level audit of **SICKLE** (Sani et al., 2024, WACV),
a multi-sensor satellite benchmark of smallholder paddy fields in the
Cauvery Delta, Tamil Nadu, India; and a data-regime study of when temporal
modelling pays on it.

Every number in the manuscripts is regenerated from this repository's
artefacts by one command, and checked against them (`scripts/build_all.py`,
`scripts/verify_all.py`).

---

## Two studies in one repository

### 1. The audit: what a benchmark number says about an individual field

The released models are re-evaluated, the defects that affect reproduction
are measured, and every task is scored field by field as well as pooled.

- **Reproduction.** Of the 60 released models, all 48 regression checkpoints
  (yield and phenology dates) reproduce their reported errors and 9 of 12
  crop-type cells do.
- **Three defects** in the released code (`docs/MODEL_DEFECTS.md`):
  - evaluation of frozen weights depends on batch size, by up to 32.7 IoU
    points, because the temporal pad mask is realigned by truncation;
  - a zero denominator in the masked temporal mean gives NaN on 13 of 1,937
    training samples;
  - three of the four released UTAE crop-type baselines never trained.
- **The pooled score hides failing fields.** A pooled IoU of 0.8665 sits
  beside 4 of 36 validation fields below 0.5, and the same fields fail
  across independent training runs.
- **Yield.** None of the 24 released yield models is shown to beat a
  constant predictor. Their association with the reference lies between
  plots, not between seasons.
- **Label-free screening.** Seed disagreement and model confidence do not
  reliably find the crop-type fields that fail, and a pre-registered yield
  screen ranks error no better than chance.

Manuscript builders: `scripts/make_paper_ieee.py` (journal length),
`scripts/make_paper_ieee_conf.py` (conference length) and
`scripts/make_paper_rse.py` (Remote Sensing of Environment format). The
generated documents are not tracked.

### 2. The temporal complexity study (data-regime sweep)

Does the advantage of a satellite time series over a single acquisition
grow as more labelled plots are added? Ninety training runs (plot counts 50
to 315, five seeds) under an analysis protocol frozen before any cross-run
result was computed (`configs/sweep_analysis_protocol.yaml`).

- **Primary metric (pooled IoU):** the advantage is positive at every plot
  count but does not grow with it: slope +0.0045 IoU per doubling, 95 % CI
  −0.055 to +0.064. Verdict: **not supported (inconclusive)**.
- **Secondary metric (mean field IoU):** the advantage shrinks as plots are
  added: slope −0.082 per doubling, 95 % CI −0.156 to −0.008.

Results: `results/sweep/sweep_analysis.json`. This study is separate from
the audit papers. The ten-rung ablation it builds on is in
`docs/LADDER_RESULTS.md`.

---

## Data

The SICKLE data, trained weights and reference code belong to their authors
and are distributed by them on request. **None of it is included here.**
Per-field labels, and values derived from them, are kept out of version
control.

To reproduce, request SICKLE from its authors, then copy `.env.example` to
`.env` and set `SICKLE_DATA_ROOT` (the extracted dataset) and `SICKLE_RUNS`
(the released checkpoints). `docs/REPRODUCE.md` gives the full procedure
and environment; `docs/CHECKPOINT_INVENTORY.md` lists every checkpoint used,
with checksums.

## Repository

```
configs/     ladder rungs v1-v10, sweep and screening protocols, tolerances
src/         dataset, model, training, evaluation, runtime patches
scripts/     reproduction gate, analyses, figures, manuscript builders,
             build_all.py (rebuild everything) and verify_all.py (checks)
results/     tracked summaries and analysis outputs (no labels, no imagery)
docs/        defects, reproduction gate, references, ladder results,
             reproduction guide, commit map
tests/       unit tests
```

`docs/COMMIT_MAP.md` maps commit IDs recorded in run manifests before the
history rewrite of 2026-10-23 to their current IDs.

## Licence

The code in this repository is released under the MIT licence (see
`LICENSE`). The licence covers this code only, not SICKLE.
