# The paper

> **Correction, 2026-10-12.** The per-field figures in this document used IoU averaged over both classes, which caps any imperfect single-crop field at 0.5 (35 of the 36 validation fields hold one crop). Scored on the crop each field holds, the median field is 0.957 (not 0.672) and 4 of 36 fields fall below 0.50 (not 12). The current figures are in `results/field_distributions.json` and the papers; this record is left as written.


What is publishable from this work, stated after a literature search rather
than before one.

---

## What is NOT the contribution

**The per-field vs pixel-pooled metric finding is already published.**

> Wang, Berman, Rannen-Triki, Torr, Tuia, Tuytelaars, Van Gool, Yu, Blaschko.
> *Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and
> Evaluation of Fine-grained Intersection over Union.* **NeurIPS 2023.**

They show per-dataset mIoU is biased toward large objects, propose fine-grained
and worst-case mIoUs, and frame it as model and dataset auditing — benchmarked
on 15 networks across 12 natural and aerial datasets. Per-object vs per-pixel
accuracy assessment is also established in agricultural remote sensing.

So *"apply plot-level IoU to SICKLE"* is an application of a known method and
is **not** a novelty claim. The metric work stays in the paper as **applied
methodology with a citation**, not as the contribution.

---

## Working title

> **What Does a Smallholder Crop Benchmark Measure? A Reproducibility and
> Sensitivity Audit of SICKLE**

---

## The four claims, and what backs each

### C1 — Four defects make the published numbers irreproducible

| defect | measurement |
|---|---|
| Evaluation depends on **batch size** | `[L8]_unet3d`: 0.7071 / 0.6561 / 0.6409 at batch 32 / 16 / 8 — **6.5 IoU points on frozen weights**. `pad_collate` pads the temporal axis to the per-batch maximum; the pad mask is realigned by truncation, not pooling. |
| **Unguarded division by zero** | NaN logits on **13/1937** training samples, 0/227 validation. No `isnan`/`isfinite`/`nan_to_num` anywhere in the reference repository. |
| **Three of four released UTAE baselines never trained** | best epoch 1; `[S2]_utae` and `[L8]_utae` both score exactly 0.3812 with paddy IoU **0.000**. |
| The date tasks' **"MAPE" is MAE ÷ 183** | a constant denominator, not a percentage error, computed per pixel. Verified exact on **9/9** released runs across all three date tasks. |

Plus: **16 of 21 crop labels have zero validation plots**, so only binary paddy
classification is evaluable on this benchmark.

**This is the strongest claim. It is specific, checkable, and not pre-empted.**

### C2 — Five of nine modelling choices are indistinguishable from noise

A 10-rung cumulative ablation at 3 seeds, 30 runs, 109 h. Each rung changes
exactly one factor.

Survive: temporal information **+0.1198**, Sentinel-1 radar **+0.1860** —
both 4–10× the seed-to-seed spread.

Inconclusive: augmentation, the third sensor, model capacity, optimiser
schedule, plot-balanced sampling.

**Two of these looked like results at one seed** (augmentation −0.0496,
capacity −0.0098) and vanish at three. Model complexity bought nothing while
consuming ~75% of the compute.

### C3 — Per-field failure is systematic, not random

Seed-to-seed correlation of per-field IoU **+0.53 to +0.58**; **11 of 12**
fields scoring below 0.5 do so in *every* independent run.

Wang et al. measure bias *in a metric*. They do not test whether the same
objects fail across independent training runs. **This is the part that makes
selective automation feasible** — automate reliable fields, route the rest to
human verification — and it is the most likely novel element here.

### C4 — The audit extends to regression, where the field is forced

Wang et al. are segmentation-only. For yield and phenology dates the field is
not a metric choice: mask band 5 stores **yield ÷ pixel count**, so a
per-pixel yield error is meaningless, and a harvest date is one date for a
field.

Across four tasks, 64 released models:

| task | aggregate | median field | p90 | max |
|---|---:|---:|---:|---:|
| crop type | IoU 0.8665 | **0.672** | — | 33% below 0.5 |
| crop yield | MAE 551.70 kg | 405.58 | 1225.27 | 1792.46 |
| harvesting date | MAE 6.77 d | 4.87 d | 15.99 d | 20.87 d |
| transplanting date | MAE 3.49 d | 2.21 d | 8.11 d | 16.16 d |

**The direction is task-dependent**, which is the interesting part: for crop
type the median field is *worse* than the aggregate; for the regression tasks
the median field is *better* but the tail is much heavier — roughly one
field-season in ten is off by more than two weeks on harvest date.

---

## Why anyone should care — the operational hook

India's **YES-TECH** programme uses satellite yield estimation for PMFBY crop
insurance across **344 districts in 12 states**, with a minimum 30% weightage
to modelled yield and **no upper cap** (Madhya Pradesh uses 100%). The official
116-page manual specifies validation at **Gram Panchayat** level — explicitly
"the smallest administrative unit" — and across all 116 OCR'd pages there is
no requirement to validate below it. Estimates "shall not be entertained in any
court of law", are RTI-exempt, and are not published.

**This motivates the work. It is not a result of it**, and the paper must say
so: the audit measures SICKLE, a research benchmark, not YES-TECH's
operational models.

### The three per-field decisions, and where each belongs in the paper

All four audited tasks correspond to a decision made about an individual
field, which is why per-field error is the relevant quantity for each.

| decision | the per-field quantity | measured here |
|---|---|---|
| **Insurance payout** (PMFBY / YES-TECH) | this field-season's **yield** | MAPE 32.74%; median absolute error 405.58 kg, max 1792.46 |
| **Procurement scheduling** (when to open mandis, allocate transport and storage) | this field's **harvest date** | aggregate MAE 6.77 d; **p90 15.99 d, max 20.87 d** |
| **Irrigation timing** (when water is needed in the delta) | this field's **transplanting date** | aggregate MAE 3.49 d; **p90 8.11 d, max 16.16 d** |

**These go in the Introduction as motivation and in the Discussion as
implications. They do NOT go in Results.** The measured quantity is the error
distribution; the consequence for a mandi schedule or an irrigation release is
an argued implication, not a finding. A reviewer who sees "this improves
procurement" in Results will ask for procurement data, and there is none.

The defensible form in Discussion is conditional:

> *"Harvest-date error has a p90 of 16 days. If procurement scheduling were
> driven by such a model, roughly one field-season in ten would be misplaced
> by more than two weeks -- a window within which a farmer either sells at MSP
> or sells to a trader. The aggregate MAE of 6.8 days does not expose this."*

### Explicitly excluded

**Interstate water allocation.** SICKLE covers only the Tamil Nadu Cauvery
delta (Coastal Cauvery, Coastal Vennar, Lower Cauvery, Middle Vennar, Upper
Cauvery), 2018-2020. There is **no Karnataka data**, no dam storage, no
release records and no water-balance model. Paddy area is one input to one
step of a long chain, and the Cauvery dispute concerns compliance with a
tribunal schedule rather than demand estimation. Any claim in this direction
would not survive review and is left out.

---

## Structure

1. **Introduction** — operational deployment is outpacing benchmark scrutiny
2. **Related work** — SICKLE (WACV 2024); fine-grained metrics (Wang et al.
   2023); reproducibility in remote sensing
3. **Data and protocol** — 351 plots, unlabelled test split, the gate, the
   pre-declared tolerance
4. **Reproduction** — 64 released models across 4 tasks
5. **Defects** — C1
6. **Controlled ablation** — C2
7. **Per-field analysis** — C3, C4
8. **Discussion** — what a benchmark must report before operational use
9. **Limitations** — one delta, 36 validation plots (41 field-seasons from 14
   plots for yield), validation-only, 2018–2020

---

## Target venues, realistic

| venue | fit | note |
|---|---|---|
| **IEEE JSTARS** | good | applied EO, accepts thorough empirical work |
| **Computers and Electronics in Agriculture** | good | applied agriculture, Elsevier |
| **MDPI Remote Sensing** | easiest | open access, fast, APC |
| ISPRS Journal | higher bar | would want more than one dataset |
| CV4A workshop (ICLR/NeurIPS) | good fit for C1+C3 | short format |

An audit of a single benchmark is unlikely to reach a top-tier CS venue. In an
applied remote-sensing journal it is a solid contribution.

---

## What is done and what is left

| | |
|---|---|
| Controlled ladder, 30 runs × 3 seeds | **done** |
| Reproduction gate, crop type 12/12 | **done** |
| Plot-level audit, crop type 12/12 | **done** |
| Regression audit, 52 models | **running** |
| Four defects documented | **done** |
| Policy context verified | **done** |
| **Figures** | **to do** |
| **Related-work reading** | **to do** |
| **Writing** | **to do** |
| Data-regime sweep (17/90) | paused; optional, report as partial |

The evidence is substantially complete. What remains is figures, reading and
writing — not new science.
