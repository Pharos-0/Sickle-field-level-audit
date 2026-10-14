# Manuscript audit — SICKLE field-level audit, IEEE conference version

> **Correction, 2026-10-12.** The per-field figures in this document used IoU averaged over both classes, which caps any imperfect single-crop field at 0.5 (35 of the 36 validation fields hold one crop). Scored on the crop each field holds, the median field is 0.957 (not 0.672) and 4 of 36 fields fall below 0.50 (not 12). The current figures are in `results/field_distributions.json` and the papers; this record is left as written.


Built 2026-10-11 from `scripts/make_paper_ieee.py`.
Manuscript: `docs/SICKLE_Field_Level_Audit_IEEE.docx` / `.pdf` — 5 pages, 4,361 words.

This file records what was changed in the manuscript and why, what was checked,
and what was **not**. Where a check could not be performed it says so rather
than reporting a pass.

---

## 1. Structural changes made in this phase

| # | Change | Reason |
|---|---|---|
| 1 | Restructured to the eight-section IEEE outline (Introduction / Related Work / Dataset and Experimental Setup / Methodology / Results and Analysis / Discussion / Limitations and Threats to Validity / Conclusion) | The previous draft interleaved method and result; a reviewer could not tell which numbers were protocol and which were findings |
| 2 | Two-column layout, full-width title block, 9.5 pt Times, column-width figures redrawn at 3.35 in / 400 dpi | Figures drawn for a 6.5 in page lose their axis labels when scaled into a column |
| 3 | Added Related Work as its own section with four subsections | The draft cited Wang et al. only in passing; the metric-bias result is prior work and had to be positioned as such before the contributions are stated |
| 4 | Added an explicit Scope paragraph in the Introduction | Separates this audit from the in-progress data-regime study, which must not be read as part of it |
| 5 | Moved all sample sizes into Section III | They were scattered across results, so the reader met "36 fields" after the claims that rest on it |

## 2. Claim-level corrections

These are changes to what the manuscript asserts, not to its wording.

| # | Was | Now | Evidence |
|---|---|---|---|
| 1 | "with the twelve crop-type cells this gives **64 verified models**" | "52 of 52 regression gates passed; **9 of 12** crop-type cells reproduce within the declared 0.01 IoU tolerance and **3 miss it, by 0.0114, 0.0123, 0.0345**. Across the two groups 64 models were re-evaluated and 61 reproduced." | `results/gate/*_bs32.json`; surfaced by `scripts/verify_all.py` reporting 9/12 |
| 2 | Abstract: "we reproduce 64 released models" | "we re-evaluate 64 released models … 52 of 52 regression checkpoints reproduce … and 9 of 12 crop-type cells do" | as above |
| 3 | Conclusion: "Reproducing 64 released models" | "Re-evaluating 64 released models across four tasks, of which 61 reproduce within our declared tolerances" | as above |
| 4 | Contributions (iii): "audit of 64 released models" | now reads the count from disk at build time | cannot drift from `results/` |

The tolerance was **not** widened to absorb the three misses. The manuscript
states the residual and notes it is consistent with the batch-composition
sensitivity reported as defect D1.

## 3. Reference corrections

All 18 references were re-checked against Crossref or the publisher's page in
this phase. Four entries were wrong or incomplete and were corrected:

| Ref | Defect found | Correction | Source |
|---|---|---|---|
| SICKLE (Sani 2024) | page range given as **5995–6004** | **5983–5992** | Crossref `10.1109/WACV57701.2024.00589`, `page` field |
| Ploton 2020 | first initial given as **N. Ploton** | **P. Ploton** (Pierre Ploton) | Crossref `10.1038/s41467-020-18321-y` |
| Koukos 2024 | first initial **H.**; author list abbreviated | **A. Koukos** (Alkiviadis) + all six authors + pp. 1540–1545 | Crossref `10.1109/IGARSS53475.2024.10641793` |
| Reuss 2025 | author list abbreviated to "et al." with 5 authors | all five listed; vol. 12, no. 1, art. 664 | Crossref `10.1038/s41597-025-04952-7` |

IEEE's >6-author rule was then applied uniformly: entries with more than six
authors (SICKLE 8, Rao 8, Wang 9, Ploton 13) print first author + "et al.";
entries with six or fewer list all. The full author lists remain in
`docs/references.json` and in `REFERENCE_VERIFICATION.xlsx` as the verification
record.

URLs are printed only for the three sources whose URL is their only locator
(the YES-TECH manual, the PMFBY guidelines, the Down To Earth articles).
Retrieval URLs for proceedings papers are kept in the JSON as verification
evidence but are not part of the printed citation.

## 4. A numbering defect found and fixed in the build itself

The prose cites reference numbers as literals. The first version of the build
renumbered them to IEEE first-appearance order and then **wrote the new
numbering back over `docs/references.json`**. On the next build the literals
resolved against the rewritten file, silently remapping every citation. This
was caught when the reference list came back in a different order than the one
that had just been inspected.

Fix: `docs/references.json` holds the authoring numbering and is never
rewritten by the build; the manuscript's numbering goes to the generated
`docs/references_final.json`, which `make_reference_table.py` reads so the
spreadsheet and the PDF cannot disagree. The corrupted file was restored by
matching entries on first author — content, not position. Build idempotence
is now checked: two consecutive builds produce byte-identical
`references_final.json`.

## 5. Figure corrections

| Figure | Defect | Fix |
|---|---|---|
| Fig. 3 (per-field IoU) | overlaid the **mean** per-field IoU (0.675) while the text quotes the **median** (0.672), and rounded the headline to 0.867 against the text's 0.8665 | overlays the median; headline shown to 4 dp |
| Fig. 4 (error tails) | three independent y-scales in one column; tick labels ran into the neighbouring panel's bars | constrained layout, smaller tick type, y-margin |

## 6. Wording checked and left unchanged

A sweep of the rendered PDF for `causal`, `first study`, `no previous work`,
`novel architecture`, `state-of-the-art`, `proves`, `guarantee`,
`significantly outperform` returned exactly one hit: the sentence that
*disclaims* causality ("we term this controlled component attribution and make
no causal claim"). No forbidden claim appears.

The manuscript continues to state, in Related Work and again in Limitations,
that per-field evaluation is established methodology from Wang et al. and is
applied here rather than proposed.

## 7. What this audit did NOT establish

- **Independent reproduction.** Every check ran on this machine against this
  repository's artefacts. No one else has run the released code and data.
- **The three non-reproducing crop-type cells were not diagnosed.** Their
  deltas are reported; the cause is not established beyond noting consistency
  with D1.
- **Author identity.** The author block is a placeholder. The manuscript cannot
  be submitted until it is filled in.
- **Two primary sources have not been read in full by the author**: the SICKLE
  paper and Wang et al. (NeurIPS 2023). Both are load-bearing — Wang et al.
  determines what in this paper is and is not novel.
- **Page range for one reference** is the only Crossref field for SICKLE that
  was previously wrong; the remaining entries' page ranges were taken from the
  publisher record and spot-checked, not re-derived from the PDFs.
- **The data-regime sweep (second study) is not part of this manuscript** and
  none of its results appear here.

---

## 8. Second review round (2026-10-11) — findings and corrections

An external review raised four points. Three were valid and exposed defects
that had survived the first round; one was already satisfied. What follows is
what was found when each was checked against the artefacts.

### 8.1 Task count — a real inconsistency, now resolved

Checking the counts revealed two separate problems, not one.

**The released tree ships 2-epoch `_debug` smoke tests beside the baselines.**
Crop type already excluded its one (12 of 13 cells gated); the regression
tasks silently included four. So "64 models" counted four smoke tests as
released models, and the two halves of the total were computed by different
rules. All counts now exclude `_debug`:

| Task | Checkpoints audited | Reproduced |
|---|---|---|
| crop type (segmentation) | 12 | 9 |
| yield | 24 | 24 |
| harvesting date | 12 | 12 |
| transplanting date | 12 | 12 |
| **total** | **60** | **57** |

**SICKLE annotates five targets, not four.** Sowing date was never part of the
per-field audit and the manuscript did not say so. Section IV-A now states the
inventory explicitly: five annotated targets, four audited per field, sowing
date included only in the metric check of Section V-A. It also explains why
yield carries twice as many checkpoints as the date tasks (each cell is
released in a per-sample and a `_season` variant).

`verify_all.py` gained a check asserting the exclusion, so the count cannot
silently drift back.

### 8.2 A capped check inflated nothing but understated a result

While tracing the counts, `verify_all.py`'s date-metric check was found to
read `sorted(d_.iterdir())[:4]` — the first four runs per task. That cap is
where the manuscript's "twelve released date runs" came from; it was not that
there were twelve. The cap was arbitrary and has been removed. The claim is
now checked against **all 39** released date runs across all three date tasks,
sowing included, and holds in 39/39. The manuscript says 39.

### 8.3 Reference [4] cited a figure neither article contains

The entry bundled two Down To Earth articles under one URL. Both were
retrieved and read in full:

- **"344 districts across 12 states" appears in neither article.** Tracing it
  further, it comes from a secondary report of a government statement, and the
  state count varies across sources (nine, ten, and twelve in different
  reports) with no official release located. **The claim has been removed from
  the manuscript.** It was not load-bearing.
- The Madhya Pradesh 100 % weightage claim **is** supported, by the first
  article, verbatim: "Madhya Pradesh took the lead by adopting YES-TECH for
  select crops" / "allocated 100 per cent weightage to it as compared to CCEs".
- **Both articles are 2026, not 2025** as the entry stated.

The entry is now split into two references, each with its own URL, author,
correct date, and a recorded statement of which claim it supports — and, for
the second, an explicit note that it contains no district count and no
weightage figure, so neither is attributed to it. The second article is cited
for farmer-level contestation, which it does support. Reference count: 19.

### 8.4 The policy claim was wider than the manual

The conclusion read that the estimates "are neither appealable in court nor
obtainable under the Right to Information Act". Reading the OCR'd manual, what
it actually provides is narrower and concerns *past* outputs:

- Clause 5.1.4: model outputs of past years "shall not be used for (a) making
  comparisons with other datasets, (b) revising the loss assessments of past
  years, and (c) challenging the estimates of current/past years".
- Past estimates "fall within the category of 'intellectual property' under
  section 8(1)(d) of the Right to Information Act, 2005, and not be made
  public".

A third sentence in that passage ends "...court of law or appellate authority",
but OCR page interleaving has lost its subject, so **it is not quoted or relied
on**. The conclusion now quotes the two provisions above and states explicitly
that the paper draws no legal conclusion from them, since whether and how an
estimate can be contested is a question of law outside its scope.

The other two manual quotations were re-verified verbatim in this round and
are exact: clause 3.3.10 on the 30 % floor and states' discretion to go
higher, and "at the smallest administrative unit, i.e., Gram Panchayat (GP)".

### 8.5 Consistency of caution about the ablation — applied

The reviewer asked that the caution shown in the results section be carried
into the abstract and conclusion. Both now state that three seeds on a
36-field validation set support a direction rather than an effect size, and
that the inconclusive rungs are not evidence of no effect.

### 8.6 Related work — gap now stated, not implied

Section II gained subsection E, "The gap this paper addresses", which names
what each cited strand establishes and what it leaves untested, and maps each
untested point to the specific section that tests it.

### 8.7 One consequence caught downstream

Excluding the smoke tests shifts Table I (e.g. harvest reported MAE
13.97 → 13.11, p90 28.43 → 27.86). Figure 4 is built from the same file and
still filtered on the old rule, which would have put the figure and the table
in disagreement. `make_figures_ieee.py` now applies the identical exclusion.

---

## 9. Third review round (2026-10-11)

The review was written against the pre-second-round reports (it cites 11/11,
18 references, 52/52, 5 pages). Its substantive points were acted on anyway.
Two of them, carried out properly, changed the paper's claims.

### 9.1 Reading the SICKLE paper demoted one of the four findings

The review asked that the two load-bearing papers be read. Reading the SICKLE
preprint (arXiv 2312.00069) against the manuscript found that **"defect 4" is
not a defect.** The paper states the definition explicitly: it normalises "the
MAE by the maximum possible date, i.e. 183, instead of the ground truth
value", and gives the reason — a conventional MAPE overshoots when the
sowing-date range approaches zero. The paper's own main results table (Table
3) reports the date tasks in MAE days, not in that column.

The manuscript now says so. The finding is restated as: the definition is
documented and reasonable, but it is not carried in the artefacts — the
released logs label the column MAPE with no qualification — and our
contribution is the empirical confirmation that the value equals MAE/183 in
39/39 runs, so the column is not comparable to a percentage error reported
elsewhere. Headline changed from "four defects" to "three defects and one
documented metric that the artefacts make easy to misread", in the abstract,
the contributions list, the Section V-A heading and the conclusion.

Two smaller corrections came from the same read:

- **UTAE baselines are supplementary models.** Table 3 reports U-Net3D for
  time series and U-Net2D for single images. The three degenerate UTAE runs
  are still shipped as baselines and the finding stands, but the manuscript
  now states they are not the paper's headline models.
- **Plot count.** The dataset paper reports 388 unique plots; the released
  index exposes 351. Counting the released CSV directly: 2,370 rows, 351
  unique PLOT_IDs, train 1,937 rows / 315 plots, val 227 rows / 36 plots,
  test 206 rows with **every PLOT_ID blank**. The difference is the test
  split, and this independently corroborates the unlabelled-test-split
  finding. Both numbers are now given with the explanation. A sentence in
  Section V-B that argued from "351 plots" now correctly says 315 training
  plots.

### 9.2 The YES-TECH source ambiguity is resolved

Both documents were obtained and compared.

- The **48-page** document is "YES-TECH under PMFBY — Manual for
  Implementation", MNCFC/DAFW, **January 2023**, no version or issue number.
  Clause 4.3.6: "The error (nRMSE) between the observed and modeled yield
  should not be more than 30% in historical data. All IUs where the error is
  more than 30% should not use this methodology."
- The **116-page** document cited as [2] is Document Control Number
  DAFW/MNCFC/PMFBY/YES-TECH/Manual_V1.0, Issue no. 01, published **11 July
  2023**, and is the only YES-TECH manual listed on MNCFC's own manual page.

The later, versioned, officially published document governs. Searching its
OCR — including on a de-spaced copy, since the OCR drops spaces — NRMSE
appears 7 times, only in lists of accuracy metrics an agency should report,
and the phrases "not be more than 30" and "error is more than 30" do not
appear at all. **The governing manual carries no binding accuracy floor.**

This is now stated in the manuscript with both documents cited, since it
strengthens the motivation rather than weakening it. The superseded manual is
reference [20]. Reference count: 20.

### 9.3 The three non-reproducing cells

Reworded. The manuscript now says explicitly that the cause is not
established, that batch-composition sensitivity is a candidate rather than a
diagnosis, and that it was not isolated.

### 9.4 Modest real-world framing — verified, unchanged

Already present in the abstract, Discussion and Limitations, and in the
Conclusion's closing sentence ("This does not pay a farmer or raise a yield").
Re-read and left as is.

---

## 10. Fourth audit round (2026-10-11)

Three errors were found by recomputing claims from raw evidence rather than
re-reading the text. Two of them I had introduced myself in earlier rounds.

### 10.1 I had corrupted the SICKLE page range — reverted

The audit prompt flagged this and was right. In round one I "corrected" the
page range from 5995–6004 to **5983–5992** on the strength of the Crossref
`page` field. Reading the **official CVF Open Access record** directly, its
own BibTeX gives `pages = {5995-6004}`.

The original value was correct and my correction was the error. It is
reverted. The discrepancy is real — Crossref's record for the IEEE DOI
`10.1109/WACV57701.2024.00589` does return `5983-5992`, and IEEE Xplore and
CVF pagination for WACV differ — so `references.json` now records both, states
which is cited and why (the CVF record is the proceedings page the authors'
own BibTeX points to), and notes that an earlier draft printed the wrong one.
The title casing was also aligned to the official record ("Annotated With").

**Lesson applied:** indexer metadata does not override a publisher record.

### 10.2 The ablation was mis-partitioned and one rung was missing

Recomputing the nine adjacent steps from `ladder_crop_type.csv`:

| step | change | Δ pixel-pooled IoU | vs σmax = 0.0417 |
|---|---|---|---|
| v1→v2 | full temporal sequence | **+0.1198** | exceeds, up |
| v2→v3 | augmentation | −0.0110 | inconclusive |
| v3→v4 | class-weighted loss | **−0.0484** | exceeds, down |
| v4→v5 | Sentinel-1 radar | **+0.1860** | exceeds, up |
| v5→v6 | Landsat-8 third sensor | +0.0075 | inconclusive |
| v6→v7 | UTAE for U-Net3D | **−0.0696** | exceeds, down |
| v7→v8 | wider encoder | +0.0004 | inconclusive |
| v8→v9 | cosine schedule | −0.0078 | inconclusive |
| v9→v10 | plot-balanced sampling | −0.0110 | inconclusive |

Two errors follow. The manuscript said **"two of nine steps produce effects
larger than the seed-to-seed spread"** — **four** do, two upward and two
downward. And the **class-weighted loss rung (v3→v4, −0.0484) was absent from
the manuscript entirely**; the prose enumerated eight of the nine steps.

Both fixed. Section V-B now derives the full partition from the file at build
time and prints every step with its delta, so a rung cannot be dropped. The
threshold is also now defined explicitly (a step is inconclusive when |Δ| <
σmax, the largest seed-to-seed SD on any rung — a conservative choice; the
per-pair criterion gives the identical partition), and three caveats were
added: inconclusive ≠ absent, σmax is a descriptive spread and not a
confidence interval so exceeding it is not a significance test, and the
ordering is cumulative. Abstract, contributions and conclusion updated to
"5 of 9 inconclusive" with the four distinguishable steps described correctly.

A new check, `Ladder: all adjacent steps partitioned`, asserts the nine steps
sum to the reported partition. `verify_all.py` is now 13/13.

### 10.3 The cross-seed correlation range was understated

The manuscript said per-field IoU correlates between seeds at **r = +0.52 to
+0.58**. Recomputing all three seed pairs: **+0.5239, +0.5673, +0.6337**. The
upper end is +0.63, not +0.58. The range is now computed from the prediction
files at build time, reported over all pairs, and accompanied by a statement
that three seeds give three estimates on 36 fields, so a range is reported
rather than a pooled coefficient and no p-value is attached.

### 10.4 Claims that verified exactly, recomputed from raw files

| Claim | Manuscript | Recomputed | Source |
|---|---|---|---|
| season-level samples | 2,370 | 2,370 | released CSV |
| unique plot IDs in index | 351 | 351 | released CSV |
| train | 1,937 rows / 315 plots | 1,937 / 315 | released CSV |
| validation | 227 rows / 36 plots | 227 / 36 | released CSV |
| test rows with blank PLOT_ID | 206 | 206 | released CSV |
| yield/phenology field-seasons | 41 | 41 | audit_summary.csv |
| pixel-pooled IoU | 0.8665 | 0.866497 | ladder_crop_type.csv |
| median field IoU | 0.672 | 0.6721 | per-plot IoU files |
| fields below 0.5 | 12 of 36 (33 %) | 12 of 36 | per-plot IoU files |
| below 0.5 in every run | 11 of 12 | 11 of 12 | per-plot IoU files |
| screening recall @ 50 % workload | 92 %, CI [68 %, 100 %] | 0.9167, [0.6842, 1.0] | triage_v5.json |
| screening precision | 61 % | 0.6111 | triage_v5.json |
| confidence-only @ 33 % | 67 %, CI [40 %, 91 %] | 0.6667, [0.4000, 0.9091] | triage_v5.json |
| batch 32 / 16 / 8 | 0.7071 / 0.6561 / 0.6409 | same | gate JSONs |
| reproduction | 48/48 and 9/12 | same | audit_summary.csv, gate JSONs |

### 10.5 Policy wording narrowed again

Both the abstract and the conclusion said validation is "specified only at the
Gram Panchayat", which asserts a restriction the manual does not state. Both
now say the manual identifies the Gram Panchayat as the smallest
administrative unit for its stated objective and that our documentary review
found **no explicit requirement** below it — stated as an absence in the
document, explicitly not a prohibition and not a claim about practice.

### 10.6 Consistency matrix

| Claim or artefact | Manuscript | Evidence | Audit log | Validation report | Status |
|---|---|---|---|---|---|
| page count | 6 | 6 (rendered) | — | 6 | consistent |
| references | 20 | 20 in JSON | 20 | 20 | consistent |
| citation numbers | 1–20, contiguous | 20 entries | — | — | consistent |
| SICKLE pages | 5995–6004 | CVF BibTeX | documented | — | consistent |
| ablation partition | 5 of 9 inconclusive | 5 of 9 | tabulated | 4+5 recorded | consistent |
| seed correlation | +0.52 to +0.63 | +0.5239…+0.6337 | recorded | — | consistent |
| claim checks | — | 13/13 | — | 13/13 | consistent |
| tests (project venv) | — | 55 passed | — | 55 passed | consistent |
| tests (clean venv) | — | 46 passed, 1 skipped | — | 46 passed, 1 skipped | consistent |
| audited models | 60, 57 reproduced | 60 / 57 | 60 / 57 | 60 / 57 | consistent |

Stale content checked for and **not** found in the current reports: "18
references", "52/52", "11 / 11". The "five pages" sentence was stale and has
been rewritten.

---

## 11. Overnight round (2026-10-11): yield-reliability extension

The manuscript gained one new experiment, Section V-H, and one existing
finding was resolved. Both outcomes are negative or partly negative and both
are reported as such.

### 11.1 A gap that made the screening claim uncheckable

The crop-type screening of Section V-F could not be verified by anyone
without re-running inference: the per-field signals were never written to
disk. The same was true of yield — `per_field_abs_error` was saved, but an
absolute error has already discarded the prediction, so no label-free signal
could be derived from it. Both exports were added
(`results/triage/per_field_signals_v5.csv`,
`per_field_pred` / `per_field_true` in every regression gate) and all 24
non-smoke-test yield checkpoints were re-evaluated on CPU to populate them.

### 11.2 The yield screen fails, and the failure is the contribution

Protocol frozen in `configs/yield_screen_protocol.yaml` **before any result
was computed**, including the threshold, the baselines and the rejection
criteria. Signal: coefficient of variation of predictions across the 24
checkpoints, computable at inference time. Target: ensemble-mean APE above
50 %, stated in the paper as a study choice and not a YES-TECH or PMFBY
criterion. No epsilon was introduced anywhere: truths span 800–3,840 and all
ensemble means are positive, so both denominators are safe by inspection and
the protocol asserts that rather than patching it.

| | |
|---|---|
| checkpoints | 24 |
| field-seasons | 41 |
| **independent plots** | **14** |
| unreliable field-seasons | 8 |
| AUROC | **0.311**, 95 % CI [0.103, 0.634] |
| Spearman(signal, APE) | +0.046 |
| verdict | **not demonstrated** — CI contains chance |

Recall falls below the random-referral expectation at all four workloads
tested (12 % vs 29 %, 25 % vs 37 %, 38 % vs 51 %, 38 % vs 73 %). With 14
plots the intervals are far too wide to call the signal actively harmful;
what they exclude is any claim that it works.

Checks that the outcome is not an artefact:

- **Threshold sensitivity.** AUROC across APE thresholds 30/40/50/75/100 %:
  0.570, 0.464, 0.311, 0.219, 0.321 — scattered around chance.
- **Model subsets.** Base runs only 0.312, season runs only 0.458.
- **Baselines.** Smallest-area-first 0.367, lowest-predicted-yield-first
  0.591. The best of the three is a baseline, and its margin is well inside
  the uncertainty this sample supports.
- **Selective prediction.** Withholding the top half by disagreement moves
  median APE 18.1 % → 13.5 % but leaves the mean at 32.3 % → 32.9 %. A screen
  that worked would move both; the median shift alone is consistent with
  reordering noise and the paper says so.

**Ensemble validity, disclosed.** The 24 checkpoints are 4 sensor sets × 3
architectures × 2 season variants, not 24 independent models. Median pairwise
prediction correlation is **0.924**, reaching 0.997 for base runs against
their own season twins. The paper therefore calls the quantity *inter-model
disagreement*, never *predictive uncertainty* — calibration was not
evaluated — and scopes the finding to this ensemble on this sample.

### 11.3 The three unreproduced crop-type cells are now diagnosed

All three were swept across evaluation batch sizes. They do **not** share a
cause, and the paper no longer implies they do.

| Cell | Behaviour across batch size | Verdict |
|---|---|---|
| `[S2]_unet3d` | 0.7086 / 0.7192 / 0.7082 at 8 / 16 / 64 vs released 0.7173; misses only at the pinned 32 | **explained** by batch composition |
| `[S1]_unet3d` | 0.4616 → 0.7888 from batch 1 to 16, then bit-identical at 16/32/64; never reaches 0.8011 | **not explained**; saturates 0.0123 short |
| `[S2]_convlstm` | ranges over 0.178 IoU with no monotone trend; closest value is the pinned 32 itself, 0.0345 away | **not explained** |

The `[S2]_unet3d` result is itself an instance of defect D1: whether a cell
"reproduces" depends on the evaluation batch size chosen.

### 11.4 A defect in my own diagnostic, found and fixed

`diagnose_gate_misses.py` treated a non-zero exit from `phase5_gate` as a
crashed run. But that script exits 1 whenever the verdict is FAIL — the
expected outcome for exactly the cells being diagnosed. The first artefact it
produced therefore discarded valid measurements and reported "spread 0.0000,
score barely moves with batch size" for two cells whose scores move by 0.178
and 0.327. Success is now decided by the artefact on disk and its mtime,
never by the return code, and a `--cached-only` mode builds the report from
existing measurements without launching anything.

### 11.5 Manuscript changes

- New Section **V-H**, generated from `results/triage/yield_screen.json`, with
  motivation, falsifiable hypothesis, export procedure, signal and target
  definitions, leakage controls, baselines, Table II, Fig. 7 and limits.
- **Abstract**, **contributions** (new item v), **limitations** and
  **conclusion** updated to carry the negative result.
- Section **V-D** residuals paragraph replaced with the measured three-cell
  diagnosis.
- Headline count corrected from "four defects" to "three defects and one
  documented metric the artefacts make easy to misread" in round 9 stands.

### 11.6 Scope decisions taken without asking

- **ConvLSTM at batch 1 and 4 was abandoned** after one such evaluation
  exceeded 45 minutes of CPU with the GPU committed to another job. Batch 1
  and 4 cannot answer whether any batch size closes the gap — on the first
  cell both moved the score far away from the released value — so they were
  dropped from the default sweep and the omission is recorded in the script's
  help text. `[S2]_convlstm` is still covered at 1, 8, 16, 32 and 64.
- **Figure 7 was redrawn stacked** after the first version, drawn at page
  width, was clipped inside a 3.3 in IEEE column.
