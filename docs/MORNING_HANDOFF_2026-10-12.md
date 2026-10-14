# Morning handoff — 2026-10-12

## The two papers

| file | what it is |
|---|---|
| `docs/Thesis_Paper.pdf` (+ `.docx`) | **The paper you asked for.** Answers all three clauses of the objective — reliability audit, safeguards, decision-support prototype — with a conclusion built clause by clause on it. About 8 pages. |
| `docs/SICKLE_Field_Level_Audit_IEEE.pdf` (+ `.docx`) | The publication-scoped audit, clauses one and two only. New conclusion, structured on the objective. 10 pages. |

Both are generated from the artefacts of both projects at build time. No
number in either is typed in, so both follow the final seeds automatically.

The thesis paper **refuses to save** if its own text drops the canonical
objective, claims something the evidence rules out (detecting crop damage,
validated with farmers, ready for PMFBY, …), or leaves the simulated section
unlabelled.

## The conclusion, in one paragraph

> Under the evaluated conditions, the prototype is weakest where historical
> records are unavailable, a setting in which its satellite-based estimates
> have not demonstrated sufficient added value to reliably prioritize
> verification. The contribution is therefore not that satellite imagery
> enables field-level decisions. It is a reproducible account of when such
> predictions should not be trusted, a disciplined rejection of safeguards
> that do not work, and a decision-support design that stays honest while
> its central model is weak.

Each clause of the objective is given a verdict in the paper's Section 9:
**supported** (conditions of unreliability), **supported as an
investigation** (safeguards), **partially supported** (decision support).

## One decision I made that you should check

**The ablation table now compares every rung on the same three seeds, and
the extra v3/v4 seeds go only into the paired follow-up they were trained
for.**

Why: with the extra seeds folded into the v3 and v4 means, the table
compared an eight-seed v4 against a three-seed v5. The radar effect then
read **+0.1566** in the IEEE paper and **+0.1860** in the paired analysis of
the *same* seeds, and the noise yardstick σmax moved for a reason unrelated
to any step. The balanced table is **byte-for-byte identical to the
published three-seed table**, so nothing already reported changes; the
follow-up is reported as what it is, a targeted experiment on one step.
Every finished run is still recorded in the table's `n_seeds_finished`
column. If you would rather fold the extra seeds into the means, that is a
one-function change in `scripts/aggregate_ladder.py`, and both papers will
follow it.

## Bugs found overnight, all fixed and guarded

Preparing the rebuild for the new seeds exposed these. Each is fixed at the
source and has a check that would catch it again.

1. **The aggregator averaged an unfinished run in.** `aggregate_ladder.py`
   labelled a v3 run at 19 of 100 epochs as INCOMPLETE but still put it in
   the v3 mean, and the paper said "seven seeds at v3" when six had
   finished. I had written in four places that this script excluded such
   runs; **that was false**, and all four are corrected. The published
   table was never affected.
2. **Eight places in the IEEE paper hard-coded the ablation partition** —
   "the two that hurt change the loss or the architecture", "these two
   effects", "resolving three steps, bounding four, leaving two open",
   "estimated from three numbers". Interim data already made several of
   them false. All are now derived.
3. **`drift_check.py` watched only v5 seeds**, and counted runs without
   honouring the completion flag. It now compares a seed signature across
   every rung.
4. **The yield-signal count was wrong across four documents.** It is
   **nine**, not eight; an earlier README table silently dropped `low
   predicted yield`. The stale count is now a banned phrase.
5. **The old IEEE conclusion overreached**: "the difficult fields are
   identifiable in advance", "adopt selective automation", "would expose
   failures before they reach the people". The evidence supports none of
   the three. The YES-TECH right-to-information and no-contest provisions,
   which lived only in that conclusion, were moved verbatim into the
   Discussion rather than lost.

## Still needs you

- **Author name, email and target venue.** Both papers carry placeholders;
  I will not invent them. The venue also decides whether 10 pages is a
  problem for the IEEE version.
- **Native-speaker review of the Tamil report.** 38 of 46 lines, including
  every safety warning, are untranslated. The report now says so at the
  top. I did not machine-translate safety text.
- **The `reviewed: false` amendment**, if one appears in
  `configs/tolerances.yaml` after the seeds land. Under the balanced design
  it should not be needed, because no declared value moves.

## What happened after you went to sleep

**The seeds finished at 09:53 and the automated rebuild was fully green.**
v3 and v4 now have eight finished seeds each. Drift was 31 of 31 within
tolerance before anything was adopted — under the balanced table no
declared number moved, so no amendment was needed — and every check
passed, `independent_check` included. Your data-regime sweep restarted at
09:54 and is training.

**The follow-up result for class weighting (v3 to v4), with eight paired
seeds: -0.0298, 95 % CI [-0.0651, +0.0054], still underpowered.** The
effect shrank as seeds were added: -0.0484 at three, -0.0357 at six,
-0.0298 at eight. The three-seed reading was partly seed noise. Settling
it would now need about sixteen seeds. Both papers report this as it is.

## Prototype validation, added this morning

All pre-registered in `sickle-guard/configs/verification_protocol.yaml`
before any result.

- **Prioritisation from field records: demonstrated.** Every
  record-based policy far outperforms random inspection on held-out plots.
- **The workflow against the register a team already keeps: advantage
  not demonstrated.** It finds at most one field-season more; every
  Bonferroni-adjusted interval includes zero. The value belongs to the
  records, not to the satellite workflow.
- **Fields with no record:** reserving 10 % of a 33 % budget for them
  keeps recorded recall at 97 % and lifts cold-start recall from 0 to 19 %.
  The satellite does **not** beat a random choice among those fields
  (+0.062, CI [-0.235, +0.364]).
- **Safety under failure:** ten predefined tests, including the real
  fields the workflow missed and an invariance test showing warnings do
  not soften as the estimate rises. A deliberately reassuring mutant was
  caught.
- **Independent outcomes: none available.** CSISA crop-cut data covers
  rice in Nepal and maize in Odisha with no confirmed repeat plots; PMFBY
  damage photographs carry no stated licence. Kept as a limitation.

## On "demonstrated farmer benefit"

It cannot honestly move far without people or independently measured
outcomes, and your own constraints rule out primary data collection. What
this morning's work reaches is the highest rung available without them:
**demonstrated prioritisation value on the evaluated dataset** — which
turns out to belong to the field records rather than to the satellite
model. That is stated as such in both papers.

## Added later on 2026-10-12: four extensions and a new objective wording

**Objective, as you asked:** "We identify conditions under which
satellite-based agricultural predictions can be unreliable, empirically
evaluate safeguards for detecting those failures, and develop a
decision-support prototype to help prioritize verification while guarding
against false reassurance." I kept "conditions under which" and
"empirically evaluate" (the reasons are in `sickle-guard/THESIS_SCOPE.md`);
the change is logged as an amendment in `configs/claims.yaml`, and
"protecting farmers from false reassurance" stays a banned phrase.

All three new studies ran under `sickle-guard/configs/extension_protocol.yaml`,
frozen before any of them was computed. For L the protocol discloses that
season means had been seen while searching for the official data.

| study | kind | headline |
|---|---|---|
| L. Labels against Tamil Nadu's records | observed | recalled Kuruvai yield is 0.77 of the state's crop-cutting yield (Samba 0.90, inside the range); in 2020-21, with about Rs 782 crore of delta crop-damage relief, recalled yields fell 3.9 %, no more than the control, and **the satellite model's estimates did not move** |
| C. Imperfect inspectors | simulated inspector, observed rankings | a register-guided inspector at sensitivity 0.30 matches a perfect random one; held-out split too small to settle the pre-registered test; over all field-seasons the advantage survives to 0.3 |
| D. YES-TECH what-if | simulated | at error 0.20, a unit 10 % short is paid nothing 29 % of the time at 100 % weight vs 11 % at 30 %; rechecking units just above the threshold is about four times as efficient as random rechecks, and is not free |
| Step 4 as software | software; visits simulated | `src/inspection_workflow.py`: blind task sheets, evidence rule, refusal of decision language, SHA-256 audit chain attacked by its tests |

**New deliverables**

- `docs/Policy_Brief.pdf`: two pages, five rules for a verification
  office, each labelled observed or simulated.
- `docs/Thesis_Paper.pdf`: now 12 pages, with new Sections 3 (official
  records), 5 (step four), 6 (two more simulated studies) and 9 (what an
  office can take from this).
- Verification console, published as a private page:
  https://claude.ai/artifact/Vf5Raw6KUuEPRp9YxShKKv (local copy
  `sickle-guard/results/console/verification_console.html`). Share it from
  the page's Share menu before the panel can open it.
- IEEE paper: the Madhya Pradesh 100 % sentence now cites the primary
  source (Lok Sabha Starred Q.25, 21 July 2026) instead of saying it was
  unverified. Still 10 pages.

**Sources found and verified this time.** Read and confirmed: the Lok
Sabha reply (primary); the CSE field assessment (secondary, qualitative);
and the Tamil Nadu Agriculture Department tables (official), parsed by
script into `sickle-guard/data/official/` with `SOURCES.yaml`. **Still not
found:** the Hindustan Times "Hingani" figures, and official 2020-21
district rice yields.

**Checks:** 152 sickle-guard tests, `check_claims` 47/47, `verify_all`
39/39, `build_all` green and reproducible.

## Response to the external review (same day)

| review point | what changed |
|---|---|
| 1. damage season overread | abstract, Section 3, Figure 3 caption, practice rule 4, conclusion, brief finding 4, console card and README now say the model's estimates were unchanged, that this is about the model's responsiveness, and that which plots were damaged is unknown. Practice rule 4 is now a validation rule ("check that a yield model responds to documented damage seasons"), not a generalisation |
| 2. small held-out set | abstract and Section 5 state 27 plots and 11 below-threshold field-seasons beside the 7/7 and 1/4 counts, and say the uncertainty is wide; the brief's table caption says so too |
| 3. baseline comparison central | abstract now says the advantage over the office's register, not only over the last-harvest lookup, is not demonstrated and that the value lies in the records; brief finding 1 says the workflow's one extra field is not distinguishable from chance |
| 4. brief's strongest wording | "nobody in the chain is prompted to look" removed. The opening now cites the grievance committees and helpline from the Lok Sabha reply, attributes the 20-30 % figure to insurers in a three-district Madhya Pradesh assessment, and says that is one state's experience, not a measured property of YES-TECH |
| 5. placeholders, Tamil, venue | **still needs you** |
| next step: pilot protocol | `sickle-guard/PILOT_PROTOCOL.md`, proposed and not run: union design, blind crop-cutting on every inspected field plus a 10 % random audit sample, sized by `src/pilot_power.py`. Records vs random needs about 118 fields; the satellite's edge over the register, at its observed size, about 12,505, so the pilot is not built around it |

Checks after these changes: 153 tests, `check_claims` 52/52, `build_all`
green, IEEE 10 pages, thesis 12, brief 2.

## RESOLVED: per-field score corrected (was: open issue found while rendering the prediction images)

The per-field score used in the IEEE paper and thesis is macro IoU over
{paddy, non-paddy}. 35 of the 36 validation fields hold one crop in every
season, and for such a field that score is 1.0 when every pixel is right and
**at most 0.5 once a single pixel is wrong** (the absent class's IoU becomes
0). Re-scored with the IoU of each field's own crop (`scripts/field_metric_check.py`,
v5, three seeds, CPU inference reproducing the published per-field values
exactly):

| per-field measure | median field | fields below 0.50 |
|---|---|---|
| macro IoU (as published) | 0.672 | 12 of 36 |
| IoU of the field's own crop | 0.957 | 4 of 36 |

So "a pooled IoU of 0.8665 beside a median field of 0.672" and "12 of 36
fields below 0.50" are mostly the metric's step. What survives: one field
holds 47 % of scored pixels and moves the pooled figure by up to 0.261;
errors concentrate in a few fields the pooled number does not identify; the
batch-size and constant-baseline findings are unaffected.

**Corrected in both papers on 2026-10-12, at your go-ahead.** What changed:

- `src/evaluation/metrics.py`: field IoU (`plot_own_class_iou`), with tests
  pinning the defect (a 99 %-correct field: macro 0.495, field 0.99).
- `scripts/field_confusions.py`: per-field confusion matrices for every
  rung and seed, from CPU inference, checked against training's pooled
  scores. Every per-field number and figure now comes from these.
- Field distribution: median 0.957, mean 0.855, 4 of 36 below 0.50, the 3
  worst carry 66 % of the shortfall; size vs accuracy Spearman +0.16 (CI
  spans zero). The macro values stay on the record (`macro_contrast`).
- Crop-type screen: "unreliable" now means field IoU below 0.50: 4 fields
  (20, 124, 143, 219), all inside the old 12. Seed disagreement no longer
  beats chance; confidence carries some information beyond size (p = 0.030)
  but is not separable from checking the smallest fields first (+0.031,
  CI -0.257 to +0.457). Verdict unchanged in form: partial. The original
  analysis reproduces exactly (`*_macro.json`). Two post-hoc amendments are
  in `configs/conditional_screen_protocol.yaml` (target; size compared in
  both directions).
- IEEE: abstract, contributions, Sections V-C, V-D, V-F, V-G, discussion,
  conclusion rewritten; section D's "direction is task-dependent" was false
  and is replaced. Thesis: abstract, Section 3, Section 4, Section 7,
  conclusion.
- `configs/tolerances.yaml`: six declared values adopted with a dated
  amendment marked as a measurement defect. `verify_all.py`: five checks
  that encoded the old framing rewritten to guard the corrected claims.
- Live docs corrected (`EVIDENCE_TO_DESIGN.md`, `YESTECH_POLICY_CONTEXT.md`);
  historical records carry a dated correction note.

