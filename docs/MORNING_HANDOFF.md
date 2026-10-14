# Morning handoff — 2026-10-11

Everything below was run, not planned. Where something was not completed it
says so and why.

**Bottom line:** the paper is complete and internally consistent at 8 pages
with 20 references, all 18 claim checks, 12 independent re-derivations and 71
tests passing. The new experiment that was added **failed**, and the paper
reports it as a failure — that is the honest outcome, not a shortfall.
It is **not submission-ready**, for three reasons that need you, listed in
§8.

---

## 1. Task ledger

| Task | Status | Evidence |
|---|---|---|
| Per-field-season yield prediction export | complete | 24/24 gates carry `per_field_pred` |
| Crop-type screening signal export | complete | `results/triage/per_field_signals_v5.csv`, 36 rows |
| Frozen screening protocol | complete | `configs/yield_screen_protocol.yaml`, frozen before results |
| Yield screening experiment | complete — **negative** | `results/triage/yield_screen.json` |
| Yield figure | complete | `results/figures_ieee/fig_yield_screen.png` |
| Diagnosis of 3 unreproduced cells | complete | `results/gate/diagnosis_misses.json` |
| Manuscript integration (Section V-H) | complete | 8-page PDF, inspected page by page |
| Abstract / contributions / limitations / conclusion | complete | updated to carry the negative result |
| New tests | complete | 16 added, 71 total passing |
| Reproducibility docs | complete | `docs/REPRODUCE.md` |
| One-command build | complete | `scripts/build_all.py`, green |
| Reports updated | complete | audit §11, validation §6 |
| ConvLSTM at batch 1 and 4 | **not done** | >45 min CPU per evaluation; see §6 |
| Independent reproduction by another person | **not done** | needs a second person |
| Target-venue formatting compliance | **blocked** | no venue named |

## 2. Jobs run and exit codes

| Job | Exit | Note |
|---|---|---|
| yield prediction export, 24 checkpoints (CPU) | 0 | `ALL YIELD GATES DONE` |
| `triage_eval.py` v5, 3 seeds (CPU) | 0 | reproduced the existing triage result and added the signal CSV |
| `diagnose_gate_misses.py` (CPU) | 0 | first artefact was wrong; see §5 |
| `yield_screen.py` | 0 | |
| `build_all.py` | 0 | all steps + all checks |
| GPU sweep (separate study) | still running | untouched all night |

## 3. The new result, in full

**Question.** Can disagreement among released yield checkpoints identify
field-seasons whose yield estimate is unreliable, using only inference-time
information?

**Answer: no, on this sample.**

| | |
|---|---|
| checkpoints evaluated | 24 |
| field-seasons | 41 |
| **independent plots** | **14** |
| unreliable field-seasons (APE > 50 %) | 8 |
| AUROC | **0.311**, 95 % CI **[0.103, 0.634]** |
| Spearman(signal, APE) | +0.046 |
| verdict | **not demonstrated** (CI contains chance 0.500) |

Recall is **below** random referral at every workload tested:

| Workload | Referred | Caught | Recall | 95 % CI | Random expects |
|---|---|---|---|---|---|
| 29 % | 12/41 | 1/8 | 12 % | [0 %, 43 %] | 29 % |
| 37 % | 15/41 | 2/8 | 25 % | [0 %, 86 %] | 37 % |
| 51 % | 21/41 | 3/8 | 38 % | [0 %, 86 %] | 51 % |
| 73 % | 30/41 | 3/8 | 38 % | [0 %, 86 %] | 73 % |

With 14 plots the intervals are too wide to call the signal harmful. What
they exclude is any claim that it works.

**Sensitivity analyses, all consistent with the null.**

- APE threshold 30 / 40 / 50 / 75 / 100 % → AUROC 0.570 / 0.464 / 0.311 /
  0.219 / 0.321.
- Base runs only 0.312; season runs only 0.458.
- Baselines: smallest-area-first 0.367, lowest-predicted-yield-first 0.591.
  The best of the three is a **baseline**, and its margin is well inside the
  uncertainty this sample supports.
- Selective prediction: withholding the top half by disagreement moves median
  APE 18.1 % → 13.5 % but leaves the mean at 32.3 % → 32.9 %. A working screen
  moves both; the median shift alone is consistent with reordering noise.
- Field size: Spearman area vs APE +0.216, area vs signal −0.135.

**Ensemble validity.** The 24 checkpoints are 4 sensor sets × 3 architectures
× 2 season variants — not 24 independent models. Median pairwise prediction
correlation **0.924**, max 0.997 (base runs against their own season twins).
The paper therefore says *inter-model disagreement*, never *uncertainty*, and
scopes the finding to this ensemble on this sample.

**Does this strengthen the paper? Yes.** Without it, a reader of Section V-F
would reasonably assume the crop-type screening result extends to the yield
estimates an assessment actually consumes. It does not, and now the paper
says so with a pre-registered protocol behind it.

## 4. The three unreproduced crop-type cells — now diagnosed

They do not share a cause.

| Cell | Across batch size | Verdict |
|---|---|---|
| `[S2]_unet3d` | 0.7086 / 0.7192 / 0.7082 at 8 / 16 / 64 vs released 0.7173; misses only at the pinned 32 | **explained** by batch composition |
| `[S1]_unet3d` | 0.4616 → 0.7888 from batch 1 to 16, then bit-identical at 16/32/64 vs released 0.8011 | **not explained**; saturates 0.0123 short |
| `[S2]_convlstm` | spans 0.178 IoU with no monotone trend; closest value is the pinned 32, 0.0345 away | **not explained** |

`[S2]_unet3d` is itself an instance of defect D1: whether a cell "reproduces"
depends on the evaluation batch size chosen.

## 5. A mistake I made and corrected

`diagnose_gate_misses.py` treated a non-zero exit from `phase5_gate` as a
crashed run. That script exits 1 whenever the verdict is FAIL — the expected
outcome for precisely the cells being diagnosed. The first artefact therefore
threw away valid measurements and reported "spread 0.0000, score barely moves
with batch size" for two cells whose scores move by 0.178 and 0.327 IoU. I
caught it reading the output, fixed it to decide success from the artefact
and its mtime rather than the return code, added a `--cached-only` mode, and
regenerated. `verify_all.py` now pins the one-of-three split so it cannot
drift back.

## 6. What was not completed, and why

- **ConvLSTM at batch 1 and 4.** One such evaluation exceeded 45 minutes of
  CPU with the GPU committed to your sweep. Batch 1 and 4 cannot answer
  whether any batch size closes the gap — on the first cell both moved the
  score far *away* from the released value — so they were dropped from the
  default sweep. `[S2]_convlstm` is still covered at 1, 8, 16, 32 and 64, and
  the omission is recorded in the script's help text and in the manuscript.
- **Independent reproduction by another person.** Needs a second person.
- **Target-venue compliance.** No venue named, so page limit and template
  cannot be checked. At 8 pages this exceeds a typical 6-page conference
  limit; if you name the venue I can cut or reformat.

## 7. Test and validation status

| Layer | Result |
|---|---|
| `verify_all.py` (project venv) | **18/18 PASS** |
| `verify_all.py` (clean venv) | 18/18 PASS |
| `independent_check.py` | **12 passed, 0 failed, 0 skipped** |
| `pytest` (project venv) | **71 passed** |
| `pytest` (clean venv) | 46 passed, 1 skipped (torch blocked by Smart App Control) |
| Build idempotence | confirmed |
| PDF render | 8 pages, inspected page by page |
| Forbidden-claim sweep | clean; every "payout" hit is policy context or a disclaimer |
| Citations | 1–20 contiguous, order of first appearance |

The independent checker re-derived AUROC 0.311 and the CI [0.103, 0.634] with
its own implementation, sharing no code with the experiment that produced
them.

## 8. Blockers for submission — these need you

1. **Author block is a placeholder** — `[Author name(s) — to be completed]`.
2. **No venue named**, so template and page limit are unverified. 8 pages is
   long for a conference.
3. **No independent reproduction.** Everything was verified against artefacts
   in this repository by me. That is not the same thing, and the paper says
   so.

## 9. Exact paths

```
docs/SICKLE_Field_Level_Audit_IEEE.docx      manuscript, 8 pp
docs/SICKLE_Field_Level_Audit_IEEE.pdf       rendered, inspected
docs/REFERENCE_VERIFICATION.xlsx             20 refs, verification status
docs/MANUSCRIPT_AUDIT.md                     change log, §11 is tonight
docs/FINAL_VALIDATION_REPORT.md              passed / failed / skipped / blocked
docs/MORNING_HANDOFF.md                      this file
docs/REPRODUCE.md                            exact commands
docs/deliverable_hashes.json                 SHA-256 of every artefact
configs/yield_screen_protocol.yaml           frozen before results
results/triage/yield_screen.json             full experiment output
results/triage/yield_per_field.csv           41 field-seasons
results/triage/per_field_signals_v5.csv      36 crop-type fields
results/gate/diagnosis_misses.json           three-cell diagnosis
results/figures_ieee/fig_yield_screen.png    Fig. 7
scripts/yield_screen.py                      the experiment
scripts/independent_check.py                 imports nothing from src/
scripts/build_all.py                         one-command rebuild
tests/test_yield_screen.py                   16 new tests
```

## 10. Recommended next action

Give me the target venue. Then I can check the template and page limit, and
cut the paper to fit if it needs it. Everything else that can be done without
you is done.
