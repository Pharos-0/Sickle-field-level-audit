# Final validation report

Run 2026-10-11 on the machine holding this repository (Windows 11, Python 3.12,
project virtualenv `.venv`).

Scope note, stated first because it bounds everything below: **a passing check
here means the specified check passed against the artefacts in this
repository.** It does not establish that an independent researcher reproduced
the work. No such independent reproduction has been performed.

---

## 1. Deliverables

| # | File | State | Detail |
|---|---|---|---|
| 1 | `docs/SICKLE_Field_Level_Audit_IEEE.docx` | PRODUCED | 8 pages, 7,034 words, two-column IEEE layout |
| 2 | `docs/SICKLE_Field_Level_Audit_IEEE.pdf` | PRODUCED | exported via Word COM, 8 pages, inspected page by page |
| 3 | `docs/REFERENCE_VERIFICATION.xlsx` | PRODUCED | 20 references — 15 peer-reviewed, 5 government/press |
| 4 | `docs/MANUSCRIPT_AUDIT.md` | PRODUCED | change log with evidence per change |
| 5 | `docs/FINAL_VALIDATION_REPORT.md` | this file | |

Supporting: `docs/references.json` (authoring numbering, hand-maintained),
`docs/references_final.json` (generated, the manuscript's numbering),
`docs/deliverable_hashes.json` (SHA-256 prefixes and sizes of all of the
above plus the six figures), `results/figures_ieee/*.png` (6 figures,
3.35 in, 400 dpi).

## 2. Checks that were run

### 2.1 Claim verification — `scripts/verify_all.py`

**18 / 18 PASS.** Each check names the file it read.

| Claim | Result | Detail |
|---|---|---|
| D1 batch-size dependence | PASS | IoU spread 0.0661 across batch sizes 8/16/32 |
| D3 degenerate UTAE baselines | PASS | 2 cells with paddy IoU exactly 0 |
| D4 date "MAPE" == MAE/183 | PASS | **39/39** released date runs match exactly (an arbitrary `[:4]` cap was removed this round; see audit §8.2) |
| Reproduction (regression) | PASS | 48/48 gates (4 `_debug` smoke tests excluded) |
| Ladder: all adjacent steps partitioned | PASS | 9 steps = 4 exceeding σmax 0.0417 (2 up, 2 down) + 5 inconclusive |
| Audited count excludes 2-epoch smoke tests | PASS | 4 `_debug` runs excluded from the 48 reported |
| Reproduction (crop type) | PASS | 9/12 cells at the declared tolerance |
| Ladder: temporal exceeds seed spread | PASS | +0.1198 vs max seed sd 0.0417 |
| Ladder: radar exceeds seed spread | PASS | +0.1860 vs max seed sd 0.0417 |
| Ladder: three seeds per rung | PASS | seeds per rung = 3 |
| Triage: LOO recall reported with CI | PASS | 92 %, CI [68 %, 100 %] |
| Triage: sample size stated honestly | PASS | 36 fields, 12 unreliable → 1 field = 8.3 pp |
| Provenance: runs record a clean commit | PASS | 39/50 manifests `git_dirty=false` |

Note on the last row: 11 of 50 manifests record a dirty tree. The check passes
because it requires a majority of clean manifests, not all of them. Those 11
runs are not provenance-clean and this is not presented as if they were.

Note on "Reproduction (crop type)": the check passes at 9/12, and the
manuscript reports 9/12 explicitly rather than describing all cells as
verified. See `MANUSCRIPT_AUDIT.md` §2.

Note on the audited total: it is **60 models, of which 57 reproduce**, not 64.
The released tree ships 2-epoch `_debug` smoke tests that an earlier draft
counted as baselines. See `MANUSCRIPT_AUDIT.md` §8.1.

### 2.2 Unit tests — `pytest`

**71 / 71 passed**. This includes 16 new tests over the yield experiment (AUROC tie handling, leave-one-plot-out leakage, plot-level bootstrap, CV and APE denominators, protocol freeze, export alignment), and includes
`tests/test_geographic_holdout.py`, which pins the leave-one-out property by
spiking a held-out region's own values and asserting its threshold does not
move.

### 2.3 Reference verification

All 18 entries re-checked in this phase. Method per entry is recorded in
`docs/references.json` and reproduced in the spreadsheet.

- 11 verified through the Crossref REST API by DOI (title, authors, venue,
  year, and where present volume/issue/pages read from the returned record).
- 4 verified by retrieving the proceedings page (HTTP 200) and reading title
  and author list from it: Wang 2023, Lakshminarayanan 2017, Geifman &
  El-Yaniv 2017, Shi 2015.
- 4 government/press sources verified by retrieving the document itself. The
  YES-TECH manual was downloaded, OCR'd in full (116 pages) and the quoted
  passages located by page; its three quoted clauses were re-verified verbatim
  in the second round. Both Down To Earth articles were retrieved and read in
  full on 2026-10-11.

**Four entries were found to be wrong or incomplete and were corrected** in
the first round (SICKLE page range, Ploton initial, Koukos initial and author
list, Reuss author list) — `MANUSCRIPT_AUDIT.md` §3.

**A fifth was found in the second round and was more serious**: the bundled
press reference carried a figure ("344 districts across 12 states") that a
direct read shows is in **neither** cited article, and gave 2025 as the year
when both are 2026. The figure has been removed from the manuscript; the
reference is split into two, each tied to the claim it actually supports —
`MANUSCRIPT_AUDIT.md` §8.3.

Nothing in the reference list was written from memory.

### 2.4 Citation integrity

Extracted from the rendered PDF: citation numbers present are exactly
1–18, contiguous, with no gaps and nothing cited that is not listed. Numbering
follows order of first appearance. Build idempotence verified: two consecutive
builds produce byte-identical `references_final.json`.

### 2.5 Forbidden-claim sweep

Rendered PDF text searched for `causal`, `first study`, `no previous work`,
`novel architecture`, `state-of-the-art`, `proves`, `guarantee`,
`significantly outperform`. One hit, in the sentence that disclaims causality.
No forbidden claim present.

### 2.6 Visual inspection

Every page rasterised at 105 dpi and inspected individually after each
rebuild. Two figure defects were found this way and fixed (Fig. 3 overlaid the
mean while the text quoted the median; Fig. 4's panels collided). An earlier
six-page render whose last page held only two reference lines was reduced to
five by removing proceedings retrieval URLs from the printed citations; the
manuscript has since grown back to **6 pages** as corrections were added, and
its final page carries roughly half a column of references rather than a
near-empty page. Rendering has been through Word only; no second DOCX-to-PDF
renderer was available (see §3).

## 3. Checks that were NOT run, and why

| Check | State | Why |
|---|---|---|
| Independent reproduction by another person or machine | **NOT RUN** | Requires a second person; none was available. This remains the single largest gap. |
| Clean-environment rebuild from `requirements.txt` | **RUN — partial pass, see §4** | A fresh venv built from the documented procedure. Dependency resolution passes; the environment cannot import torch on this machine. |
| LibreOffice cross-renderer check of the PDF | **NOT RUN** | LibreOffice is not on PATH on this machine. The PDF was produced and inspected through Word only. |
| Plagiarism / similarity screening | **NOT RUN** | No similarity service available in this environment. |
| Grammar/style pass by a tool | **NOT RUN** | Done by reading only. |
| Diagnosis of the 3 non-reproducing crop-type cells | **DONE** | All three swept across batch size. One explained by batch composition, two not. See `MANUSCRIPT_AUDIT.md` §11.3. |
| Yield-reliability screening (Section V-H) | **DONE — negative** | Pre-registered protocol; AUROC 0.311, CI [0.103, 0.634] contains chance. Reported as not demonstrated. |
| Verification of the press figures (344 districts, 12 states, MP at 100 %) | **NOT POSSIBLE** from the manual | They do not appear in the YES-TECH manual. The manuscript attributes them to press reporting and states they are not independently verified. |
| Author reading of SICKLE and Wang et al. in full | **NOT DONE** | Outstanding task for the author. Wang et al. determines what in this paper is and is not novel. |

## 4. Clean-environment rebuild (run 2026-10-11)

A fresh virtualenv was created and the documented procedure followed exactly:
torch pinned from the CUDA index first, then `requirements.txt`. Reported
without rounding up.

**What passed.**

- The install completes. `pip` resolves the full requirement set with no
  conflict.
- **The torch pin holds.** After installing the complete `requirements.txt`,
  including `timm` 1.0.30 — the package that previously replaced the working
  build with `torch 2.14.1+cpu` — `pip list` reports `torch 2.11.0+cu128` and
  `torchvision 0.26.0+cu128`. The pin does what the file's comment says it
  does. This was the specific risk flagged in the previous report.
- `scripts/verify_all.py` run with the clean interpreter: **18/18 PASS**. It
  is stdlib-only, so this is a genuine fresh-interpreter run.
- `pytest` with the clean interpreter: **46 passed, 1 skipped** (the skip is
  the torch-dependent file, for the reason below).

**What failed, and why.**

`import torch` raises `OSError: [WinError 4551] An Application Control policy
has blocked this file` on `torch/lib/c10.dll`. Windows Smart App Control
blocks the freshly written DLLs. This was reproduced in two locations — under
`AppData\Local\Temp` and on the `D:` drive beside the project — so it is the
policy reacting to newly downloaded binaries, not a path problem. The
project's existing `.venv` is unaffected because its DLLs predate the block.

Consequently `tests/test_patches.py` (9 tests covering the NaN fix) **could
not run** in the clean environment. It also did not skip cleanly:
`pytest.importorskip("torch")` catches `ImportError`, not `OSError`, so
collection aborted for the whole suite. **This was fixed**: the gate now
catches `OSError` too. The clean environment reports **46 passed, 1 skipped**;
the project venv still reports **55 passed**.

**What this does and does not establish.** It establishes that the documented
dependency specification is correct and sufficient, and that every claim check
passes from an interpreter that shares nothing with the working environment.
It does **not** establish that the GPU pipeline can be rebuilt from scratch on
this machine, and it is not independent reproduction by another person. The
`_cleanenv_verify` venv is left at `D:/FPD/_cleanenv_verify` and can be
deleted.

## 5. Known blockers before submission

1. **Author block is a placeholder.** `[Author name(s) — to be completed by the
   author]` and `[email — to be completed]` appear on page 1.
2. **No independent reproduction.** Section VII already says the verification
   script confirms agreement with this repository's artefacts and is not a
   substitute for independent reproduction, but the gap is real.
3. **The author has not read the two load-bearing sources in full.**
4. **An OCR gap in the manual.** A sentence in the passage on past estimates
   ends "...court of law or appellate authority", but page interleaving in the
   OCR has lost its subject. It is **not quoted or relied on**. If the legal
   framing matters to the final version, read that page of the PDF directly.
5. ~~Unresolved source conflict over which YES-TECH manual governs.~~
   **RESOLVED 2026-10-11** — see `MANUSCRIPT_AUDIT.md` §9.2. The 116-page
   V1.0 manual of July 2023 governs; the 48-page January 2023 document is
   earlier and unversioned. The manuscript now cites both and states that the
   governing manual carries no binding accuracy floor.

## 6. Overnight round: the yield extension

A new experiment was added and it **failed**, which is reported in the paper
with the same prominence a positive result would receive.

- Protocol frozen in `configs/yield_screen_protocol.yaml` before any result
  existed, including the threshold, baselines and rejection criteria.
- 24 checkpoints, 41 field-seasons, **14 independent plots**, 8 unreliable.
- AUROC **0.311**, 95 % CI [0.103, 0.634]. The interval contains chance, so
  the signal is **not demonstrated**; recall is below random referral at all
  four workloads.
- Robustness: threshold sensitivity (AUROC 0.219–0.570 across five
  thresholds), model subsets (0.312, 0.458), two baselines (0.367, 0.591).
- Ensemble members correlate at a median of **0.924**, so the finding is
  scoped to this ensemble and the quantity is called inter-model
  disagreement, not uncertainty.

The independent checker re-derived AUROC 0.311 and the interval
[0.103, 0.634] with its own implementation, sharing no code with the
experiment.

## 7. Statement of what the evidence supports

The manuscript's numerical claims are each backed by a file in this
repository, and `scripts/verify_all.py` re-derives them from those files in a
single command. That is the strength of the evidence and the limit of it. The
study measures model error on a public research benchmark. It does not measure
payouts, farmer incomes or yields, it does not audit any operational
YES-TECH model, and the manuscript makes no claim about any of those.
