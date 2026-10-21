"""Cover letter, a courtesy e-mail to the SICKLE authors, and a submission
checklist for the Remote Sensing of Environment submission.

    python scripts/make_rse_letters.py

Numbers come from the same loader as the manuscript (make_paper_ieee.load),
so the letters cannot disagree with the paper. Outputs go to docs/rse/
(git-ignored). Nothing is sent anywhere: the e-mail is a draft for the
authors to review and send themselves.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

import make_paper_ieee as M                                   # noqa: E402
import make_paper_rse as R                                    # noqa: E402

OUTDIR = PROJ / "docs" / "rse"


def facts():
    n = M.load()
    f, w = n["bs_focus"], n["bs_worst_run"]
    fc, wc = n["bs_cells"][f], n["bs_cells"][w]
    big = sorted(k for k in wc if abs(wc[k] - max(wc.values())) < 1e-9)
    return {
        "n": n, "focus": f, "fc": fc, "worst": w, "wc": wc,
        "spread": 100 * n["bs_spread"][w], "big_from": big[0],
        "small_bs": min(wc), "ys": n["ys"]["threshold_free"],
    }


def cover_letter(F):
    n, ys = F["n"], F["ys"]
    corr = R.CORRESPONDING
    paras = [
        dt.date.today().strftime("%d %B %Y"),
        "The Editor-in-Chief\nRemote Sensing of Environment",
        "Dear Editor,",
        f"We submit the manuscript “{R.TITLE[0].upper() + R.TITLE[1:]}” "
        "for consideration as an Original Research Article in Remote "
        "Sensing of Environment.",
        "Satellite estimates of crop type and yield are now entering "
        "crop-insurance decisions in India, and the models behind them are "
        "developed and compared on research benchmarks that report accuracy "
        "pooled over pixels. A payout, however, concerns a single field. We "
        "therefore audited what one such benchmark, SICKLE (Sani et al., "
        "2024), says about individual fields.",
        f"We re-evaluated the {n['n_eval']} released models. All "
        f"{n['n_models']} regression checkpoints reproduce and {n['ct_pass']} "
        f"of {n['ct_total']} crop-type cells do, but evaluating frozen weights "
        f"moves intersection over union by up to {F['spread']:.1f} points with "
        f"batch size alone. A pooled IoU of {n['v5']:.4f} hides fields that "
        f"fail: {n['n_below']} of {n['n_fields']} fall below 0.5, the same ones "
        f"across independent runs. None of the {len(n['yield_vs_const'])} "
        "released yield models is shown to beat a constant predictor, and a "
        "pre-registered label-free screen for yield error performs at chance "
        f"(area under the ROC curve {ys['auroc']:.2f}).",
        "The paper fits the journal's scope in agriculture (crop mapping, "
        "yield prediction) and machine learning for remote-sensing data, and "
        "its contribution is the kind of statistically sound accuracy "
        "assessment the journal asks of every paper: plot-clustered "
        "intervals, a controlled ablation at several seeds, and a yield "
        "screen whose decision rule was fixed before its result was "
        "computed. Although the study "
        "uses one benchmark, its lessons concern evaluation practice, and "
        "SICKLE is in active use: Google Scholar listed 28 citing works on "
        "17 October 2026.",
        "This manuscript is original, has not been published, and is not "
        "under consideration elsewhere. All authors have approved it and "
        "declare no competing interests. The use of a generative AI tool in "
        "preparing the work is declared in the manuscript, as the journal "
        "requires. The SICKLE data are distributed by their authors on "
        "request and are not redistributed; our code and records are "
        "described in the Data availability statement.",
        "Thank you for considering our work.",
        "Yours sincerely,",
        f"{corr}\n(corresponding author, on behalf of all authors)\n"
        f"{R.AFFIL['a']}\n{R.EMAILS[corr]}",
    ]
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"
    d.styles["Normal"].font.size = Pt(12)
    for p in paras:
        d.add_paragraph(p)
    out = OUTDIR / "SICKLE_Audit_RSE_cover_letter.docx"
    d.save(out)
    return out


def email(F):
    n, fc, wc = F["n"], F["fc"], F["wc"]
    authors = " and ".join(name for name, _ in R.AUTHORS)
    text = f"""Subject: SICKLE: findings from an independent reproduction, before journal submission

Dear Dr Sani, Dr Anand and colleagues,

We are {authors}, at the Department of Data Science and Business Systems, SRM Institute of Science and Technology, India. Thank you for releasing SICKLE together with trained models; our study would not have been possible without them.

We have re-evaluated the {n['n_eval']} released models and are preparing a paper for Remote Sensing of Environment. Before we submit, we would like to tell you what we found, so that you can correct us or comment.

Most of the release reproduces: all {n['n_models']} regression checkpoints reproduce their reported errors within tolerance, and {n['ct_pass']} of {n['ct_total']} crop-type cells do. We found three issues.

1. Evaluation depends on batch size. In pastis_unet3d.forward the temporal pad mask is realigned to the pooled length by truncation (pad_mask[:, :final.shape[-1]]), so whether masking engages depends on which samples share a batch. With frozen weights, unmodified code and the same data, {F['focus']} scores {fc[32]:.4f}, {fc[16]:.4f} and {fc[8]:.4f} macro IoU at batch sizes 32, 16 and 8, and {F['worst']} ranges from {wc[F['small_bs']]:.4f} at batch {F['small_bs']} to {max(wc.values()):.4f} from batch {F['big_from']} upwards ({F['spread']:.1f} IoU points).

2. A zero denominator in the masked temporal mean. When a sample's early acquisitions are empty, the truncated mask can mark every remaining step as padding, and the division gives NaN. With Sentinel-2 at batch size 32 this affects 13 of 1,937 training samples and none of the 227 validation samples, so evaluation is unaffected but training receives NaN losses. Clamping the denominator at 1 left every batch that was already finite bit-identical in our tests.

3. Three of the four released UTAE crop-type baselines appear not to have trained: each has its best checkpoint at epoch 1, and [S2]_utae and [L8]_utae both score 0.3812 macro IoU with a paddy IoU of 0.000. We could not establish the cause. utae.py contains a similar unguarded division, but in the mean-aggregation branch, which the configuration we ran (agg_mode att_group) does not reach, so it does not explain this.

Two smaller notes. The phenology logs' column labelled MAPE equals MAE divided by 183 in all 39 released date runs, exactly as your paper defines it; we only point out that readers may take it for a percentage error. And in convlstm.py, (~pad_mask).sum(dim=1) - 1 evaluates to -1 for a fully padded sample, which selects the last timestep; we have not measured whether this occurs in practice.

Our paper also evaluates the benchmark field by field: the pooled crop-type IoU of {n['v5']:.4f} sits beside {n['n_below']} of {n['n_fields']} validation fields scoring below 0.5, and none of the {len(n['yield_vs_const'])} released yield models is shown to beat a constant predictor on the validation fields. We present these as limits of what a pooled benchmark score can show about individual fields, not as errors in your work.

We would be glad to share the manuscript and our code, including the runtime patch for item 2. If you see anything we have misread, we would be grateful to hear before we submit.

With thanks and best regards,

{R.AUTHORS[0][0]}
{R.AUTHORS[1][0]} (corresponding author)
Department of Data Science and Business Systems
SRM Institute of Science and Technology, Kattankulathur 603203, Tamil Nadu, India
{R.EMAILS[R.CORRESPONDING]}
"""
    out = OUTDIR / "email_to_SICKLE_authors.txt"
    out.write_text(text, encoding="utf8")
    return out


def checklist(F):
    n = F["n"]
    pending = "Code availability (item 5) is still open." \
        if R.CODE_AVAILABILITY is None else "No author decision is pending."
    text = f"""# RSE submission checklist (SICKLE audit)

Generated by scripts/make_rse_letters.py. Files are in docs/rse/.

## Files to upload
- Manuscript: SICKLE_Audit_RSE.docx (Word, single column, no line numbers)
- Highlights: SICKLE_Audit_RSE_highlights.docx
- Figures: figures/Figure_1.pdf ... Figure_9.pdf (vector); Figure_3 also as PNG (satellite imagery, 400 dpi)
- Cover letter: SICKLE_Audit_RSE_cover_letter.docx

## Entries in the submission system
- Article type: Original Research Article
- Title: {R.TITLE[0].upper() + R.TITLE[1:]}
- Authors, in this order (given name / family name):
  1. Pharos Sophy Samuel T J (exactly as on the Aadhaar card): first/given name "Pharos Sophy", last/family name "{R.FAMILY['Pharos Sophy Samuel T J']}". The byline then prints "Pharos Sophy Samuel T J"; citations will index the family name as "{R.FAMILY['Pharos Sophy Samuel T J']}". Check the byline in the PDF proof the system builds.
  2. S. Ganesh Kumar: family name {R.FAMILY['S. Ganesh Kumar']}; given name "S. Ganesh". Corresponding author, {R.EMAILS['S. Ganesh Kumar']}
- Affiliation (both): {R.AFFIL['a']}
- Keywords: {'; '.join(R.KEYWORDS)}
- Declarations tool: no competing interests; no specific funding.
- Generative AI: declared in the manuscript (section before the references).
- Preprint on SSRN: optional; decide before submitting.

## Before submitting
- [ ] {pending}
- [ ] An independent reproduction run of docs/REPRODUCE.md by someone else.
- [ ] Courtesy e-mail to the SICKLE authors (draft: email_to_SICKLE_authors.txt), sent by you, ideally with time for a reply.
- [ ] The IEEE conference version of this audit is NOT submitted anywhere (the cover letter states the work is not under consideration elsewhere).
- [ ] Shahin and Pradhan (2026) is "in press": add volume and pages if assigned.
- [ ] Adjei (2026) is an arXiv preprint: check for a published version.
- [ ] Open the APSAC link (MNCFC, 2023b) in a browser; it was unreachable on 2026-10-16.
- [ ] Gong et al. (2025): read the full text if access allows; the paper cites only what its abstract states.
- [ ] Final build: python scripts/make_paper_rse.py --final (refuses while any decision is pending).
- [ ] Optional: a graphical abstract (at least 1328 px wide, readable at 13 cm).

## Numbers quoted in the cover letter and e-mail (from the manuscript loader)
{n['n_eval']} models re-evaluated; {n['n_models']}/{n['n_models']} regression checkpoints and {n['ct_pass']}/{n['ct_total']} crop-type cells reproduce; batch-size swing up to {F['spread']:.1f} IoU points; pooled IoU {n['v5']:.4f}; {n['n_below']}/{n['n_fields']} fields below 0.5; {len(n['yield_vs_const'])} yield models, none shown to beat a constant; yield screen AUROC {F['ys']['auroc']:.2f}.
"""
    out = OUTDIR / "SUBMISSION_CHECKLIST.md"
    out.write_text(text, encoding="utf8")
    return out


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    F = facts()
    assert F["fc"].keys() >= {8, 16, 32}, F["fc"]
    for f in (cover_letter(F), email(F), checklist(F)):
        print("wrote", f)
