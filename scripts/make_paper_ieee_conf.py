"""The audit paper, six-page IEEE conference version.

    python scripts/make_paper_ieee_conf.py

Writes docs/SICKLE_Audit_IEEE_conf.docx. The full ten-page version
(scripts/make_paper_ieee.py, docs/SICKLE_Field_Level_Audit_IEEE.docx) is kept
for a journal; this one selects from it for a conference page limit.

Every number comes from make_paper_ieee.load(), the full version's loader, so
the two versions cannot disagree; the layout helpers, the citation
renumbering and the author block are shared too. Refuses to save if the text
claims the released yield models lose to a constant, reads the yield
correlation as season-level skill, or contains an un-negated overstatement
from the SICKLE-Guard claims registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

import make_paper_ieee as M                                  # noqa: E402
from make_paper_ieee import (spaced_percent, _cnt, _sgn, author_block, body, figure,  # noqa: E402
                             heading, lst, renumber, two_columns)

OUT = PROJ / "docs" / "SICKLE_Audit_IEEE_conf.docx"
TITLE = ("Aggregate Metrics Hide Field-Level Failures: A Reliability and "
         "Reproducibility Audit of the SICKLE Satellite Crop-Monitoring "
         "Benchmark")


def _nw(k):
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(k, str(k))


def build():
    n = M.load()
    tg = n["tg"]
    d50c = {r["target_workload"]: r
            for r in tg["low_confidence"]["leave_one_out"]}[0.5]
    lad = n["fd"]["ladder_crop_type"]
    q, b50 = lad["quantiles"], lad["below"]["0.5"]
    mcx = lad["macro_contrast"]
    fs = n["fs"]
    yv = n["yvc"]
    yb = yv["best"]
    dc = yb["decomposition"]
    bp, wp = dc["between_plots"], dc["within_plots"]
    ys = n["ys"]["threshold_free"]
    V = n["sv"]
    sd_, lc_, sf_ = (V["signals"][k] for k in
                     ("seed_disagreement", "low_confidence", "smallest_first"))
    pn = V["permutation_null"]
    C = n["cs"]
    t1 = C["test_1_conditional_permutation"]["low_confidence"]
    sa = C["test_2_out_of_sample"]["fit_free"]["screen alone"]
    f, w_ = n["bs_focus"], n["bs_worst_run"]
    fc, wc = n["bs_cells"][f], n["bs_cells"][w_]

    doc = Document()
    s0 = doc.sections[0]
    s0.left_margin = s0.right_margin = Inches(0.62)
    s0.top_margin = s0.bottom_margin = Inches(0.7)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(0)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(18)
    t.paragraph_format.space_after = Pt(10)
    author_block(doc)
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    sec.left_margin = sec.right_margin = Inches(0.62)
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    two_columns(sec)

    # ------------------------------------------------------------ abstract
    ab = doc.add_paragraph()
    ab.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    rr = ab.add_run("Abstract—")
    rr.bold = True
    rr.italic = True
    rr.font.size = Pt(9)
    rr = ab.add_run(
        "India has begun to automate crop-insurance payouts. Under PMFBY's "
        "YES-TECH framework, technology-based yield estimates carry at "
        "least 30 % of the weight in the yield that settles claims, "
        "Madhya Pradesh gave them 100 % in 2025-26, and claims are worked "
        "out automatically from the shortfall. The models behind such "
        "estimates are developed on research benchmarks that report "
        "accuracy pooled over pixels, while a payout concerns a field. We "
        "therefore ask what a benchmark number says about an individual "
        f"field, auditing {n['n_eval']} released SICKLE models across four "
        f"tasks. All {n['n_models']} regression checkpoints (yield and "
        f"dates) reproduce and {n['ct_pass']} of {n['ct_total']} crop-type "
        "cells do, but three defects undermine reproducibility, most "
        "visibly an evaluation of frozen weights that moves by up to "
        f"{100 * max(n['bs_spread'].values()):.1f} IoU points with batch "
        f"size alone; and none of the {len(n['yield_vs_const'])} released "
        "yield models is shown to beat a constant predictor. A pooled IoU of "
        f"{n['v5']:.4f} hides a concentrated tail: the median field scores "
        f"{n['median_field']:.3f}, but {n['n_below']} of {n['n_fields']} "
        "fields fall below 0.5 and the same fields recur across "
        "independent runs. Label-free safeguards do not reliably find "
        "them: for crop type, the stronger of two signals does not beat "
        "checking the smallest fields first, and for yield, ensemble "
        f"dispersion ranks error at AUROC {ys['auroc']:.2f} (95 % CI "
        f"{ys['auroc_ci']['lo']:.2f}–{ys['auroc_ci']['hi']:.2f}), no "
        "better than chance. Benchmarks that feed field-level decisions "
        "should report per-field error, and a safeguard must be validated "
        "on the quantity it gates. We audit a public benchmark, not any "
        "deployed system.")
    rr.font.size = Pt(9)
    kw = doc.add_paragraph()
    kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw.paragraph_format.space_before = Pt(5)
    kw.paragraph_format.space_after = Pt(6)
    rr = kw.add_run("Index Terms—")
    rr.bold = True
    rr.italic = True
    rr.font.size = Pt(9)
    rr = kw.add_run("crop monitoring, reproducibility, evaluation metrics, "
                    "per-field reliability, crop insurance, smallholder "
                    "agriculture, semantic segmentation.")
    rr.font.size = Pt(9)

    # ------------------------------------------------------------ I
    heading(doc, "I.  Introduction")
    body(doc, "India has begun to automate crop-insurance payouts. Under the "
              "Pradhan Mantri Fasal Bima Yojana [3], the YES-TECH framework "
              "gives technology-based yield estimates at least 30 % of the "
              "weight in the yield that settles a claim, with no upper cap "
              "[2]; in 2025-26 Madhya Pradesh gave them 100 % and "
              "Maharashtra 50 %, and claims are “automatically worked "
              "out based on the shortfall in yield” [21]. Farmers there "
              "contest the resulting assessments [5], [22].", 0.0)
    body(doc, "The assessment unit is the Gram Panchayat [2], but a village "
              "estimate is assembled from the fields in it: if a model errs "
              "systematically on some fields, the errors bias the village "
              "figure rather than cancel. The research benchmarks on which "
              "such models are developed report accuracy pooled over "
              "pixels, and so cannot show where those errors sit.")
    body(doc, "We audit SICKLE [1], smallholder paddy in the Cauvery Delta "
              "of Tamil Nadu, and ask what a published benchmark number says "
              "about an individual field. Our contributions are: (i) a "
              f"reproduction of {n['n_eval']} released models, which finds "
              "three defects that undermine reproducibility although most "
              "checkpoints reproduce; "
              "(ii) a field-level evaluation showing that the pooled score "
              "conceals a tail of failing fields that recurs across runs, "
              "and that the common two-class form of per-field IoU misstates "
              "single-crop fields; (iii) evidence that no released yield "
              "model is shown to beat a constant, and that the association "
              "it does have lies between fields, not between seasons; (iv) a "
              "controlled ablation in which the steps that add information "
              "exceed seed spread and added capacity does not; and (v) a test of label-free "
              "safeguards, weak for crop type and, in a pre-registered "
              "test, at chance for yield. We audit a public benchmark, not "
              "any deployed system, and make no claim about payouts or "
              "outcomes for farmers.")

    # ------------------------------------------------------------ II
    heading(doc, "II.  Related Work")
    body(doc, "That aggregate segmentation metrics are biased towards large "
              "objects is established, and per-object evaluation is its "
              "remedy [8]; more generally, aggregate metrics make it hard to "
              "predict how a system performs in a particular situation [25]. "
              "In smallholder regions, mapping accuracy differs by field "
              "type, with small and water-stressed fields the hardest [28]. "
              "Incomplete reporting obstructs reproduction [10], and "
              "run-to-run variance can exceed the differences a benchmark "
              "reports [26]. Disagreement among independently trained "
              "models is a standard uncertainty signal [11], and selective "
              "prediction defers uncertain cases [12]. For index insurance, "
              "yield data covering many fields but few years overstate "
              "insurance quality [27]. We claim none of these strands; what "
              "this paper adds is their measurement, together, on a "
              "benchmark whose objects are farm fields with an "
              "administrative consequence.", 0.0)

    # ------------------------------------------------------------ III
    heading(doc, "III.  Data and Methods")
    body(doc, "SICKLE's test split is unlabelled, so every number here is a "
              f"validation number: {n['n_fields']} fields for crop type and "
              "41 field-seasons from 14 plots for yield and the phenology "
              f"dates. We re-evaluated the {n['n_eval']} released models "
              "across four tasks with the authors' code and declared "
              "tolerances (0.01 IoU for crop type, 10 % relative for the "
              "regression tasks). Each field is scored on the crop or crops "
              "it holds: field IoU averages IoU only over the classes present "
              f"in the field's truth. {lad['n_single_crop_fields']} of the "
              f"{n['n_fields']} fields hold one crop, and the common "
              "two-class average caps any imperfect single-crop field at "
              "0.5, so it would misstate them. A field's pixels from all "
              "its seasons are pooled into one confusion matrix and fields "
              "are weighted equally [8]; for yield, a field-season's yield is "
              "the sum of its per-pixel values. Only binary paddy "
              "classification is evaluable, since 16 of the 21 crop labels "
              "have no validation plots.", 0.0)
    body(doc, "Evaluation uses a batch size of 32 in full precision "
              "throughout. Our own training uses an effective batch of 32, "
              "formed as micro-batches of 8 with four accumulation steps "
              "under bfloat16 autocast to fit an 8 GB device; this is held "
              "constant across rungs, so rung-to-rung differences stay "
              "controlled, but absolute values are not comparable with runs "
              "trained at a true batch of 32. The yield baseline is a "
              "constant issuing the "
              "training-split mean for every field-season, a comparison the "
              "benchmark does not report.")
    body(doc, "A ten-rung ablation (v1 to v10) changes one factor per rung, "
              "from a single "
              "acquisition through the temporal sequence, augmentation, "
              "class weighting, Sentinel-1 radar and Landsat-8 to a change "
              "of architecture (UTAE [18] for U-Net3D [17]), width, schedule "
              f"and sampling, each run at {n['seed_phrase']}. A step is "
              "inconclusive when its change is below the largest "
              "seed-to-seed standard deviation, a descriptive rule rather "
              "than a test; a paired-by-seed 95 % interval is reported "
              "beside it. Two label-free signals, seed disagreement and "
              "low confidence, are scored as screens for fields below 0.5 "
              "and compared with random referral and with checking the "
              "smallest fields first. Intervals resample fields or plots, "
              "never pixels; for yield, plots are resampled with their "
              "seasons kept together. The best yield model's correlation is "
              f"split into a between-plot part ({bp['n']} plot means) and a "
              f"within-plot part ({wp['n']} field-seasons with each plot's "
              "own mean removed).")

    # ------------------------------------------------------------ IV
    heading(doc, "IV.  Results")
    heading(doc, "A.  Reproduction and three defects", 2)
    body(doc, f"{n['n_pass']} of {n['n_models']} regression checkpoints "
              f"reproduce within tolerance, and {n['ct_pass']} of "
              f"{n['ct_total']} crop-type cells do; the other "
              f"{n['ct_total'] - n['ct_pass']} miss by "
              f"{min(n['ct_miss']):.3f} to {max(n['ct_miss']):.3f}. Three "
              "defects nonetheless undermine reproducibility: they affect "
              "how any score is evaluated, whether training runs, and which "
              "baselines are meaningful. First, "
              "evaluation depends on batch size: with frozen weights the "
              f"released {f} scores "
              + ", ".join(f"{fc[k]:.4f}" for k in sorted(fc, reverse=True))
              + " at batch sizes "
              + ", ".join(str(k) for k in sorted(fc, reverse=True))
              + f", and {w_} moves {100 * n['bs_spread'][w_]:.1f} IoU points "
                f"(from {min(wc.values()):.4f} to {max(wc.values()):.4f}), "
                "because the temporal pad mask is realigned by truncation "
                "and engages depending on which samples share a batch "
                "(Fig. 1). "
                "Second, the same realignment divides by zero on samples "
                "whose early acquisitions are empty (13 of 1,937 training "
                "samples), so training fails while reproduction succeeds. "
                "Third, three of the four released UTAE crop-type baselines "
                "never trained: all are best at epoch 1, and two predict one "
                "class everywhere. These are the three defects. A fourth "
                "finding is a labelling issue and is not counted as a "
                "defect: the phenology logs' column labelled MAPE equals MAE "
                "divided by the constant 183 in all 39 released date runs. "
                "That is the definition the dataset paper gives [1], so the "
                "values are correct, but they are not a percentage error and "
                "should not be compared with one.", 0.0)
    figure(doc, "fig_batch.png",
           "Fig. 1.  Evaluation of a frozen checkpoint depends on batch size: "
           "same weights, same data, unmodified code.", 2.4)

    heading(doc, "B.  What the pooled score conceals", 2)
    body(doc, f"Field-level results use v5 (Sentinel-2 with radar), the rung with "
              f"the highest plot-level score "
              f"({float(n['L']['v5']['plot_macro_iou']):.3f}); v6 adds Landsat-8 "
              f"and scores {n['d_l8']:.4f} higher pooled, within seed spread. "
              f"For v5 the pixel-pooled IoU is "
              f"{n['v5']:.4f}, close to the average field "
              f"({n['mean_field']:.3f}); the median field scores "
              f"{n['median_field']:.3f}. The shortfall is concentrated: "
              f"{b50['n']} of {n['n_fields']} fields fall below 0.5, a share "
              f"of {b50['frac']:.0%} with a 95 % interval of "
              f"{b50['ci'][0]:.0%}–{b50['ci'][1]:.0%} from a percentile "
              f"bootstrap over the {n['n_fields']} fields; with "
              f"{_nw(b50['n'])} such fields the estimate is necessarily "
              "imprecise. Fig. 2 shows "
              "them beside fields mapped almost perfectly. One field "
              f"supplies {100 * fs['share_largest']:.0f} % of the "
              f"{fs['n_pixels_in_pooled_metric']:,} scored pixels; removing "
              "it alone moves a released checkpoint's pooled figure by up to "
              f"{n['loo_max_abs']:.3f}. Scoring matters as much as pooling: "
              f"the two-class average reports a median field of "
              f"{mcx['median']:.3f} and {mcx['below']['0.5']} fields below "
              "0.5, most of them a few wrong pixels from perfect (Fig. 3).",
         0.0)
    figure(doc, "fig_satellite.png",
           "Fig. 2.  Sentinel-2 chips of SICKLE validation fields beside the "
           "prediction of the v5 seed-0 checkpoint, the run whose field "
           "scores the text reports; red marks wrong pixels inside the "
           "surveyed field. (a) Fields chosen by size, not by result. (b) "
           "The fields whose IoU averaged over three seeds is below 0.5, "
           "each at its season with the most wrong pixels. The first number "
           "is the field's IoU over all its seasons at seed 0; in (b) the "
           "three-seed mean follows and sets the colour. Chips are cropped "
           "around each field.")
    figure(doc, "fig_field_distribution.png",
           f"Fig. 3.  (a) Field IoU over the {n['n_fields']} validation "
           "fields at v5, with the pooled value. (b) The "
           "same as a quantile plot. (c) Field IoU against the pixels a field "
           "contributes; no trend is distinguishable from zero.")
    body(doc, "The same fields fail in every run: field IoU correlates "
              f"between every pair of seeds (r = {min(n['seed_r']):+.2f} to "
              f"{max(n['seed_r']):+.2f}), and {n['n_bad_always']} of the "
              f"{n['n_bad_fields']} failing fields are below 0.5 in all of "
              "them (Fig. 4). The pattern holds across tasks (Table I): for "
              "every regression task the typical field-season is better than "
              "the reported error and the worst decile far worse.")
    figure(doc, "fig_seeds.png",
           "Fig. 4.  Field IoU between two independent seeds; each point is "
           "a validation field.", 2.4)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    rr = cap.add_run("TABLE I\nReported versus per-field-season error, "
                     "medians across released models")
    rr.font.size = Pt(7.5)
    rr.bold = True
    tb = doc.add_table(rows=1, cols=5)
    tb.style = "Table Grid"
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(("Task", "Reported MAE", "Typical", "p90",
                           "p90/typical")):
        c = tb.rows[0].cells[i]
        c.text = h
        for p_ in c.paragraphs:
            for r_ in p_.runs:
                r_.bold = True
                r_.font.size = Pt(7.5)
    for key, lab, u in (("crop_yield", "Yield", "kg/acre"),
                        ("harvesting_date", "Harvest date", "days"),
                        ("transplanting_date", "Transplant date", "days")):
        v = n[key]
        row = tb.add_row().cells
        for i, val in enumerate((f"{lab} ({u})", f"{v['agg_mae']:.2f}",
                                 f"{v['median']:.2f}", f"{v['p90']:.2f}",
                                 f"{v['p90'] / v['median']:.2f}×")):
            row[i].text = val
            for p_ in row[i].paragraphs:
                for r_ in p_.runs:
                    r_.font.size = Pt(7.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    heading(doc, "C.  No released yield model is shown to beat a constant", 2)
    body(doc, "A constant issuing the training-split mean, "
              f"{n['const_value']:.0f} kg/acre, scores MAE "
              f"{n['const_mae']:.1f} on the yield validation set. All "
              f"{n['n_worse_mae']} released yield checkpoints have a higher "
              f"observed MAE; the best, {n['best_ckpt'][0]}, scores "
              f"{n['best_ckpt'][1]:.1f}, with R² = "
              f"{_sgn(n['best_ckpt'][3], '+.3f')}. These are point estimates "
              f"on {yv['n_field_seasons']} field-seasons from "
              f"{yv['n_plots']} plots. Resampling plots, the best "
              "checkpoint's excess MAE is "
              f"{_sgn(yb['mae_minus_constant'], '+.1f')} kg/acre (95 % CI "
              f"{_sgn(yb['ci95'][0], '+.1f')} to {_sgn(yb['ci95'][1], '+.1f')})"
              f", and none of the {yv['n_checkpoints']} intervals lies "
              "wholly above or below zero: no checkpoint is shown to beat "
              "the constant, and none is shown to be worse.", 0.0)
    body(doc, f"The best checkpoint correlates with the reference at "
              f"r = {_sgn(yb['pearson_r'], '+.3f')}, but that association "
              f"lies between plots: the {bp['n']} plot means correlate at "
              f"r = {_sgn(bp['r'], '+.3f')} (95 % CI "
              f"{_sgn(bp['ci95'][0], '+.2f')} to {_sgn(bp['ci95'][1], '+.2f')}"
              f"), while within plots the {wp['n']} field-seasons correlate "
              f"at r = {_sgn(wp['r'], '+.3f')} (95 % CI "
              f"{_sgn(wp['ci95'][0], '+.2f')} to {_sgn(wp['ci95'][1], '+.2f')}"
              "), although the estimate does move from season to season "
              f"(standard deviation {dc['within_plot_sd_pred']:.0f} against "
              f"{dc['within_plot_sd_ref']:.0f} kg/acre). The model ranks "
              "fields by their usual yield and shows no evidence of tracking "
              "a field's bad seasons, on which a claim turns [21]. The yield "
              "labels are farmers' recollections [1]; self-reports have "
              "elsewhere proved a poor basis for calibrating satellite "
              "models [29], [30], so recalled labels are a plausible "
              "contributor to these results, not a demonstrated "
              "explanation.")

    heading(doc, "D.  Ablation: which changes exceed seed spread", 2)
    # sigma_max is a descriptive threshold; the paired-by-seed interval is
    # the separate question of whether a change is distinguishable from 0.
    _st = {r_["step"]: r_ for r_ in n["lp"]["steps"]}
    _lab = {f"v{int(x['to'][1:]) - 1}->{x['to']}": x["label"]
            for x in n["pos"] + n["neg"] + n["inc"]}

    def _ex0(k):
        lo, hi = _st[k]["ci95_paired"]
        return lo > 0 or hi < 0

    _res = [k for k in _st if _ex0(k)]
    _gap = [k for k in _st if k in _lab and not _ex0(k)
            and k in {f"v{int(x['to'][1:]) - 1}->{x['to']}"
                      for x in n["pos"] + n["neg"]}]
    assert all(x["kind"] == "information" for x in n["pos"]), n["pos"]
    body(doc, f"Of the {n['n_steps']} ablation steps, "
              f"{len(n['pos']) + len(n['neg'])} exceed the seed-spread rule "
              f"(σmax = {n['sd_max']:.4f}): upward, " + lst(n["pos"])
              + "; downward, " + lst(n["neg"]) + f". The other "
              f"{len(n['inc'])} are inconclusive: " + lst(n["inc"])
              + " (Fig. 5). The rule is descriptive, not a significance "
              "test. Paired by seed, the 95 % interval excludes zero for "
              f"{_cnt(len(_res))} steps: "
              + lst([{"label": _lab[k], "delta": _st[k]["paired_mean"]}
                     for k in _res]) + ". "
              + "".join(
                  f"The {_lab[k]} step exceeds σmax at {n['seed_phrase']}, "
                  f"but with {_st[k]['n_seeds']} seeds its paired interval "
                  f"[{_sgn(_st[k]['ci95_paired'][0], '+.3f')}, "
                  f"{_sgn(_st[k]['ci95_paired'][1], '+.3f')}] includes zero, "
                  "so it is not resolved. " for k in _gap)
              + "Both upward steps add information; the steps that add "
              "capacity or change optimisation stay within seed spread. "
              "These are changes larger than run-to-run noise on this "
              "benchmark, not evidence of general superiority; inconclusive "
              "is not evidence of no effect, and because the order is "
              "cumulative each step is attributable only within this "
              "sequence.", 0.0)
    figure(doc, "fig_ladder.png",
           f"Fig. 5.  Ten-rung ablation at {n['seed_phrase']}; error bars "
           "are seed-to-seed standard deviation. The shaded rungs (v7 to v10) "
           "change the architecture, its width, the schedule or the "
           "sampling, not the information the model sees.")

    heading(doc, "E.  Can failing fields be found without labels?", 2)
    body(doc, f"With {tg['n_bad']} failing fields every rate is a statement "
              "about a handful of fields. Referring the half of fields with "
              f"the lowest confidence catches {d50c['n_caught']} of "
              f"{tg['n_bad']} at {d50c['precision']:.0%} precision (Fig. 6). "
              f"Seed disagreement ranks them at AUROC {sd_['auroc']:.3f} "
              f"(95 % CI {sd_['auroc_ci'][0]:.3f}–"
              f"{sd_['auroc_ci'][1]:.3f}), which includes chance; confidence "
              f"at {lc_['auroc']:.3f} ({lc_['auroc_ci'][0]:.3f}–"
              f"{lc_['auroc_ci'][1]:.3f}), which "
              + ("equals, and so does not exceed,"
                 if abs(lc_['auroc'] - pn['auroc_p97_5']) < 1e-9 else
                 "does not exceed" if lc_['auroc'] < pn['auroc_p97_5'] else
                 "exceeds")
              + " the 97.5th percentile of random referral's AUROC over "
              f"{pn['n_permutations']:,} orderings "
              f"({pn['auroc_p97_5']:.3f}); and checking the "
              f"smallest fields first, with no model, scores "
              f"{sf_['auroc']:.3f}. Against a null that preserves the size "
              f"relationship, confidence scores p = {t1['p_one_sided']:.3f}, "
              "so it carries information that size does not; but its gain "
              "over size has an interval of "
              f"[{_sgn(sa['ci_vs_smaller_first'][0], '+.3f')}, "
              f"{_sgn(sa['ci_vs_smaller_first'][1], '+.3f')}], and is not "
              "shown. The two "
              "results answer different questions and do not conflict: the "
              "permutation test asks whether confidence holds any "
              "information beyond field size, and it does; the interval asks "
              "whether ranking by confidence beats the operational baseline "
              "of checking the smallest fields first, and that is not "
              "established.", 0.0)
    figure(doc, "fig_triage.png",
           "Fig. 6.  Screening performance of two label-free signals for "
           "fields below 0.5. The diagonal is random referral; thresholds are "
           "set leave-one-out.")
    body(doc, "For yield, the quantity an assessment uses, a screen "
              "pre-registered before any result ranks field-season error by "
              f"the dispersion of the {n['ys']['n_checkpoints']} released "
              f"checkpoints at AUROC {ys['auroc']:.2f} (95 % CI "
              f"{ys['auroc_ci']['lo']:.2f}–{ys['auroc_ci']['hi']:.2f}), "
              "no better than chance (Fig. 7). A crop-type result must "
              "therefore not be assumed to carry over to yield.")
    figure(doc, "fig_yield_screen.png",
           "Fig. 7.  Pre-registered yield screen. Top: the dispersion signal "
           "against realised error, one marker per field-season. Bottom: "
           "recall against workload with plot-bootstrap intervals; the "
           "diagonal is random referral, which every interval includes. The "
           f"independent unit is the plot (n = {n['ys']['n_plots']}).")

    # ------------------------------------------------------------ V
    heading(doc, "V.  Discussion and Limitations")
    body(doc, "Because failing fields recur across runs, failure is not "
              "random, which is what would make it worth referring some "
              "fields for human verification; whether a label-free signal "
              "can choose them is not shown here, and on yield the "
              "pre-registered screen performs at chance. "
              "Benchmarks that feed field-level decisions should therefore "
              "report per-field error distributions and the evaluation "
              "batch size, and a confidence indicator should be shown to "
              "discriminate, on the quantity it gates, before it is shown to "
              "anyone: one that does not discriminate offers false "
              "reassurance with the authority of a number. The scheme's own "
              "documents leave this gap open: the YES-TECH manual specifies "
              "estimation and validation at the Gram Panchayat [2], and in its "
              "116 pages we found no requirement to validate below that unit "
              "and no binding accuracy floor, whereas an earlier, unversioned "
              "manual of January 2023 set an nRMSE ceiling of 30 % [20]. This "
              "describes the documents, not practice, which we cannot "
              "observe.", 0.0)
    body(doc, f"The evaluation set is small: {n['n_fields']} fields for crop "
              "type and 41 field-seasons from 14 plots for yield, from one "
              "delta and one cropping system in 2018-2021; intervals are "
              f"wide and no operating threshold is established. "
              f"{n['seed_phrase'].capitalize()} per rung afford limited power, so "
              "inconclusive rungs are not evidence of no effect. The yield "
              "labels are recalled, not measured. Our automated checks "
              "establish that the manuscript agrees with the saved "
              "artefacts, not independent reproduction by another "
              "researcher. We audit SICKLE, not YES-TECH's operational "
              "models, which are not public, and make no claim about their "
              "accuracy.")

    # ------------------------------------------------------------ VI
    heading(doc, "VI.  Conclusion")
    body(doc, "India has begun to automate crop-insurance payouts, and the "
              "yield behind them increasingly comes from models of the kind "
              "this benchmark is used to develop. On SICKLE, a benchmark "
              "number says less about a field than it appears to: "
              "evaluation shifts with batch size, a pooled IoU of "
              f"{n['v5']:.4f} hides a recurring tail of failing fields, no "
              "released yield model is shown to beat a constant, and "
              "label-free safeguards do not reliably find the failures. "
              "Per-field reporting, and safeguards validated on the quantity "
              "they gate, are what field-level decisions require.", 0.0)

    heading(doc, "Data and Code Availability")
    body(doc, "The SICKLE data and the released checkpoints are distributed "
              "by their authors on request [1] and are not redistributed "
              "here. Our code, configurations, per-run records, the scripts "
              "that regenerate every number and figure in this paper, and a "
              "checksum inventory of every checkpoint used are available from "
              "the authors on request and will be released publicly with the "
              "final version.", 0.0)
    heading(doc, "Acknowledgment")
    body(doc, "We thank the authors of SICKLE for making the dataset and "
              "their trained models available.", 0.0)

    # ------------------------------------------------------------ refs
    refs, uncited = renumber(doc, n["refs"])
    heading(doc, "References")
    for r_ in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Inches(0.22)
        pf.first_line_indent = Inches(-0.22)
        pf.space_after = Pt(2)
        loc = r_.get("doi") or ""
        only_locator = any(k in r_.get("type", "")
                           for k in ("Government", "Press", "Parliament",
                                     "Presentation"))
        tail = (f" [Online]. Available: {loc}"
                if loc.startswith("http") and only_locator
                else f" doi: {loc}." if loc and not loc.startswith("http")
                else "")
        run = p.add_run(f"[{r_['n']}] {r_['entry']}{tail}")
        run.font.size = Pt(7.6)
    return doc, refs, n


def self_check(text: str, n) -> list[str]:
    guard = Path(__import__("os").environ.get(
        "SICKLE_GUARD_ROOT", PROJ.parent / "sickle-guard"))
    sys.path.insert(0, str(guard / "src"))
    from check_claims import NEGATORS, norm
    low = norm(text).lower()
    problems = []
    reg = yaml.safe_load((guard / "configs/claims.yaml").read_text(
        encoding="utf8"))
    phrases = [str(r["phrase"]) for r in reg["prose_rules"]["must_not_appear"]]
    phrases += ["first study", "lose to a constant", "loses to a constant",
                "worse than a constant", "order field-seasons informatively",
                "causal"]
    for ph in phrases:
        ph_ = norm(ph.lower())
        i = low.find(ph_)
        while i >= 0:
            window = low[max(0, i - 110):i]
            if not any(neg in window for neg in NEGATORS):
                problems.append(f"un-negated {ph!r}")
            i = low.find(ph_, i + 1)
    dc = n["yvc"]["best"]["decomposition"]
    for v in (dc["between_plots"]["r"], dc["within_plots"]["r"]):
        if _sgn(v, "+.3f") not in text:
            problems.append(f"decomposition value {v:+.3f} not quoted")
    if "[author" in low or "to be completed" in low:
        problems.append("author placeholder left in")
    return problems


def main() -> int:
    doc, refs, n = build()
    text = "\n".join(p.text for p in doc.paragraphs)
    problems = self_check(text, n)
    if problems:
        print("CONFERENCE PAPER NOT WRITTEN:")
        for p in problems:
            print("  -", p)
        return 1
    spaced_percent(doc)
    doc.save(str(OUT))
    print(f"wrote {OUT}")
    print(f"  {len(text.split()):,} words, 7 figures, 1 table, "
          f"{len(refs)} references")
    print("  self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
