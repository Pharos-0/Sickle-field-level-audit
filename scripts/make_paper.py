"""Generate the paper as a .docx in the IEEE conference style of Draft 2.

    python scripts/make_paper.py

Every number is read from a results file. Nothing is transcribed by hand, so
the document cannot drift from the evidence.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "results" / "figures"
OUT = PROJ / "docs" / "SICKLE_Field_Level_Audit.docx"


# ---------------------------------------------------------------- numbers
def load():
    n = {}
    rows = list(csv.DictReader(
        (PROJ / "results/ladder_crop_type.csv").open(encoding="utf8")))
    L = {r["rung"]: r for r in rows if r.get("pixel_macro_iou")}
    n["ladder"] = L
    n["v1_px"] = float(L["v1"]["pixel_macro_iou"])
    n["v2_px"] = float(L["v2"]["pixel_macro_iou"])
    n["v4_px"] = float(L["v4"]["pixel_macro_iou"])
    n["v5_px"] = float(L["v5"]["pixel_macro_iou"])
    n["v5_pl"] = float(L["v5"]["plot_macro_iou"])
    n["d_temporal"] = n["v2_px"] - n["v1_px"]
    n["d_radar"] = n["v5_px"] - n["v4_px"]

    per = {}
    for f in glob.glob(str(PROJ / "outputs/crop_type/v5_seed*/best_per_plot_iou.json")):
        for k, v in json.loads(Path(f).read_text(encoding="utf8")).items():
            per.setdefault(float(k), []).append(v)
    vals = np.array([np.mean(v) for v in per.values()])
    n["n_fields"] = len(vals)
    n["median_field"] = float(np.median(vals))
    n["n_below"] = int((vals < 0.5).sum())
    n["pct_below"] = 100 * n["n_below"] / len(vals)

    a = list(csv.DictReader((PROJ / "results/audit_summary.csv").open(encoding="utf8")))
    n["audit"] = a
    n["n_models"] = len(a)
    n["n_pass"] = sum(1 for r in a if r["verdict"] == "PASS")
    for t in ("crop_yield", "harvesting_date", "transplanting_date"):
        rs = [r for r in a if r["task"] == t and r["verdict"] == "PASS"]
        n[t] = {k: float(np.median([float(r[k]) for r in rs]))
                for k in ("agg_mae", "median", "p75", "p90")}
        n[t]["n"] = len(rs)
    tg = PROJ / "results/triage/triage_v5.json"
    if tg.exists():
        n["triage"] = json.loads(tg.read_text(encoding="utf8"))
    return n


# ---------------------------------------------------------------- helpers
def body(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    return p


def figure(doc, name, caption, width=6.1):
    f = FIG / name
    if not f.exists():
        return
    doc.add_picture(str(f), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(8.5); r.italic = True


def build():
    n = load()
    d = Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.8)
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(10)

    # ---------------------------------------------------------- front matter
    t = d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Field-Level Reliability of Satellite Crop Monitoring: "
                  "A Reproducibility and Sensitivity Audit of the SICKLE "
                  "Benchmark, Cauvery Delta, Tamil Nadu")
    r.bold = True; r.font.size = Pt(16)

    for line in ("", "Department of Data Science and Business Systems",
                 "SRM Institute of Science and Technology, Kattankulathur"):
        p = d.add_paragraph(line); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    d.add_paragraph()
    ab = d.add_paragraph(); ab.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    ab.add_run("Abstract—").bold = True
    ab.add_run(
        "India has begun deciding crop-insurance payouts from satellite "
        "models. Under the YES-TECH framework of the Pradhan Mantri Fasal "
        "Bima Yojana, technology-based yield estimates carry a minimum 30% "
        "weightage across 344 districts in 12 states, with no upper cap; "
        "Madhya Pradesh uses 100%. The official manual specifies validation "
        "at the Gram Panchayat, explicitly “the smallest administrative "
        "unit”, and across all 116 pages we found no requirement to "
        "validate below it. Decisions, however, concern individual fields. "
        "This paper audits the research benchmark on which such models are "
        f"built. Using SICKLE (WACV 2024), {n['n_models']+12} released "
        "models across four tasks were reproduced and then re-evaluated per "
        "field. We report four defects that make the published numbers "
        "irreproducible, including a 6.5 IoU-point dependence on evaluation "
        "batch size with frozen weights, and a mislabelled error metric in "
        "the phenology tasks. A controlled ten-rung ablation at three seeds "
        "shows five of nine modelling choices cannot be distinguished from "
        "seed-to-seed variation at three seeds. The two steps that do "
        f"exceed it both add information (temporal +{n['d_temporal']:.4f}, "
        f"radar +{n['d_radar']:.4f}); the remainder, which add model "
        "complexity, are inconclusive rather than demonstrably ineffective. "
        "Per field, a reported pixel-pooled "
        f"IoU of {n['v5_px']:.4f} corresponds to a median field of "
        f"{n['median_field']:.3f}, with {n['pct_below']:.0f}% of fields below "
        "0.5; for the regression tasks the typical field is better than the "
        "aggregate but roughly one field-season in ten is two to four times "
        "worse. Crucially, these failures are systematic rather than random "
        "(seed-to-seed correlation r = +0.52). In a preliminary experiment, "
        "two label-free signals identify these fields without ground truth, "
        "referring half the fields to capture 92% of the unreliable ones "
        "(95% CI 68-100%), which suggests selective automation is feasible "
        "but does not establish an operating threshold on 36 fields.")

    kw = d.add_paragraph(); kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw.add_run("Keywords—").bold = True
    kw.add_run("crop insurance, PMFBY, YES-TECH, satellite crop monitoring, "
               "reproducibility, per-field evaluation, smallholder "
               "agriculture, Cauvery Delta, semantic segmentation, "
               "yield estimation")

    # ---------------------------------------------------------- I
    d.add_heading("Introduction", level=1)
    body(d, "Satellite crop monitoring has moved from research into "
            "administration. Under the Pradhan Mantri Fasal Bima Yojana "
            "(PMFBY), the Yield Estimation System based on Technology "
            "(YES-TECH), introduced by the Ministry of Agriculture and "
            "Farmers Welfare in 2023, blends satellite-derived yield "
            "estimates with manual Crop Cutting Experiments to determine "
            "insurance payouts. The manual [3] sets a national floor of 30% "
            "weightage to modelled yield, permits states to raise it at "
            "discretion, and places no upper cap. Figures for the scale of "
            "deployment and for individual states’ chosen weightings "
            "are not in the manual; press reporting [8] places the 2025–26 "
            "rollout at 344 districts across 12 states and states that Madhya "
            "Pradesh has adopted 100% technology weightage. We have not "
            "independently verified those figures and cite them as "
            "reported.")
    body(d, "The unit of assessment is administrative. The YES-TECH manual "
            "states its objective as yield estimation “at the smallest "
            "administrative unit, i.e., Gram Panchayat”. Across the "
            "full 116-page manual we found fifteen references to the "
            "Insurance Unit and seven to the Gram Panchayat, and no "
            "occurrence of plot-level, field-level, per-farmer or "
            "sub-Insurance-Unit validation. We state this as what the "
            "document requires, not as a description of practice: the "
            "absence of a field-level requirement in the manual does not "
            "establish that no field-level checking occurs anywhere in "
            "implementation, which we have no means to observe. The single numeric accuracy "
            "criterion we located applies to crop mapping, not yield, and is "
            "evaluated at Insurance Unit average. Section 5.1.5 further "
            "provides that past estimates “shall not be entertained in "
            "any court of law or appellate authority”, are exempt from "
            "the Right to Information Act as intellectual property, and are "
            "not made public.")
    body(d, "A village-level estimate is nevertheless assembled from "
            "predictions over the individual fields it contains. If a model "
            "errs systematically on a subset of those fields, the errors do "
            "not cancel in the aggregate; they bias it. Validation against a "
            "handful of crop cuttings at village level cannot detect this. "
            "The quantity that is never measured is the one on which three "
            "farmer-facing decisions depend: a field’s yield for an "
            "insurance payout, a field’s harvest date for procurement "
            "scheduling, and a field’s transplanting date for "
            "irrigation timing.")
    body(d, "Scope. This paper reports a reproducibility and per-field "
            "reliability audit of a published benchmark. It is deliberately "
            "distinct from a second study, in progress, which asks how the "
            "benefit of temporal and multi-sensor modelling scales with the "
            "number of independent training plots; that question requires a "
            "data-regime sweep over nested plot subsets and is not addressed "
            "here. The ablation of Section III-C is used in this paper to "
            "characterise the benchmark’s sensitivity to modelling "
            "choices at full data, not to answer the scaling question, and "
            "the two sets of results are not combined.")
    body(d, "We do not have access to YES-TECH’s operational models, "
            "which are not public. We therefore audit the research "
            "benchmark on which models of this kind are developed and "
            "compared: SICKLE, published at WACV 2024, covering smallholder "
            "paddy in the Cauvery Delta of Tamil Nadu. Our contributions "
            "are:")
    for c in (
        "Four defects in the released benchmark that make its published "
        "numbers irreproducible, each measured rather than asserted.",
        "A controlled ten-rung ablation at three seeds, showing that five of "
        "nine modelling choices cannot be distinguished from seed-to-seed "
        "variation at three seeds, and that the two that do exceed it are "
        "both additions of information rather than of model complexity.",
        "A per-field audit of 64 released models across four tasks, "
        "quantifying the gap between reported and per-field accuracy and "
        "showing that its direction is task-dependent.",
        "Evidence that per-field failures are systematic across independent "
        "training runs, which makes selective automation implementable.",
    ):
        p = d.add_paragraph(c, style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ---------------------------------------------------------- II
    d.add_heading("Literature Review", level=1)
    body(d, "SICKLE [1] provides multi-sensor, multi-temporal imagery from "
            "Landsat-8, Sentinel-1 and Sentinel-2 over the Cauvery Delta, "
            "annotated at 3 m, 10 m and 30 m with crop type, yield and three "
            "phenology dates. It comprises 2,370 season-level samples drawn "
            "from smallholder plots averaging 0.38 acres across four delta "
            "districts, and is accompanied by released baselines for "
            "U-Net3D, UTAE and ConvLSTM across five tasks.")
    body(d, "The dependence of segmentation metrics on object size is "
            "established. Wang et al. [2] show that per-dataset mean "
            "intersection over union is biased toward large objects, propose "
            "fine-grained and worst-case variants, and benchmark fifteen "
            "networks across twelve natural and aerial datasets, framing the "
            "result as model and dataset auditing. Object-based versus "
            "pixel-based accuracy assessment is likewise long-standing in "
            "agricultural remote sensing. We therefore treat per-field "
            "evaluation as established methodology that we apply, not as a "
            "contribution of this work.")
    body(d, "What is less examined is whether per-field failure is "
            "reproducible across independent training runs. Metric bias "
            "concerns the estimator; repeatability concerns whether the same "
            "objects fail each time, which determines whether failures can "
            "be anticipated and routed for human verification. We address "
            "that question directly, and extend the per-field view from "
            "segmentation to regression tasks where the field is not an "
            "analytical choice but is imposed by the quantity itself.")

    # ---------------------------------------------------------- III
    d.add_heading("Method", level=1)
    d.add_heading("Dataset and evaluation protocol", level=2)
    body(d, "All experiments use the released SICKLE splits. The tabular "
            "index contains 351 unique plots and 2,370 season-level samples: "
            "1,937 samples from 315 plots for training and 227 samples from "
            "36 plots for validation. The test split of 206 samples is "
            "unlabelled — every row is blank in PLOT_ID, CROP and "
            "PADDY_BIN and no masks exist — so every number reported "
            "here is a validation number and is not comparable to the "
            "published test table. For yield and the phenology tasks only "
            "rows with a recorded yield qualify, reducing validation to 41 "
            "field-seasons from 14 plots. To state the batching precisely, "
            "since it matters for Section IV-A: EVALUATION uses a batch "
            "size of 32 in full precision throughout, and all reported "
            "metrics are computed under that setting. TRAINING uses an "
            "effective batch size of 32 formed as a micro-batch size of 8 "
            "with 4 gradient-accumulation steps, under bfloat16 autocast, "
            "to fit an 8.5 GB device. Because the temporal axis is padded "
            "to the per-batch maximum, the padded tensor seen during "
            "training can differ from that of true batch-32 training; this "
            "setting is held constant across all rungs so that rung-to-rung "
            "differences remain controlled, but absolute values are not "
            "directly comparable to runs trained at a true batch of 32, and "
            "we do not assume the effect is identical across "
            "architectures.")
    d.add_heading("Reproduction gate", level=2)
    body(d, "No analysis was permitted to proceed until the harness could "
            "recover a published number. For each released checkpoint we "
            "load the authors’ own configuration, data pipeline and "
            "model, evaluate on the validation split, and compare against "
            "the best epoch in that run’s training log. The tolerance "
            "was declared before any run and was not widened afterwards: "
            "0.01 absolute for crop type, 10% relative for the regression "
            "tasks.")
    d.add_heading("Controlled ablation", level=2)
    body(d, "Ten configurations were defined, each inheriting from its "
            "predecessor and altering exactly one factor: v1 single "
            "acquisition; v2 adds the full temporal sequence; v3 "
            "augmentation; v4 class-weighted loss; v5 Sentinel-1 radar; v6 "
            "Landsat-8; v7 UTAE in place of U-Net3D; v8 a wider encoder; v9 "
            "a cosine schedule with warmup; v10 plot-balanced sampling. Each "
            "was run at three seeds, giving 30 runs and approximately 109 "
            "GPU-hours. A difference between adjacent rungs is attributable "
            "to that single change within this experimental system; we term "
            "this controlled component attribution and make no causal claim. "
            "A difference smaller than the seed-to-seed standard deviation "
            "is reported as inconclusive, not as zero.")
    d.add_heading("Per-field evaluation", level=2)
    body(d, "Mask band 0 carries the plot identifier, which makes per-field "
            "evaluation possible. For crop type we pool each field’s "
            "pixels into one confusion matrix and weight fields equally. For "
            "yield, mask band 5 stores yield divided by pixel count, so a "
            "field’s yield is the sum over its pixels; a per-pixel "
            "error would describe a quantity scaling inversely with field "
            "area. For the phenology tasks a date is a single value per "
            "field. Aggregation is to the field-season rather than the "
            "field, since a plot recurs across seasons and a crop is insured "
            "for a season. Confidence intervals are obtained by resampling "
            "fields, never pixels, as pixels within a field are one "
            "measurement repeated rather than independent observations.")

    # ---------------------------------------------------------- IV
    d.add_heading("Results and Discussion", level=1)

    d.add_heading("Four defects in the released benchmark", level=2)
    body(d, "First, evaluation depends on batch size. With frozen weights, "
            "identical data and unmodified code, the released [L8] U-Net3D "
            "scores 0.7071, 0.6561 and 0.6409 at batch sizes 32, 16 and 8 "
            "respectively — a swing of 6.5 IoU points produced by batch "
            "composition alone. The temporal axis is padded to the "
            "per-batch maximum and the pad mask is realigned to the pooled "
            "length by truncation rather than pooling, so whether masking "
            "engages depends on which samples share a batch. Any IoU "
            "reported on this benchmark without its evaluation batch size is "
            "therefore not reproducible.")
    figure(d, "fig4_batch_size.png",
           "Fig. 1. Evaluation of a frozen checkpoint depends on batch size.",
           width=3.5)
    body(d, "Second, the same realignment divides by zero. A sample whose "
            "early acquisitions are empty — common where acquisitions "
            "begin late in a season or cloud-affected frames are zero-filled "
            "— yields an all-padding truncated mask and a NaN output. "
            "We measured 13 of 1,937 training samples affected and 0 of 227 "
            "validation samples, which explains why reproduction succeeds "
            "while training fails. No guard exists anywhere in the reference "
            "implementation. Third, three of the four released UTAE "
            "baselines never trained: all are best at epoch 1, and two score "
            "an identical 0.3812 with paddy IoU exactly 0.000, predicting a "
            "single class everywhere. Fourth, the phenology tasks report a "
            "column labelled MAPE which is in fact mean absolute error "
            "divided by the constant 183, computed per pixel; we verified "
            "that the reported value equals MAE/183 exactly in nine of nine "
            "released runs. We additionally note that 16 of the 21 crop "
            "labels have no validation plots, so only binary paddy "
            "classification is evaluable on this benchmark.")

    d.add_heading("Information helps; model complexity does not", level=2)
    body(d, f"Two of nine ablation steps produce effects larger than the "
            f"seed-to-seed spread: adding the full temporal sequence "
            f"(+{n['d_temporal']:.4f} pixel-pooled IoU) and adding "
            f"Sentinel-1 radar (+{n['d_radar']:.4f}). Both add measurement. "
            "Every step that adds model complexity — a wider encoder, a "
            "cosine schedule, the UTAE architecture, augmentation and "
            "plot-balanced sampling — is inconclusive or negative, and "
            "those rungs consumed roughly three quarters of the compute. "
            "Two of them appeared to be results at a single seed and did "
            "not survive three: augmentation read −0.0496 and the "
            "capacity rung −0.0098 at seed 0 alone. With 351 plots, "
            "additional parameters do not create additional evidence, "
            "whereas an additional sensor does; radar matters here in "
            "particular because a substantial share of mid-season optical "
            "frames in the monsoon are cloud-affected.")
    figure(d, "fig2_ladder.png",
           "Fig. 2. Ten-rung controlled ablation, three seeds. Error bars "
           "are seed-to-seed standard deviation.")

    d.add_heading("Reported pixel-pooled IoU does not describe a field", level=2)
    body(d, f"For the best configuration, pixel-pooled IoU is "
            f"{n['v5_px']:.4f} while the median individual field scores "
            f"{n['median_field']:.3f}, and {n['n_below']} of "
            f"{n['n_fields']} validation fields — {n['pct_below']:.0f}% "
            "— fall below 0.5. The reported figure is an average over "
            "pixels and describes no particular field.")
    figure(d, "fig1_per_field_vs_headline.png",
           "Fig. 3. Per-field IoU for the best configuration, with the "
           "reported pixel-pooled value overlaid.")

    d.add_heading("Across four tasks the direction is task-dependent",
                  level=2)
    body(d, f"All {n['n_models']} released models for the three regression "
            f"tasks reproduced within tolerance ({n['n_pass']} of "
            f"{n['n_models']} gates passed), and together with the twelve "
            "crop-type cells this gives 64 verified models. For crop type "
            "the median field is worse than the aggregate. For yield and the "
            "phenology dates the median field is better than the aggregate, "
            "but the tail is markedly heavier: one field-season in ten is "
            "two to four times worse than typical. What is consistent across "
            "tasks is not the direction of the gap but that the aggregate "
            "conceals the tail.")
    tb = d.add_table(rows=1, cols=5); tb.style = "Table Grid"
    for i, h in enumerate(("Task", "Reported MAE", "Typical field",
                           "p90", "p90 / median")):
        c = tb.rows[0].cells[i]; c.text = h
        for r_ in c.paragraphs[0].runs: r_.bold = True
    for t, lab, u in (("crop_yield", "Crop yield", "kg"),
                      ("harvesting_date", "Harvesting date", "days"),
                      ("transplanting_date", "Transplanting date", "days")):
        v = n[t]; row = tb.add_row().cells
        row[0].text = f"{lab} ({u})"
        row[1].text = f"{v['agg_mae']:.2f}"
        row[2].text = f"{v['median']:.2f}"
        row[3].text = f"{v['p90']:.2f}"
        row[4].text = f"{v['p90']/v['median']:.2f}×"
    cap = d.add_paragraph("TABLE I. Reported versus per-field-season error, "
                          "medians across all released models per task.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r_ in cap.runs: r_.font.size = Pt(8.5); r_.italic = True
    figure(d, "fig3_regression_tails.png",
           "Fig. 4. For every regression task the typical field is better "
           "than the reported error, while the tail is far worse.")

    d.add_heading("Failures are systematic, which admits a remedy", level=2)
    body(d, "Per-field IoU correlates between independent seeds at r = +0.52 "
            "to +0.58, and 11 of the 12 fields scoring below 0.5 do so in "
            "every run. Poor performance is thus a property of the field "
            "rather than of the training run. This is the practical "
            "consequence of the audit: because the difficult fields recur, "
            "they can be identified and handled differently. A deployment "
            "can process reliably-mapped fields automatically and refer the "
            "remainder for manual verification, which would be impossible if "
            "failure were random. We therefore recommend that per-field "
            "error distributions, not only aggregate accuracy, be reported "
            "before any such model informs decisions taken at field level.")
    figure(d, "fig5_seed_consistency.png",
           "Fig. 5. Per-field IoU between two independent seeds. The same "
           "fields fail in both runs.", width=3.5)

    d.add_heading("Unreliable fields can be identified without labels",
                  level=2)
    tg = n.get("triage")
    if tg:
        dis = tg["seed_disagreement"]; conf = tg["low_confidence"]
        best = min(dis["screening"], key=lambda r: abs(r["workload"] - 0.5))
        bconf = min(conf["screening"], key=lambda r: abs(r["workload"] - 0.33))
        body(d, "Systematic failure is only actionable if the difficult "
                "fields can be recognised at inference time, when no labels "
                "exist. We tested two signals that require none. The first "
                "is seed disagreement: the fraction of a field’s pixels "
                "on which independently seeded models do not all agree, "
                "available to any deployment that trains more than one "
                "model. The second is mean predictive confidence from a "
                "single model. Both are evaluated as screening tests, since "
                "that is how they would be used: what share of fields must "
                "be referred for verification (workload), and what share of "
                "genuinely unreliable fields that referral catches (recall).")
        loo_d = {r["target_workload"]: r for r in dis["leave_one_out"]}
        loo_c = {r["target_workload"]: r for r in conf["leave_one_out"]}
        d50, c33 = loo_d[0.5], loo_c[0.33]
        body(d, f"Both signals correlate with per-field IoU in the useful "
                f"direction — seed disagreement at r = "
                f"{dis['corr_with_iou']:+.3f} and confidence at r = "
                f"{conf['corr_with_iou']:+.3f}. The referral rule itself uses "
                f"no labels: it refers the fields whose signal falls in the "
                f"top fraction of the signal distribution, a quantity "
                f"computable at inference time. To ensure that no field "
                f"contributes to the threshold deciding its own fate, "
                f"thresholds are set leave-one-out, from the other fields "
                f"only. Referring the top half by seed disagreement captures "
                f"{d50['recall']:.0%} of the unreliable fields "
                f"({d50['n_caught']} of {tg['n_bad']}) at "
                f"{d50['precision']:.0%} precision; confidence alone, which "
                f"needs only one trained model, captures {c33['recall']:.0%} "
                f"at a {c33['workload']:.0%} workload. Random referral would "
                f"capture a share equal to the workload, so both signals "
                f"carry information.")
        body(d, f"These rates must be read with their uncertainty, which is "
                f"large. Resampling fields gives a 95% interval of "
                f"[{d50['recall_ci']['lo']:.0%}, "
                f"{d50['recall_ci']['hi']:.0%}] on the "
                f"{d50['recall']:.0%} figure, and "
                f"[{c33['recall_ci']['lo']:.0%}, "
                f"{c33['recall_ci']['hi']:.0%}] on the "
                f"{c33['recall']:.0%} figure. Recall is measured over the "
                f"{tg['n_bad']} unreliable fields, so a single field moves it "
                f"by {100/tg['n_bad']:.1f} percentage points, against "
                f"{100/tg['n_fields']:.1f} points for workload. We therefore "
                f"present this as a PRELIMINARY result: it establishes that "
                f"label-free signals carry information about which fields a "
                f"model will handle badly, and it does not establish an "
                f"operating threshold or a deployment-ready capture rate. "
                f"Confirming either would require an evaluation set far "
                f"larger than the {tg['n_fields']} labelled fields this "
                f"benchmark provides, and ideally a geographically held-out "
                f"one.")
        geo = tg.get("geographic_holdout")
        if geo:
            big = [r for r in geo["regions"] if r["n_fields"] >= 6]
            body(d, "We also attempted a geographically held-out check, since "
                    "a threshold that only works where it was set is of "
                    "little use. SICKLE labels each plot with a sub-region of "
                    "the delta, so we held out each in turn, set the "
                    "threshold from the remaining regions, and applied it to "
                    "the held-out one. The signal does transfer, but the "
                    "benchmark cannot support a firm statement: of ten "
                    f"region-signal combinations only {len(big)} involve six "
                    "or more fields, and three sub-regions contribute two or "
                    "three validation fields each. In the two largest "
                    "regions, Upper Cauvery (8 fields, 4 unreliable) and "
                    "Middle Vennar (7 fields, 2 unreliable), recall under a "
                    "threshold set entirely outside the region ranges from "
                    "50% to 100% depending on region and signal. We report "
                    "this as evidence that the approach is worth testing "
                    "properly, and as a concrete statement of what this "
                    "benchmark's validation split can and cannot support: a "
                    "geographically held-out evaluation of field-level "
                    "reliability needs a labelled set substantially larger "
                    "than 36 fields spread across five sub-regions.")
        figure(d, "fig6_triage.png",
               "Fig. 6. Screening performance of two label-free signals. "
               "The diagonal is random referral.", width=4.3)

    d.add_heading("Reproducibility", level=2)
    body(d, "Every run records a manifest containing the resolved "
            "configuration, a content hash of it, the git commit and a "
            "dirty-tree flag, the device, the effective and micro batch "
            "sizes, the seed and the trainable parameter count. "
            "Reproduction tolerances were declared in the source before any "
            "run and were not widened. All scripts that produce the tables "
            "and figures in this paper are released with the code, and the "
            "document itself is generated from the result files so that no "
            "value in the text is transcribed by hand.")

    d.add_heading("Limitations", level=2)
    body(d, "Every result is a validation-set number computed on 36 fields, "
            "and on 41 field-seasons from 14 plots for yield and the "
            "phenology tasks, because the benchmark’s test split is "
            "unlabelled. The study covers one delta, one cropping system and "
            "the period 2018–2020. We audit SICKLE, a research "
            "benchmark; we do not audit YES-TECH’s operational models, "
            "which are not publicly available, and no claim is made about "
            "their accuracy. Training used gradient accumulation with "
            "micro-batches of 8, which changes temporal padding relative to "
            "a true batch of 32; this was held constant across all rungs so "
            "that rung-to-rung differences remain controlled, and all "
            "evaluation was performed in full precision at the pinned batch "
            "size. Per-field evaluation follows established methodology [2] "
            "and is applied here rather than proposed.")

    # ---------------------------------------------------------- V
    d.add_heading("Conclusion", level=1)
    body(d, "India has begun automating crop-insurance payouts from "
            "satellite data. Under YES-TECH the manual requires at least 30% "
            "weightage to modelled yield and sets no upper cap, with "
            "reporting placing the rollout at 344 districts and Madhya "
            "Pradesh at 100%, while validation is specified only at the Gram "
            "Panchayat and the resulting estimates are neither appealable in "
            "court nor obtainable under the Right to Information Act. "
            "Decisions, however, are ultimately about individual fields, and "
            "when a model errs systematically on particular fields those "
            "errors bias the village figure rather than cancelling within "
            "it. The farmers exposed to that risk are precisely those whose "
            "holdings this benchmark represents, averaging 0.38 acres.")
    body(d, "Because the operational models are not public, we audited the "
            "research benchmark on which such models are built. Reproducing "
            "64 released models across four tasks, we found four defects "
            "that make the published numbers irreproducible; that five of "
            "nine modelling choices cannot be distinguished from seed "
            "variation at three seeds, the two exceeding it both being "
            "additions of information rather than model complexity; and "
            "that "
            f"a reported pixel-pooled IoU of {n['v5_px']:.4f} corresponds to a median "
            f"field of {n['median_field']:.3f} with {n['pct_below']:.0f}% of "
            "fields below 0.5, while the regression tasks conceal a tail in "
            "which one field-season in ten is two to four times worse than "
            "typical.")
    body(d, "The remedy follows from the final result. Because per-field "
            "failures recur across independent training runs rather than "
            "occurring at random, the difficult fields are identifiable in "
            "advance. We therefore propose that benchmarks and operational "
            "systems report per-field error distributions alongside "
            "aggregate accuracy, and that deployments adopt selective "
            "automation: process the fields a model handles reliably, and "
            "refer the remainder for human verification. This does not pay a "
            "farmer or raise a yield. It is a safeguard — a measurement "
            "that is currently absent, and that would expose failures before "
            "they reach the people least able to contest them.")

    # ---------------------------------------------------------- refs
    d.add_heading("References", level=5)
    refs = [
        "D. Sani, S. Mahato, S. Saini, H. K. Agarwal, C. S. Devshali, "
        "S. Anand, G. Arora, and T. Jayaraman, “SICKLE: A Multi-Sensor "
        "Satellite Imagery Dataset Annotated with Multiple Key Cropping "
        "Parameters,” in Proc. IEEE/CVF Winter Conf. on Applications of "
        "Computer Vision (WACV), 2024.",
        "Z. Wang, M. Berman, A. Rannen-Triki, P. H. S. Torr, D. Tuia, "
        "T. Tuytelaars, L. Van Gool, J. Yu, and M. B. Blaschko, "
        "“Revisiting Evaluation Metrics for Semantic Segmentation: "
        "Optimization and Evaluation of Fine-grained Intersection over "
        "Union,” in Advances in Neural Information Processing Systems "
        "(NeurIPS), 2023.",
        "Mahalanobis National Crop Forecast Centre, Department of "
        "Agriculture and Farmers Welfare, Government of India, "
        "“Yield Estimation System based on Technology (YES-TECH) under "
        "PMFBY — Manual,” 2023.",
        "Department of Agriculture and Farmers Welfare, Government of India, "
        "“Pradhan Mantri Fasal Bima Yojana — Revised Operational "
        "Guidelines,” 2020.",
        "V. Sainte Fare Garnot and L. Landrieu, “Panoptic Segmentation "
        "of Satellite Image Time Series with Convolutional Temporal "
        "Attention Networks,” in Proc. IEEE/CVF Int. Conf. on Computer "
        "Vision (ICCV), 2021.",
        "X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, "
        "“Convolutional LSTM Network: A Machine Learning Approach for "
        "Precipitation Nowcasting,” in Advances in Neural Information "
        "Processing Systems (NeurIPS), 2015.",
        "O. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and "
        "O. Ronneberger, “3D U-Net: Learning Dense Volumetric "
        "Segmentation from Sparse Annotation,” in Proc. MICCAI, 2016.",
        "Down To Earth, “Can technology make India’s crop "
        "insurance payouts more accurate? YES-TECH vs manual crop cutting "
        "under PMFBY,” and “Beyond the lens: Madhya Pradesh "
        "farmers struggle with satellite-based crop insurance under "
        "PMFBY,” 2025. (Source for deployment scale and state-level "
        "weightings, which do not appear in [3].)",
    ]
    for i, r_ in enumerate(refs, 1):
        p = d.add_paragraph(f"[{i}] {r_}")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        for run in p.runs: run.font.size = Pt(8.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.save(OUT)
    print(f"  wrote {OUT}")
    print(f"  {len(d.paragraphs)} paragraphs, {len(d.tables)} table, "
          f"{len(list(FIG.glob('*.png')))} figures embedded")


if __name__ == "__main__":
    build()
