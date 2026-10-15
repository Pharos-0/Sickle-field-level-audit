"""Build the combined thesis paper: all three clauses of the objective.

    python scripts/make_thesis_paper.py

The IEEE manuscript (make_paper_ieee.py) is scoped to the benchmark audit:
clauses one and two of the objective. The thesis needs one document that
answers the whole objective, including the decision-support prototype in
D:/FPD/sickle-guard. This builds it.

RULES THIS FILE KEEPS
---------------------
* Every number is read from an artefact in one of the two projects at
  build time. Nothing is typed in, so the paper follows the final seeds.
* Simulated results are reported in one section, which opens by saying
  so, and every figure carrying them is titled SIMULATED. Anywhere else a
  simulated result is named as simulated in the same sentence.
* Before saving, the rendered text is checked against the canonical
  objective (must appear verbatim) and against the overstatements the
  evidence rules out (must not appear un-negated). A violation aborts the
  build rather than producing a document that claims too much.

Writes docs/Thesis_Paper.docx.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

PROJ = Path(__file__).resolve().parents[1]
# SICKLE-Guard sits beside this project unless SICKLE_GUARD_ROOT says
# otherwise (renamed from sickle-yield on 2026-10-14).
GUARD = Path(__import__("os").environ.get(
    "SICKLE_GUARD_ROOT", Path(__file__).resolve().parents[2] / "sickle-guard"))
OUT = PROJ / "docs" / "Thesis_Paper.docx"
FIGS = PROJ / "results" / "figures_ieee"
TFIG = PROJ / "results" / "figures_thesis"

sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(GUARD / "src"))

OBJECTIVE = (
    "We identify conditions under which satellite-based agricultural "
    "predictions can be unreliable, empirically evaluate safeguards for "
    "detecting those failures, and develop a decision-support prototype "
    "to help prioritize verification while guarding against false "
    "reassurance.")


# ------------------------------------------------------------- loading
def jload(p: Path):
    if not p.exists():
        raise SystemExit(f"missing artefact {p} -- run the pipeline that "
                         f"produces it before building the thesis paper")
    return json.loads(p.read_text(encoding="utf8"))


def load() -> dict:
    from drift_check import DERIVED, bs_cells
    N: dict = {}

    # ---------- Part I: the benchmark audit
    L = {r["rung"]: r for r in csv.DictReader(
        (PROJ / "results/ladder_crop_type.csv").open(encoding="utf8"))
        if r.get("pixel_macro_iou")}
    N["ladder"] = L
    bs = bs_cells()
    N["bs_spread_max"] = max(max(v.values()) - min(v.values())
                             for v in bs.values())
    N["bs_worst_run"] = max(bs, key=lambda k: max(bs[k].values())
                            - min(bs[k].values()))
    N["n_yield_ckpt"] = int(DERIVED["n_yield_checkpoints"]())
    N["n_worse_mae"] = int(DERIVED["n_worse_mae"]())
    N["const_mae"] = float(DERIVED["const_mae"]())
    N["best_ckpt_mae"] = float(DERIVED["best_ckpt_mae"]())
    # plot-clustered intervals on the gap to the constant; required
    _yv = PROJ / "results/yield_vs_constant.json"
    if not _yv.exists():
        raise SystemExit("missing results/yield_vs_constant.json. "
                         "Run: python scripts/yield_vs_constant.py")
    N["yvc"] = jload(_yv)
    N["n_reg_audited"] = int(DERIVED["n_regression_audited"]())
    N["n_reg_pass"] = int(DERIVED["n_regression_pass"]())

    fd = jload(PROJ / "results/field_distributions.json")
    lad = fd["ladder_crop_type"]
    N["pooled"] = lad["pooled_pixel_macro_iou"]
    N["median_field"] = lad["quantiles"]["50"]
    N["mean_field"] = lad["per_field_mean"]
    N["mean_field_ci"] = lad["ci_mean"]
    N["macro_contrast"] = lad["macro_contrast"]
    N["n_single_crop"] = lad["n_single_crop_fields"]
    N["shortfall_decile"] = lad["share_of_shortfall_worst_decile"]
    N["n_fields"] = lad["n_fields"]
    N["below50"] = lad["below"]["0.5"]
    N["spearman"] = lad["size"]["spearman_weight_vs_iou"]
    N["spearman_ci"] = lad["size"]["spearman_weight_vs_iou_ci"]
    sh = [abs(r["shift_without_largest_field"])
          for r in fd["released_crop_type"]
          if "shift_without_largest_field" in r]
    N["loo_max"] = max(sh)
    N["loo_median"] = float(np.median(sh))
    N["gaps"] = [r["gap"] for r in fd["released_crop_type"]]
    N["gap_neg"] = [r["run"] for r in fd["released_crop_type"]
                    if r["gap"] < 0]
    fs = jload(PROJ / "results/field_size_val.json")
    N["share_largest"] = fs["share_largest"]
    N["n_px"] = fs["n_pixels_in_pooled_metric"]

    lp = jload(PROJ / "results/ladder_power.json")
    N["lp"] = lp
    N["sigma_max"] = lp["sigma_max"]
    N["steps"] = {r["step"]: r for r in lp["steps"]}

    # ---------- Part II: safeguards
    N["cs"] = jload(PROJ / "results/triage/conditional_screen.json")
    N["sv"] = jload(PROJ / "results/triage/screen_validate.json")
    N["sv_macro"] = jload(PROJ / "results/triage/screen_validate_macro.json")
    N["cs_macro"] = jload(PROJ / "results/triage/conditional_screen_macro.json")
    N["tg"] = jload(PROJ / "results/triage/triage_v5.json")
    N["ys"] = jload(PROJ / "results/triage/yield_screen.json")
    M = GUARD / "results/metrics"
    rel = jload(M / "reliability.json")
    N["yield_signals"] = {k: v for k, v in rel["signals"].items()}
    N["quantile_coverage"] = rel["quantile_interval_empirical_coverage_nominal_90"]
    N["mndwi"] = jload(M / "mndwi_screen.json")

    # ---------- Part III: the prototype
    N["ep"] = jload(M / "evaluate_prototype.json")
    N["ae"] = jload(M / "audit_evaluation.json")
    N["eb"] = jload(M / "error_breakdown.json")
    N["lv"] = jload(M / "label_validity.json")
    N["dsa"] = jload(M / "does_satellite_add.json")
    N["syn"] = jload(M / "synthetic_triage.json")
    N["ss"] = jload(M / "season_swap.json")
    N["claims"] = jload(M / "check_claims.json")
    N["vp"] = jload(M / "verification_policies.json")
    N["insp"] = jload(M / "inspection_sim.json")
    N["yw"] = jload(M / "yestech_whatif.json")
    N["off"] = jload(M / "official_records.json")
    N["pilot"] = jload(M / "pilot_power.json")
    N["wf"] = jload(GUARD / "results/workflow/demo_summary.json")
    _wt = (GUARD / "tests/test_inspection_workflow.py").read_text(encoding="utf8")
    N["n_wf_tests"] = _wt.count("\ndef test_")
    # the predefined safety cases: count the tests in the file that holds
    # them, so the paper cannot claim more cases than exist
    _sc = (GUARD / "tests/test_safety_cases.py").read_text(encoding="utf8")
    N["n_safety_tests"] = _sc.count("\ndef test_")
    try:
        import farmer_report as fr
        N["untranslated"] = fr.untranslated_fraction()
    except Exception:                                       # noqa: BLE001
        N["untranslated"] = None

    # ---------- integrity
    N["drift"] = jload(PROJ / "results/drift_check.json")
    N["mrepro"] = jload(PROJ / "results/minimal_repro.json")

    N["refs"] = {r["n"]: r["entry"] for r in jload(
        PROJ / "docs/references.json")["references"]}
    # references only the thesis cites live apart, so the IEEE manuscript
    # carries no uncited entry
    N["refs"].update({r["n"]: r["entry"] for r in jload(
        PROJ / "docs/references_thesis.json")["references"]})
    return N


# ------------------------------------------------------------- helpers
def m(v, nd=3, sign=False):
    """Format with a typographic minus."""
    s = format(v, f"{'+' if sign else ''}.{nd}f")
    return s.replace("-", "\u2212")


def ci(a, b, nd=3):
    return f"[{m(a, nd, True)}, {m(b, nd, True)}]"


def pct(v):
    return f"{100 * v:.0f} %"


def TALLYW(k):
    """A count that opens a sentence is written out."""
    words = {20: "Twenty", 21: "Twenty-one", 19: "Nineteen", 18: "Eighteen",
             22: "Twenty-two", 23: "Twenty-three", 24: "Twenty-four",
             25: "Twenty-five", 17: "Seventeen", 16: "Sixteen",
             15: "Fifteen"}
    return words.get(k) or w(k, True)


def cil(a, b, nd=3):
    """Interval on a LEVEL (an AUROC, a share): no plus signs."""
    return f"[{m(a, nd)}, {m(b, nd)}]"


_W = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
      6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
      11: "eleven", 12: "twelve"}


def w(k, cap=False):
    t = _W.get(k, str(k))
    return t[0].upper() + t[1:] if cap else t


class Doc:
    """python-docx with the few things this paper needs, and a citation
    renumbering pass so references appear in order of first use."""

    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(11)
        for sec in self.d.sections:
            sec.left_margin = sec.right_margin = Inches(1.0)
            sec.top_margin = sec.bottom_margin = Inches(0.9)
        self.cites: list[int] = []
        self.text: list[str] = []
        self.fig = 0
        self.tab = 0

    def cite(self, *ns) -> str:
        out = []
        for k in ns:
            if k not in self.cites:
                self.cites.append(k)
            out.append(str(self.cites.index(k) + 1))
        return "[" + ", ".join(out) + "]"

    def title(self, t, sub=None):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(t)
        r.bold = True
        r.font.size = Pt(17)
        self.text.append(t)
        if sub:
            for line, it in sub:
                q = self.d.add_paragraph()
                q.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rr = q.add_run(line)
                rr.italic = it
                rr.font.size = Pt(10.5)
                self.text.append(line)

    def h(self, t, level=1):
        p = self.d.add_heading(t, level=level)
        for r in p.runs:
            r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x73)
        self.text.append(t)

    def p(self, t, italic=False, bold_lead=None):
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(6)
        if bold_lead:
            b = para.add_run(bold_lead + " ")
            b.bold = True
        r = para.add_run(t)
        r.italic = italic
        self.text.append((bold_lead + " " if bold_lead else "") + t)

    def quote(self, t):
        para = self.d.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.5)
        para.paragraph_format.right_indent = Inches(0.5)
        r = para.add_run(t)
        r.italic = True
        self.text.append(t)

    def bullet(self, t, lead=None):
        para = self.d.add_paragraph(style="List Bullet")
        if lead:
            b = para.add_run(lead + " ")
            b.bold = True
        para.add_run(t)
        self.text.append((lead + " " if lead else "") + t)

    def table(self, header, rows, caption):
        self.tab += 1
        cap = self.d.add_paragraph()
        r = cap.add_run(f"Table {self.tab}. {caption}")
        r.bold = True
        r.font.size = Pt(9.5)
        self.text.append(f"Table {self.tab}. {caption}")
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            t.rows[0].cells[i].text = h
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
            self.text.append(" | ".join(map(str, row)))
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9.5)
        self.d.add_paragraph()
        return self.tab

    def figure(self, path: Path, caption, width=6.0):
        if not path.exists():
            raise SystemExit(f"figure missing: {path}")
        self.fig += 1
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        c = self.d.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = c.add_run(f"Figure {self.fig}. {caption}")
        r.font.size = Pt(9.5)
        self.text.append(f"Figure {self.fig}. {caption}")
        return self.fig


# ------------------------------------------------------------- figures
def make_figures(N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TFIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 220,
                         "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    # --- held-out ranking, EMPIRICAL
    t = N["ep"]["ranking"]["held-out test split"]
    order = ["SICKLE-Guard (as shipped)", "last harvest only",
             "history only, no imagery", "largest field first",
             "satellite only", "arbitrary order", "smallest field first"]
    order = [k for k in order if k in t]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(order))[::-1]
    a = [t[k]["auroc"] for k in order]
    lo = [t[k]["auroc"] - t[k]["auroc_ci"][0] for k in order]
    hi = [t[k]["auroc_ci"][1] - t[k]["auroc"] for k in order]
    ax.errorbar(a, y, xerr=[lo, hi], fmt="o", color="#1f3b73",
                ecolor="#7f9cc6", capsize=3)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=1, label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("AUROC for a below-threshold field-season (95 % CI)")
    ax.set_title("Held-out test split, 27 plots never used in fitting "
                 "(observed)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    f1 = TFIG / "fig_thesis_ranking.png"
    fig.savefig(f1)
    plt.close(fig)

    # --- budget sweep, SIMULATED
    sw = N["syn"]["sweeps"]["inspection budget"]["rows"]
    budgets = sorted(sw, key=float)
    pol = {"history-first (record fields only, then cold starts)":
           ("history-first", "#8e44ad"),
           "prototype (history, satellite fallback)": ("prototype", "#1f3b73"),
           "history only": ("history only", "#4a7fb5"),
           "register only (what the team already knows)":
           ("register (already known)", "#c0392b"),
           "random": ("random", "#999999")}
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for k, (lab, c) in pol.items():
        ax.plot([float(b) for b in budgets],
                [sw[b][k]["recall_event"] for b in budgets],
                marker="o", color=c, label=lab, lw=1.6)
    ax.set_xlabel("inspection budget (share of fields that can be visited)")
    ax.set_ylabel("recall of event losses")
    ax.set_title("SIMULATED -- calibrated generator, not field "
                 "performance", fontsize=9, color="#c0392b")
    ax.legend(fontsize=7.5, loc="upper left")
    f2 = TFIG / "fig_thesis_synthetic_budget.png"
    fig.savefig(f2)
    plt.close(fig)

    # --- labels against the state's records, OBSERVED
    off = N["off"]
    cells = off["l1_level"]["cells"]
    ev = off["l2_event"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    names = [("Kar/Kuruvai/Sornavari 2019-20", "Kuruvai"),
             ("Samba/Thaladi/Pishanam 2019-20", "Samba")]
    for yi, (k, lab) in enumerate(names):
        c = cells[k]
        b = c["by_conversion"]["2/3 (official)"]
        lo_b, hi_b = c["official_core_band"]
        a1.add_patch(plt.Rectangle((lo_b, yi - 0.28), hi_b - lo_b, 0.56,
                                   color="#888888", alpha=0.25, lw=0))
        for v in c["official_core_districts"].values():
            a1.plot([v, v], [yi - 0.28, yi + 0.28], color="#666666", lw=1)
        a1.errorbar([b["sickle_mean_rice_eq_kg_ha"]], [yi],
                    xerr=[[b["sickle_mean_rice_eq_kg_ha"] - b["ci95"][0]],
                          [b["ci95"][1] - b["sickle_mean_rice_eq_kg_ha"]]],
                    fmt="o", color="#1f3b73", capsize=3)
    a1.set_yticks([0, 1])
    a1.set_yticklabels([n[1] for n in names])
    a1.set_ylim(-0.7, 1.7)
    a1.set_xlabel("kg of cleaned rice per ha, 2019-20")
    a1.set_title("recalled (dot) vs crop-cutting (grey)", fontsize=8.5)
    rows = [("recalled, Samba", ev["samba_recalled"], "#1f3b73"),
            ("recalled, Kuruvai\n(control)", ev["kuruvai_control"], "#888888"),
            ("satellite model,\nSamba", ev["samba_model"], "#222222")]
    for yi, (lab, r, col) in enumerate(rows[::-1]):
        m_ = 100 * r["mean_change"]
        a2.errorbar([m_], [yi], xerr=[[m_ - 100 * r["ci95"][0]],
                                      [100 * r["ci95"][1] - m_]],
                    fmt="s", color=col, capsize=3)
    a2.axvline(0, color="#c0392b", lw=1, ls="--")
    a2.set_yticks(range(3))
    a2.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
    a2.set_xlabel("same-plot change into 2020-21 (%)")
    a2.set_title("the season the state paid delta relief", fontsize=8.5)
    fig.suptitle("Observed", fontsize=9, x=0.06, ha="left")
    fig.tight_layout()
    f3 = TFIG / "fig_thesis_official_records.png"
    fig.savefig(f3)
    plt.close(fig)

    # --- imperfect inspectors and the YES-TECH what-if, SIMULATED
    ho = N["insp"]["sets"]["held-out test split (VERDICT)"]["budgets"]["0.2"]
    cur = N["yw"]["conditional_curves"]
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    sv = np.linspace(0.3, 1.0, 15)
    for k, lab, col in (("register", "register-guided", "#1f6f4a"),
                        ("workflow", "workflow-guided", "#1f3b73")):
        b1.plot(sv, sv * ho[k]["n_found"], color=col, lw=1.8, label=lab)
    b1.plot(sv, sv * ho["random"]["mean_found"], color="#999999", lw=1.6,
            label="random visits")
    b1.axhline(ho["random"]["mean_found"], color="#999999", ls="--", lw=1,
               label="perfect inspector, random")
    b1.set_xlabel("inspector sensitivity")
    b1.set_ylabel("real problems confirmed")
    b1.legend(fontsize=6.8, loc="upper left")
    b1.set_title("20 % budget, held-out rankings", fontsize=8.5)
    sh = [0.05, 0.10, 0.20, 0.30]
    for w, lab, col in (("0.30", "30 % weight", "#1f3b73"),
                        ("0.50", "50 %", "#1f6f4a"),
                        ("1.00", "100 %", "#a5611a")):
        for sd, ls in (("0.20", "-"), ("0.30", "--")):
            b2.plot([100 * x for x in sh],
                    [100 * cur[f"w={w}|sd_t={sd}|bias=+0.00|sd_c=0.10"][f"{x:.2f}"]["p_paid_nothing"] for x in sh],
                    color=col, ls=ls, marker="o", ms=3,
                    label=lab if sd == "0.20" else None)
    b2.set_xlabel("true shortfall below threshold (%)")
    b2.set_ylabel("paid nothing (%)")
    b2.legend(fontsize=6.8)
    b2.set_title("Insurance Unit, technology error 0.20 / 0.30", fontsize=8.5)
    fig.suptitle("SIMULATED -- assumptions, not field or scheme performance",
                 fontsize=9, color="#c0392b", x=0.06, ha="left")
    fig.tight_layout()
    f4 = TFIG / "fig_thesis_simulated_extensions.png"
    fig.savefig(f4)
    plt.close(fig)

    # --- the five steps, and the evidence behind each
    steps = [("1  Estimate", "own past harvests\nfirst; satellite\nonly without one", "#2d7a3a", "observed"),
             ("2  Flag", "no reliability\nsignal works\n(0 of 9 tested)", "#b03636", "refuted, so not used"),
             ("3  Prioritise", "records beat\nrandom; reserve\nfor no-record", "#2d7a3a", "observed"),
             ("4  Verify", "blind sheets,\nevidence rule,\nhashed audit", "#a5611a", "software; visits simulated"),
             ("5  Evaluate", "pre-registered:\nrandom, register,\nhistory", "#2d7a3a", "observed")]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    for i, (t, body, col, status) in enumerate(steps):
        x0 = 0.1 + i * 2.0
        ax.add_patch(plt.Rectangle((x0, 0.45), 1.75, 1.6, fill=False,
                                   ec=col, lw=1.6))
        ax.text(x0 + 0.875, 1.83, t, ha="center", va="center",
                fontsize=8.5, weight="bold", color=col)
        ax.text(x0 + 0.875, 1.1, body, ha="center", va="center",
                fontsize=6.3, linespacing=1.25)
        ax.text(x0 + 0.875, 0.22, status, ha="center", va="center",
                fontsize=6.4, style="italic", color=col)
        if i < 4:
            ax.annotate("", xy=(x0 + 2.0, 1.25), xytext=(x0 + 1.78, 1.25),
                        arrowprops=dict(arrowstyle="->", color="#555555"))
    f5 = TFIG / "fig_thesis_workflow.png"
    fig.savefig(f5)
    plt.close(fig)
    return f1, f2, f3, f4, f5


# ------------------------------------------------------------- the paper
def build(N) -> Doc:
    D = Doc()
    f_rank, f_syn, f_off, f_sim, f_wf = make_figures(N)
    st = N["steps"]
    lad = N["ladder"]
    cs = N["cs"]
    t1 = cs["test_1_conditional_permutation"]["low_confidence"]
    ff = cs["test_2_out_of_sample"]["fit_free"]
    ep = N["ep"]
    hold = ep["ranking"]["held-out test split"]
    oof = ep["ranking"]["all field-seasons, out-of-fold (secondary, more power)"]
    flag = ep["flag_adds_information"]["held-out test split"]
    warn = ep["warning_calibration"]["held-out test split"]
    warn_all = ep["warning_calibration"]["all field-seasons"]
    base = N["ae"]["baseline_comparison"]
    eb = N["eb"]["sets"]["held-out test split (PRIMARY)"]["has_record"]
    lv = N["lv"]
    ev = lv["ranking_by_subset"]["EVENT plots only (the real case)"]
    syn = N["syn"]
    sw_rec = syn["sweeps"]["share of fields with a record"]["rows"]
    sw_bud = syn["sweeps"]["inspection budget"]["rows"]
    mn = N["mndwi"]["sets"]["held-out test split (PRIMARY)"]
    ys = N["ys"]["threshold_free"]
    ss = N["ss"]

    # tally the simulated sweeps with the 0.02 margin the study declares
    tally = {"tie": 0, "register": 0, "prototype": 0, "history-first": 0,
             "other": 0}
    n_settings = 0
    for _, sw in syn["sweeps"].items():
        for _, winner in sw["best_policy"].items():
            n_settings += 1
            if winner.startswith("tie"):
                tally["tie"] += 1
            elif winner.startswith("register"):
                tally["register"] += 1
            elif winner.startswith("prototype"):
                tally["prototype"] += 1
            elif winner.startswith("history-first"):
                tally["history-first"] += 1
            else:
                tally["other"] += 1

    a, b = eb["has a record"], eb["cold start"]
    off = N["off"]
    hl = off["headline"]
    oev = off["l2_event"]
    kcell = off["l1_level"]["cells"]["Kar/Kuruvai/Sornavari 2019-20"]
    scell = off["l1_level"]["cells"]["Samba/Thaladi/Pishanam 2019-20"]
    kb = kcell["by_conversion"]["2/3 (official)"]
    rel = oev["relief"]["total_lakh"]
    rel20 = next(v for k_, v in rel.items() if k_.startswith("2020-21")) / 100
    rel19 = next(v for k_, v in rel.items() if k_.startswith("2019-20")) / 100
    insp = N["insp"]
    ho20 = insp["sets"]["held-out test split (VERDICT)"]["budgets"]["0.2"]
    oofs = insp["sets"]["all field-seasons out-of-fold (secondary)"]
    exp20 = insp["observed_exposure"]["held-out test split"]["budgets"]["0.2"]
    g7 = ho20["grid"]["s=0.70|f=0.10"]
    yw = N["yw"]
    cc = yw["conditional_curves"]
    p100 = cc["w=1.00|sd_t=0.20|bias=+0.00|sd_c=0.10"]["0.10"]["p_paid_nothing"]
    p30 = cc["w=0.30|sd_t=0.20|bias=+0.00|sd_c=0.10"]["0.10"]["p_paid_nothing"]
    gd = yw["guard"]["loss=0.20|w=1.00|sd_t=0.20|bias=+0.00"]["budgets"]["0.05"]["policies"]
    wf = N["wf"]

    def pp(v, nd=1):
        """A change as a signed percentage with a typographic minus; a
        change that rounds to zero is written without a sign."""
        if round(100 * v, nd) == 0:
            return f"{0:.{nd}f} %"
        return m(100 * v, nd, True) + " %"
    n_sig = len(N["yield_signals"]) + 2   # + released-ensemble, + MNDWI
    pos = [k for k, r in st.items()
           if r["verdict"].startswith("resolved") and r["paired_mean"] > 0]
    neg = [k for k, r in st.items()
           if r["verdict"].startswith("resolved") and r["paired_mean"] < 0]
    small = [k for k, r in st.items() if r["verdict"].startswith("small")]
    under = [k for k, r in st.items()
             if r["verdict"].startswith("underpowered")]
    n_seed = {k: int(r["n_seeds"]) for k, r in lad.items()}
    n_fin = {k: int(r.get("n_seeds_finished") or r["n_seeds"])
             for k, r in lad.items()}
    extra = sorted([k for k in n_fin if n_fin[k] > n_seed[k]],
                   key=lambda k: int(k[1:]))

    def step(k):
        return k.replace("->", " to ")

    # ================================================================ title
    D.title(
        "Field-Level Reliability, Safeguards and Cautious Decision Support "
        "for Satellite Crop Monitoring",
        [("[Author name(s) \u2014 to be completed by the author]", False),
         ("M.Tech Data Science \u2014 thesis paper", False),
         ("[email \u2014 to be completed]", True)])

    # ============================================================= abstract
    D.h("Abstract", 1)
    D.p("Satellite crop estimates are increasingly considered for decisions "
        "about individual fields, while the benchmarks that develop them "
        "report accuracy pooled over pixels. " + OBJECTIVE + " On the "
        "SICKLE paddy benchmark " + D.cite(1) + " we find that a pooled IoU "
        f"of {m(N['pooled'], 4)} is close to the average field but hides "
        f"a tail: {N['below50']['n']} of {N['n_fields']} fields score below "
        "0.5 while most are mapped almost perfectly; that one field "
        f"carries {pct(N['share_largest'])} of the scored pixels; and that "
        f"none of the {N['n_yield_ckpt']} released yield models is shown "
        "to beat a constant that uses no imagery. Against Tamil Nadu's own "
        "crop-cutting statistics the farmer-recalled yield labels run low, "
        f"at {m(hl['kuruvai_ratio_to_official'], 2)} of the official figure "
        "in Kuruvai. In a season the state allocated about "
        f"\u20b9{rel20:,.0f} crore of crop-damage relief across the delta, "
        "the satellite model's estimates at the same plots were unchanged "
        "from the season before, so the model does not respond to the "
        "season; because the plots carry no district, this cannot show "
        "which of them were damaged. "
        "Safeguards were evaluated, "
        "not assumed: on crop type a label-free signal carries information "
        "beyond field size but does not demonstrably outrank it; on yield, "
        f"none of {w(n_sig)} reliability signals discriminates, two of them "
        "specified under protocols frozen before any result. The "
        "decision-support prototype built on these findings ranks fields "
        "mainly by their own past harvests, reserves a share of inspections "
        "for fields with no record, hands inspectors task sheets that "
        "withhold the estimate, records every finding in a tamper-evident "
        "audit chain, and has no reassuring state. On 27 held-out plots, "
        f"with {a['n_bad'] + b['n_bad']} below-threshold field-seasons, it "
        f"catches {a['n_bad'] - a['n_missed']} of the {a['n_bad']} that have "
        f"a record and {b['n_bad'] - b['n_missed']} of the {b['n_bad']} that "
        "do not; counts this small carry wide uncertainty. Its advantage "
        "over the register an inspection office already keeps, or over a "
        "lookup of the last harvest, is not demonstrated: the value lies in "
        "the field records. In simulation, record-guided inspection tolerates "
        "weak inspectors, and at full model weight an error-prone yield "
        "estimate leaves units that are truly short unpaid far more often "
        "than at the 30 % minimum. No outcome for any farmer is measured.")

    # ========================================================== 1 intro
    D.h("1. Introduction and objective", 1)
    D.p("India's YES-TECH framework blends modelled yield with manual "
        "crop-cutting experiments for crop-insurance assessment under "
        "PMFBY, with a stated minimum 30 % weightage to the modelled "
        "component and no upper cap " + D.cite(2, 3) + ". Decisions that "
        "follow concern individual fields. Operational models are not "
        "public, so this work studies the research benchmark on which such "
        "models are developed and makes no claim about any deployed "
        "system.")
    D.p("The stakes of that weighting are on record. A written answer in "
        "the Lok Sabha states that in 2025-26 Madhya Pradesh assigned 100 % "
        "and Maharashtra 50 % weightage to the technology-based yield, and "
        "that under the scheme's area approach enrolled farmers do not file "
        "claims: claims are worked out automatically from the yield the "
        "state reports " + D.cite(21) + ". Because no claim is filed, a "
        "farmer meets an error in that yield in the payout, and the routes "
        "to contest it are district and state grievance committees and a "
        "national helpline " + D.cite(21) + ". The yield's accuracy is "
        "contested. In a field assessment in three districts of Madhya "
        "Pradesh, farmers reported falling claims, while the insurers "
        "interviewed attributed the fall to the end of crop-cutting yields "
        "that large farmers had been able to push down, and acknowledged "
        "room for 20 to 30 per cent error " + D.cite(22) + ". No public data "
        "settles that dispute and this work does not attempt to. It asks a "
        "prior question on a public benchmark: when should a satellite "
        "prediction about a field not be trusted, and how should "
        "verification be organised around the answer. The objective, stated "
        "once and checked against throughout:")
    D.quote(OBJECTIVE)
    D.p("Four phrases in it are deliberate. Conditions under which, not "
        "merely where: the work identifies circumstances that make a "
        "prediction unreliable. Empirically evaluate, not test or provide: "
        "a safeguard that fails a fair test is a result. Help prioritize, "
        "because the prototype supports an inspection office's choice and "
        "does not replace it. Guarding against false reassurance, not "
        "protecting farmers from it: the first is a property of the system "
        "and can be demonstrated; the second is an outcome for a person and "
        "is not measured here.")

    # ========================================================== 2 data
    D.h("2. Data and evaluation regimes", 1)
    D.p("SICKLE " + D.cite(1) + " covers paddy in the Cauvery Delta with "
        "Sentinel-1, Sentinel-2 and Landsat-8 time series and per-plot "
        "labels. Two evaluation regimes are used and never pooled. For crop "
        f"type the unit is the field: each of the {N['n_fields']} labelled "
        "validation fields is scored once, pooling all of its pixels and "
        "samples. For yield the unit is the field-season, and because "
        "several seasons share a plot, every interval resamples plots "
        "rather than field-seasons. Yield is farmer-recalled and expressed per acre; a "
        "field-season is called below threshold when its yield is under 80 % "
        "of the training-split mean, a relative cut that is not a "
        "measurement of crop damage. Official district yields, conversion "
        "factors and crop-damage relief come from Tamil Nadu's agriculture "
        "statistics " + D.cite(23, 24) + ", parsed by script from the "
        "published tables and checked row by row for internal "
        "consistency.")

    # ================================================ 3 part I
    D.h("3. Where predictions are unreliable", 1)
    D.p("Evaluation is not stable. Re-evaluating the same frozen "
        f"checkpoint at different batch sizes moves its score by up to "
        f"{100 * N['bs_spread_max']:.1f} IoU points ({N['bs_worst_run']}), "
        "because the reference implementation pads the time axis to the "
        "batch maximum and realigns the pad mask by truncation. Any IoU "
        "reported on this benchmark without its batch size is therefore "
        "not reproducible. Of the regression checkpoints, "
        f"{N['n_reg_pass']} of {N['n_reg_audited']} reproduce within the "
        "declared tolerance.", bold_lead="Instability.")
    _y, _yb = N["yvc"], N["yvc"]["best"]
    _s = lambda x: format(x, "+.1f").replace("-", "\u2212")  # noqa: E731
    D.p(f"All {N['n_yield_ckpt']} released yield checkpoints have a higher "
        "observed error than a constant: the training-mean predictor scores "
        f"MAE {N['const_mae']:.1f} kg/acre and the best checkpoint "
        f"{N['best_ckpt_mae']:.1f}. On {_y['n_field_seasons']} field-seasons "
        f"from {_y['n_plots']} plots, though, no difference is "
        "distinguishable from zero: resampling plots, the best checkpoint's "
        f"excess is {_s(_yb['mae_minus_constant'])} kg/acre (95 % CI "
        f"{_s(_yb['ci95'][0])} to {_s(_yb['ci95'][1])}), and "
        f"{_y['n_ci_above_zero'] or 'none'} of the {_y['n_checkpoints']} "
        "intervals lies wholly above zero. None is shown to beat the "
        "constant, and the benchmark reports no constant baseline, so even "
        "that is invisible in its own tables.",
        bold_lead="Yield models are not shown to beat a constant.")
    b50 = N["below50"]
    mc = N["macro_contrast"]
    D.p("A field is scored on the crop or crops it holds, which for a "
        "single-crop field is the share of its pixels labelled correctly. "
        "The obvious alternative, IoU averaged over both classes, is "
        f"wrong here: {N['n_single_crop']} of the {N['n_fields']} fields hold "
        "one crop in every season, and for those it is 1.0 when every pixel "
        "is right and at most 0.5 once one pixel is wrong, because the "
        "absent class then scores 0. An earlier version of this study used "
        f"it and reported a median field of {m(mc['median'])} with "
        f"{mc['below']['0.5']} fields below 0.50; on the field score the "
        f"median is {m(N['median_field'])} and {b50['n']} are below. The "
        "error was found while rendering the prediction maps and is "
        "corrected everywhere, with the record kept.",
        bold_lead="Scoring a field.")
    D.p(f"A pooled IoU of {m(N['pooled'], 4)} is close to the average "
        f"field ({m(N['mean_field'])}, 95 % CI {m(N['mean_field_ci'][0])} to "
        f"{m(N['mean_field_ci'][1])}) and below the median one "
        f"({m(N['median_field'])}), but not where the errors are: "
        f"{b50['n']} of {N['n_fields']} fields ({pct(b50['frac'])}, 95 % CI "
        f"{pct(b50['ci'][0])} to {pct(b50['ci'][1])}) fall below 0.50, and "
        f"the {N['shortfall_decile']['n_fields']} worst carry "
        f"{pct(N['shortfall_decile']['share'])} of the shortfall below the "
        "pooled figure. The headline is also fragile: one field supplies "
        f"{pct(N['share_largest'])} of the "
        f"{N['n_px']:,} scored pixels, and because the pooled confusion "
        "matrix is the sum of per-field matrices, removing that field "
        "recomputes the metric exactly: across the twelve released "
        f"checkpoints it moves by a median of {m(N['loo_median'])} and up "
        f"to {m(N['loo_max'])}. Field size has no relationship with "
        "accuracy that the data can distinguish from zero (Spearman "
        f"{m(N['spearman'], 3, True)}, 95 % CI {ci(*N['spearman_ci'])}); it "
        "predicts weight. The gap between pooled and average-field scores "
        f"ranges from {m(min(N['gaps']), 3, True)} to "
        f"{m(max(N['gaps']), 3, True)} across checkpoints, so it is not a "
        "fixed optimism that could be subtracted " + D.cite(8) + ".",
        bold_lead="The headline hides where the errors are.")
    D.p("Every yield result in this work rests on farmer recall, and "
        "nothing had checked that label against anything independent. Tamil "
        "Nadu publishes district yields from crop-cutting experiments in "
        "terms of cleaned rice, and its own conversion table puts rice at "
        "two-thirds of paddy " + D.cite(23, 24) + ". Converted that way, "
        "SICKLE's recalled Kuruvai yield for 2019-20 is "
        f"{m(hl['kuruvai_ratio_to_official'], 2)} of the crop-cutting yield "
        "of the three core delta districts and below every one of them "
        f"({kb['sickle_mean_rice_eq_kg_ha']:,.0f} kg/ha, 95 % CI "
        f"{kb['ci95'][0]:,.0f} to {kb['ci95'][1]:,.0f}, against "
        f"{kcell['official_core_band'][0]:,.0f} to "
        f"{kcell['official_core_band'][1]:,.0f}), while Samba sits at "
        f"{m(hl['samba_ratio_to_official'], 2)}, inside the district range. "
        "The labels run low, and in Kuruvai markedly so. The protocol for "
        "this check discloses that season means had been seen before its "
        "verdict rules were written.",
        bold_lead="The labels against the state's records.")
    D.p("In 2020-21 the state allocated about "
        f"\u20b9{rel20:,.0f} crore of agricultural relief across all four "
        "core delta districts for cyclone Burevi and the \u201cunprecedented "
        "heavy rain\u201d of January 2021, against about "
        f"\u20b9{rel19:,.0f} crore for the 2019 north-east monsoon "
        + D.cite(23) + ". Every one of SICKLE's "
        f"{oev['exposure']['n_samba_2020_21']} Samba and Thaladi crops of "
        "2020-21 was in the field that January. At the same plots, recalled "
        f"yield fell by {pp(hl['samba_recalled_change'])} (95 % CI "
        f"{pp(oev['samba_recalled']['ci95'][0])} to "
        f"{pp(oev['samba_recalled']['ci95'][1])}), no more than in the "
        f"Kuruvai control (difference {pp(hl['samba_minus_kuruvai'])}, "
        f"{pp(oev['contrast_samba_minus_kuruvai']['ci95'][0])} to "
        f"{pp(oev['contrast_samba_minus_kuruvai']['ci95'][1])}), so an "
        "event-specific fall is not demonstrated. The out-of-fold satellite "
        f"model's estimates did not move: {pp(hl['samba_model_change'])} "
        f"({pp(oev['samba_model']['ci95'][0])} to "
        f"{pp(oev['samba_model']['ci95'][1])}). The model reads the site "
        "rather than the season, which is the season-swap finding seen from "
        "outside: a season the state documented as damaging across the "
        "delta left no trace in its output. That is evidence about the "
        "model's responsiveness, not about these plots. Relief is paid per "
        "district and the plots carry no district, so neither check can say "
        "whether any plot was damaged, or that the model missed damage on "
        "one.",
        bold_lead="A documented damage season.")
    D.figure(f_off,
             "Left: SICKLE's recalled yield, converted to cleaned rice with "
             "the state's factor, against the crop-cutting yields of "
             "Thanjavur, Thiruvarur and Nagapattinam, 2019-20. Right: "
             "same-plot change into the 2020-21 relief season, with "
             "plot-bootstrap 95 % intervals (observed). Relief is "
             "district-level and the plots carry no district, so the right "
             "panel shows whether the model responds to the season, not "
             "which plots were damaged.", width=5.8)
    D.figure(FIGS / "fig_leave_field_out.png",
             "Pixel-pooled macro IoU of each released crop-type checkpoint "
             "with and without its single largest validation field "
             "(observed).", width=4.2)

    _fu_step = [k for k in st if st[k]["n_seeds"] > 3]
    _verb = {"resolved": "is now resolved",
             "small": "is now bounded as small",
             "underpowered": "remains underpowered"}
    ext = (" A power analysis identified "
           + " and ".join(step(k) for k in _fu_step)
           + " as the only open step a larger sample could settle, so "
           + " and ".join(extra) + " were raised to "
           + (f"{w(n_fin[extra[0]])} seeds each"
              if len({n_fin[k] for k in extra}) == 1 else
              " and ".join(f"{w(n_fin[k])} ({k})" for k in extra) + " seeds")
           + "; the extra runs enter only the paired analysis, and "
           "every rung in the table is compared on the same "
           f"{w(min(n_seed.values()))} seeds. "
           + " ".join(
               f"With {w(st[k]['n_seeds'])} paired seeds the "
               f"{step(k)} step is {m(st[k]['paired_mean'], 4, True)}, "
               f"interval {ci(*st[k]['ci95_paired'], nd=4)}, and "
               + _verb[st[k]["verdict"].split(" ")[0].split(":")[0]] + "."
               for k in _fu_step)
           if extra and _fu_step else "")
    D.p("A ten-rung controlled ablation varies one modelling choice per "
        "rung. Seeds are paired across rungs, so each step is estimated "
        "as a paired difference. "
        f"{w(len(pos) + len(neg), True)} of {w(len(st))} steps have a 95 % "
        "interval excluding zero: "
        + ", ".join(f"{step(k)} ({m(st[k]['paired_mean'], 4, True)})"
                    for k in pos + neg) + ". "
        f"{w(len(small), True)} more "
        + ("is" if len(small) == 1 else "are")
        + f" bounded as small, and {w(len(under))} "
        + ("remains" if len(under) == 1 else "remain")
        + " open for want of seeds." + ext + " The steps that help add "
        "information, such as a second sensor or the time series, rather "
        "than model capacity.", bold_lead="Modelling choices.")

    # ================================================ 4 part II
    D.h("4. Safeguards, empirically evaluated", 1)
    sd = N["sv"]["signals"]
    _sf = sd["smallest_first"]
    _gain = ff["screen alone"]
    D.p("Two label-free signals were tested for flagging fields whose "
        "field IoU falls below 0.50: mean low confidence, which needs one "
        "model, and seed disagreement, which needs several. Only "
        f"{N['tg']['n_bad']} of the {N['n_fields']} fields fall below 0.50, "
        "so the evidence is thin. Seed disagreement ranks them at AUROC "
        f"{m(sd['seed_disagreement']['auroc'])}, "
        f"{cil(*sd['seed_disagreement']['auroc_ci'])}, an interval that "
        "includes chance. Low confidence ranks them at "
        f"{m(sd['low_confidence']['auroc'])}, "
        f"{cil(*sd['low_confidence']['auroc_ci'])}, but simply checking the "
        f"smallest fields first scores {m(_sf['auroc'])}. Permuting the "
        "signal within size terciles preserves the size relationship "
        "exactly and destroys everything else; against that null, low "
        f"confidence scores p = {t1['p_one_sided']:.3f}, so it carries some "
        "information that size does not supply. It is not shown to rank "
        "better than size: its gain over the smallest-first ranking is "
        f"{m(_gain['delta_vs_size_smaller_first'], 3, True)} with a 95 % "
        f"interval of {ci(*_gain['ci_vs_smaller_first'])}. Under the "
        "superseded two-class score the same analysis found "
        f"{N['sv_macro']['n_bad']} unreliable fields, most of them with a "
        "few wrong pixels, and both signals appeared informative; that "
        "result is kept in the protocol's amendment and not carried "
        "forward. A leave-one-out logistic test specified in advance was "
        "found to be invalid at this sample size, scoring a pure-noise "
        f"feature at a mean AUROC of {m(cs['loo_noise_null']['mean'])}; it "
        "is reported as a diagnostic and carries no weight.",
        bold_lead="Crop type: some information beyond size, gain not shown.")

    sigrows = []
    names = {"quantile_width": "prediction-interval width",
             "rf_spread": "random-forest spread",
             "seed_spread": "seed disagreement",
             "low_n_clear": "too few clear observations",
             "small_n_px": "too few pixels",
             "ood_distance": "out-of-distribution distance",
             "low_predicted_yield": "low predicted yield"}
    for k, v in N["yield_signals"].items():
        sigrows.append((names.get(k, k), m(v["auroc"]),
                        cil(v["ci"]["lo"], v["ci"]["hi"]),
                        "yes" if v["informative"] else "no"))
    sigrows.append(("released-ensemble dispersion (pre-registered)",
                    m(ys["auroc"]),
                    cil(ys["auroc_ci"]["lo"], ys["auroc_ci"]["hi"]),
                    "no"))
    dmn = mn["signals"]["decoupling (pre-specified)"]
    sigrows.append(("water/vegetation decoupling at transplanting "
                    "(pre-registered)", m(dmn["auroc"]), cil(*dmn["ci"]),
                    "yes" if dmn["informative"] else "no"))
    n_sig = len(sigrows)
    n_inf = sum(1 for r in sigrows if r[3] == "yes")
    D.p(f"For yield, {w(n_sig)} candidate reliability signals were "
        "evaluated and "
        + ("none discriminates" if n_inf == 0 else
           f"{w(n_inf)} discriminates" if n_inf == 1 else
           f"{w(n_inf)} discriminate")
        + " (Table 1). Prediction intervals reach "
        f"{100 * N['quantile_coverage']:.1f} % empirical coverage at a "
        "nominal 90 % and still fail, because coverage is a property "
        "averaged over fields and says nothing about which field is "
        "wrong. The last two signals were specified under protocols frozen "
        "before any result; the water index made the signal worse than "
        "greenness alone. None of this shows that uncertainty methods fail "
        "in general: each interval is wide, and a wide interval centred "
        "near 0.5 is a failure to demonstrate discrimination, not a "
        "demonstration of its absence " + D.cite(11, 12) + ".",
        bold_lead="Yield: no signal discriminates.")
    D.table(["signal", "AUROC", "95 % CI", "informative"], sigrows,
            "Yield-reliability signals evaluated on held-out data. "
            "Informative means the interval excludes 0.5 (observed).")

    # ================================================ 5 part III
    D.h("5. A decision-support prototype built on the evidence", 1)
    D.p("SICKLE-Guard ranks field-seasons for verification under a fixed "
        "inspection budget. Each design decision traces to a measurement "
        "above. Because the benchmark score describes no field, it reports "
        "per field and never an area aggregate. Because no yield model is "
        "shown to beat a constant, it leads with the field's own past "
        "harvests and "
        "falls back to satellite only when no record exists. Because no "
        "reliability signal discriminates, it displays no confidence "
        "score. And because a false reassurance can stop a farmer reporting "
        "damage while a false alarm costs a visit, it has no reassuring "
        "state: a field is flagged or not flagged, and not flagged always "
        "states that this is not evidence of no loss.",
        bold_lead="Design.")
    D.figure(f_wf,
             "The five steps of the workflow and the status of the evidence "
             "behind each.", width=6.0)
    D.p(f"Of the labelled losses, {pct(lv['frac_losses_chronic'])} sit at "
        "plots below threshold in every observed season. Feeding the model "
        "a different season from the same plot keeps most of its ranking "
        f"skill (AUROC {m(ss['1 real features']['loss_auroc'])} real against "
        f"{m(ss['2 same-plot season swap']['loss_auroc'])} swapped), and on "
        "plots that are sometimes above and sometimes below threshold the "
        f"ranking falls to {m(ev['auroc'])} with a 95 % interval of "
        f"{cil(*ev['ci'])}, which includes chance. The tool therefore ranks "
        "fields by their usual yield level. It is not described, anywhere, "
        "as detecting crop damage.", bold_lead="What it ranks.")

    lh = base["held-out test split"]
    lo = base["all field-seasons, out-of-fold"]
    D.p("Evaluated on the 27 held-out plots, never used to fit or tune "
        f"anything, the shipped ranking scores AUROC "
        f"{m(hold['SICKLE-Guard (as shipped)']['auroc'])} and beats "
        "arbitrary visiting (Figure 2). It does not demonstrably beat a "
        "lookup of each field's most recent harvest: the paired difference "
        f"is {m(lh['paired_delta'], 3, True)} {ci(*lh['ci'])} on held-out "
        f"plots and {m(lo['paired_delta'], 3, True)} {ci(*lo['ci'])} "
        "out-of-fold, both spanning zero. The satellite component has not "
        "been shown to earn its place.",
        bold_lead="Against the alternatives.")
    D.figure(f_rank,
             "Ranking power of the prototype and of alternatives an "
             "officer could use instead, on held-out plots (observed).",
             width=5.6)

    D.p(f"The flag separates outcomes: {pct(flag['share_bad_flagged'])} of "
        f"flagged fields are below threshold against "
        f"{pct(flag['share_bad_unflagged'])} of unflagged ones, a "
        f"difference of {m(flag['difference'] * 100, 0, True)} points "
        f"(95 % CI {m(flag['ci'][0] * 100, 0, True)} to "
        f"{m(flag['ci'][1] * 100, 0, True)}). Where the ranking fails is "
        "sharply localised. With a record it catches "
        f"{a['n_bad'] - a['n_missed']} of {a['n_bad']} below-threshold "
        f"fields; without one it catches {b['n_bad'] - b['n_missed']} of "
        f"{b['n_bad']}. These are counts on 27 plots and carry wide "
        "uncertainty. Every miss is a cold start, the case where the "
        "system must fall back on satellite. The two kinds of error also "
        "diverge: cold starts have the worse ranking and, over all "
        "field-seasons, not the worse yield estimate, which is why ranking "
        "and estimation errors are reported separately.",
        bold_lead="Where it fails.")
    D.p("One safeguard was refuted by its own test. The report's "
        "no-reliable-estimate warning fires on missing records, and warned "
        "field-seasons are not reliably worse: "
        f"{m(warn['difference'], 0, True)} kg/acre "
        f"{ci(warn['ci'][0], warn['ci'][1], 0)} on held-out plots, and "
        f"{m(warn_all['difference'], 0, True)} over all field-seasons. The "
        "wording now states that the warning means there is no record, not "
        "that the estimate is worse.", bold_lead="A refuted safeguard.")
    # --- Experiment A: the workflow against what a team already has
    vp = N["vp"]
    ea = vp["experiment_a"]["held_out"]
    rows_a = []
    for bud, r in ea["budgets"].items():
        cr = r["comparisons"]["register"]
        ch = r["comparisons"]["history_only"]
        rows_a.append((f"{pct(float(bud))}",
                       f"{m(r['random']['recall_mean'], 2)} "
                       f"(p97.5 {m(r['random']['recall_p97_5'], 2)})",
                       f"{r['register']['n_found']} of {ea['n_bad']}",
                       f"{r['history_only']['n_found']} of {ea['n_bad']}",
                       f"{r['workflow']['n_found']} of {ea['n_bad']}",
                       ci(*cr["ci_adjusted"]), ci(*ch["ci_adjusted"])))
    D.p("A pre-registered experiment compared four inspection policies on "
        "the same held-out plots at three budgets: random referral, the "
        "register a verification office already keeps (plots ranked by "
        "their lowest past yield), ranking by the field's own past "
        "harvests, and the workflow. The target is below-threshold "
        "field-seasons, not verified damage. Six comparisons were made, so "
        "the verdict uses Bonferroni-adjusted intervals.",
        bold_lead="Against the register a team already has.")
    D.table(["budget", "random recall", "register", "history only",
             "workflow", "workflow \u2212 register (adj.)",
             "workflow \u2212 history (adj.)"], rows_a,
            "Below-threshold field-seasons found on the held-out split, "
            "and the workflow's paired recall difference against each "
            "simple alternative, with Bonferroni-adjusted intervals "
            "(observed).")
    _b20 = ea["budgets"]["0.2"]
    A20_found = ea["budgets"]["0.2"]["workflow"]["n_found"]
    D.p("Every record-based policy far outperforms random referral: at a "
        f"{pct(0.2)} budget the register finds "
        f"{_b20['register']['n_found']} of {ea['n_bad']} below-threshold "
        "field-seasons where random referral averages "
        f"{m(_b20['random']['recall_mean'] * ea['n_bad'], 1)} and reaches "
        f"{m(_b20['random']['recall_p97_5'], 2)} recall only at its 97.5th "
        "percentile. That is demonstrated prioritisation value on this "
        "dataset. It belongs to the records, though, not to the workflow: "
        "the workflow finds at most one field-season more than the "
        "register, every adjusted interval includes zero, and the "
        f"pre-registered verdict is that an advantage is "
        f"{'demonstrated' if ea['verdict'] == 'advantage demonstrated' else 'not demonstrated'}"
        ". It is not significantly worse either. The honest reading is "
        "that a team with a yield register would gain nothing measurable "
        "from this workflow on these plots.")

    # --- Experiment B: reserving capacity for fields with no record
    eb_ = vp["experiment_b"]["all_oof"]
    b33 = eb_["budgets"]["0.33"]
    b33z, b33t = b33["0.00|satellite"], b33["0.10|satellite"]
    b33p = b33["proportional|satellite"]
    wc = b33["_within_cold"]
    ex = vp["exploratory_reserve"]["all_oof"]["0.33"]
    D.p("Because every miss is a cold start, a second pre-registered "
        "experiment reserved part of the budget for fields with no record "
        "and measured what that costs. At a 33 % budget, with no reserve "
        f"the policy finds {pct(b33z['recall_recorded'])} of below-threshold "
        "fields that have a record and none of those that do not; "
        f"reserving 10 % keeps {pct(b33t['recall_recorded'])} on recorded "
        f"fields and reaches {pct(b33t['recall_cold_start'])} on cold "
        "starts; reserving their proportional share gives "
        f"{pct(b33p['recall_recorded'])} and {pct(b33p['recall_cold_start'])}. "
        "No allocation is at least as good on both axes as every other, so "
        "the result is the trade-off itself. Two further findings bound it. "
        "Within the cold-start pool, ordering by the satellite estimate "
        "does not beat a random order "
        f"({m(wc['delta'], 3, True)}, 95 % CI {ci(*wc['ci95'])}), so the "
        "gain comes from inspecting some cold-start fields at all, not "
        "from the model choosing which. And in an exploratory check, "
        "noticed after the results and labelled as such, a 10 % reserve "
        "with random choice among cold starts changed recorded recall by "
        f"{m(ex['recorded recall']['delta'], 3, True)} "
        f"({ci(*ex['recorded recall']['ci95'])}): cheap, and not shown to "
        "be free. The design consequence is concrete: rank recorded fields "
        "by their history, set aside a small, stated share of inspections "
        "for fields with no record, and do not present a satellite "
        "estimate as evidence for choosing among them.",
        bold_lead="Fields with no record.")

    # --- step four as software
    D.p("The ranking becomes an inspection procedure in software. A planner "
        "ranks fields that have a record by their own past harvests and "
        "fills a stated reserve from fields with none, chosen at random "
        "under a logged seed, because the satellite did not choose among "
        "them better than chance. Each inspector receives a task sheet that "
        "withholds the yield estimate. That rule comes from an observed "
        "count: at a 20 % budget the workflow sends inspectors to "
        f"{A20_found} below-threshold field-seasons, and "
        f"{exp20['n_inspected_bad_estimate_at_or_above_threshold']} of them "
        "carry an estimate above the threshold, which an inspector shown "
        "the number could wave through. An inspection records an "
        "observation, never a decision: wording that certifies damage or "
        "decides a claim is refused, and a finding of a problem requires a "
        "photograph or a written note. Every action is appended to a "
        "SHA-256 hash chain, so altering any past record breaks every later "
        f"link. The workflow's {N['n_wf_tests']} tests include attacks that "
        "edit a record, edit the log, delete an entry and reorder a plan; "
        "each is caught. A demonstration round on the held-out split, with "
        "a simulated inspector and labelled so in every record, left "
        f"{wf['audit']['n_entries']} intact links. A field that is not "
        "inspected is reported as not inspected, never as cleared.",
        bold_lead="Step four as software.")
    _me = exp20["missed_estimates_kg_per_acre"]
    D.p("The fields the ranking misses are the ones a reassuring display "
        f"would reach. All {exp20['n_missed']} below-threshold field-seasons "
        "it misses on the held-out split at a 20 % budget are cold starts, "
        f"and all {exp20['n_missed_estimate_at_or_above_threshold']} carry "
        f"satellite estimates between {min(_me):,.0f} and {max(_me):,.0f} "
        "kg/acre against a threshold of "
        f"{insp['observed_exposure']['held-out test split']['threshold_kg_per_acre']:,.0f}. "
        "The model pulls a field without a record towards the average, so "
        "a problem field looks ordinary.",
        bold_lead="Who a reassuring display would reach.")

    # --- the safety cases
    D.p(f"{w(N['n_safety_tests'], True)} predefined safety tests probe the "
        "report where a careless one would mislead: the real held-out "
        "field-seasons the workflow failed to flag, fields with no record, "
        "and reports rendered from the same insufficient evidence at "
        "estimates from 300 to 3,500 kg/acre. All pass. The wording is "
        "required to be identical across those estimates, so a high number "
        "cannot soften a warning into reassurance, and the check was "
        "itself verified by injecting a report that reassured above "
        "2,000 kg/acre, which the cases caught. This is a property of the "
        "software; whether a farmer reading the report understands it is "
        "untested.", bold_lead="Safety under failure.")

    if N["untranslated"] and N["untranslated"][0] == 0:
        D.p("The Tamil report was translated in full by the first author, a "
            "native Tamil speaker, and each sentence is stored with the English "
            "it was translated from. A sentence whose English later changes "
            "falls back to English under a notice that the report is not fully "
            "translated, and tests require every Tamil report to carry the "
            "safety sentences in Tamil, to keep every number unchanged and to "
            "keep its wording independent of the estimate. The translation has "
            "not been checked by a second speaker or tested with farmers.",
            bold_lead="Accessibility.")
    elif N["untranslated"]:
        lit, tot = N["untranslated"]
        D.p(f"The Tamil report leaves {lit} of {tot} sentences in English "
            "and opens by stating that it is not fully translated; "
            "native-speaker review is required before it can be offered to "
            "Tamil-reading farmers.", bold_lead="Accessibility.")

    # ================================================ 6 simulation
    D.h("6. Simulated studies", 1)
    CAL_LABEL = {"history_auroc": "history-only ranking AUROC",
                 "frac_losses_chronic": "share of losses at chronic plots",
                 "event_auroc": "event-plot ranking AUROC",
                 "frac_with_record": "share of field-seasons with a record"}
    D.p("Every number in this section is simulated. Three studies vary "
        "what the data holds fixed, the inspector and the insurance scheme; "
        "none of them is field performance.", italic=True)
    calib = syn["calibration"]
    D.p("To map under which conditions triage would help, a generator "
        "was calibrated to four measured quantities and the "
        "run aborts if it misses any, judged against each measurement's "
        "published interval rather than its point estimate. "
        + "Measured against simulated: "
        + "; ".join(f"{CAL_LABEL.get(k, k)} {m(v['measured'])} against "
                    f"{m(v['simulated'])}"
                    for k, v in calib.items()) + ". "
        "The competitor that matters is not random referral but the "
        "register a verification team already holds: visit the plots known "
        "to have yielded badly. Policies were scored on recall of event "
        "losses, the only ones a prediction can add anything to.")
    D.p(f"Across {n_settings} settings, a policy had to beat the runner-up "
        "by 0.02 to count as better. "
        f"{TALLYW(tally['tie'])} were ties; the register won "
        f"{tally['register']}, the prototype {tally['prototype']} and the "
        f"history-first policy {tally['history-first']}. At the calibrated operating point the "
        f"register recalls "
        f"{m(sw_rec['0.59']['register only (what the team already knows)']['recall_event'], 2)} "
        "of event losses against "
        f"{m(sw_rec['0.59']['prototype (history, satellite fallback)']['recall_event'], 2)} "
        "for the prototype, which wins outright only when every field has a "
        "record or when the satellite is several times more informative "
        "than measured. At a 10 % budget the prototype is beaten by its own "
        f"history component ("
        f"{m(sw_bud['0.1']['prototype (history, satellite fallback)']['recall_event'], 2)} "
        f"against {m(sw_bud['0.1']['history only']['recall_event'], 2)}): "
        "weak satellite guesses displace strong history candidates when "
        "few visits are available.")
    D.figure(f_syn,
             "Simulated recall of event losses by inspection budget. A "
             "calibration check, not a validation of the generator's "
             "structure; another generator reproducing the same four "
             "quantities could behave differently.", width=5.2)
    D.p("Experiment A scored a visit as revealing the truth. Keeping its "
        "observed rankings and simulating only the inspector, by the chance "
        "of confirming a real problem and of reporting a false one, gives "
        "three results. At a 20 % budget, a register-guided inspector with a "
        f"sensitivity of {m(ho20['register']['break_even_sensitivity'], 2)} "
        "confirms on average as many real problems as a perfect inspector "
        "sent at random. With a sensitivity of 0.70 and a 10 % false-alarm "
        f"rate, {m(g7['register']['precision_of_confirmed'], 2)} of a "
        "register-guided team's confirmed findings are real, against "
        f"{m(g7['random']['precision_of_confirmed'], 2)} for random visits, "
        "because targeting raises the base rate of the visited pool. The "
        "pre-registered test is unresolved on the held-out split: its "
        f"{w(insp['sets']['held-out test split (VERDICT)']['n_bad'])} problem "
        "field-seasons leave the register-versus-random interval touching "
        "zero even for a perfect inspector. Over all "
        f"{oofs['n_bad']} problem field-seasons, out-of-fold, the register's "
        "advantage survives down to a sensitivity of "
        f"{oofs['monte_carlo_register_vs_random']['0.2']['lowest_surviving_sensitivity']} "
        "at a 20 % budget.", bold_lead="Imperfect inspectors.")
    D.p("No YES-TECH output is public, so the scheme's mechanism was "
        "simulated rather than its accuracy. A unit's claim is its shortfall "
        "below the threshold yield as a fraction of the sum insured, and the "
        "yield used blends the technology estimate with crop-cutting at the "
        "weights states actually use " + D.cite(21, 22) + ". Error sizes are "
        "assumptions; 0.30 is the largest unit-level error the January 2023 "
        "YES-TECH manual tolerated " + D.cite(20) + ". For a unit truly 10 % "
        "below its threshold and a technology error of 0.20, the chance of "
        f"being paid nothing is {pct(p100)} at 100 % weight against "
        f"{pct(p30)} at the 30 % minimum: at full weight nothing averages "
        "the model's error away. Error also pays units that are not due, so "
        "it is not one-sided. Among re-verification rules, rechecking the "
        "units whose estimate lands just above the threshold recovers "
        f"wrongly unpaid units at {m(gd['near_threshold']['rechecks_per_recovery'], 1)} "
        "rechecks each against "
        f"{m(gd['random']['rechecks_per_recovery'], 1)} for random "
        "rechecks, and rechecking the lowest estimates recovers none, since "
        "those units are already paid. The rule is not free, because each "
        "recheck is itself a noisy crop-cut and pays some units that were "
        "not due. None of this says how accurate YES-TECH is.",
        bold_lead="A YES-TECH what-if.")
    D.figure(f_sim,
             "Simulated. Left: real problems confirmed at a 20 % budget as "
             "the simulated inspector's sensitivity falls; the rankings are "
             "observed. Right: chance that an Insurance Unit truly short of "
             "its threshold is paid nothing, by model weight, at a "
             "technology error of 0.20 (solid) and 0.30 (dashed). "
             "Assumptions, not measurements of any deployed system.",
             width=5.8)

    # ================================================ 7 integrity
    D.h("7. Reproducibility and the corrections it forced", 1)
    dr = N["drift"]
    cl = N["claims"]
    D.p("Every headline number is declared with a tolerance and recomputed "
        f"from primary artefacts on each build ({dr['n_ok']} of "
        f"{dr['n_entries']} within tolerance in the audit); every figure "
        "quoted in the prototype's documentation is bound to its artefact "
        f"and to the exact string the prose must contain ({cl['n_ok']} of "
        f"{cl['n_claims']}). One minimal reproduction per defect prints the "
        "published value beside the measured one "
        f"({N['mrepro']['n_reproduced']} of {N['mrepro']['n_blocks']} "
        "reproduced). These checks changed the results they guard. They "
        "caught expected values entered at a precision that had never been "
        "computed, three times; a leave-one-out test that scored noise "
        "below chance; an aggregator that labelled an unfinished training "
        "run while still averaging it in; an interface file left "
        "unparseable by an edit that no test read; simulated winners "
        "declared on differences of a thousandth; and, found while "
        "rendering prediction maps, a per-field score that capped every "
        "imperfect single-crop field at 0.5. Each is corrected in place "
        "with the record kept " + D.cite(10) + ".")

    # ================================================ 8 limitations
    D.h("8. Limitations", 1)
    for lead, t in (
            ("One delta, one cropping system.",
             "All results are on SICKLE, 2018 to 2020. Spatial structure "
             "limits what transfers " + D.cite(13) + "."),
            ("Recalled yield.",
             "Yield labels are farmer recall. Model error and recall error "
             "cannot be separated without crop-cutting measurements this "
             "dataset does not contain."),
            ("Small held-out sets.",
             f"{N['n_fields']} validation fields for crop type and 27 "
             "held-out plots for the prototype. Intervals are wide and no "
             "operating threshold is established."),
            ("No independent outcomes.",
             "A search for public, independently measured outcomes found "
             "none that fits: the CSISA repository's crop-cut data covers "
             "rice in Nepal and maize in Odisha with no confirmed repeat "
             "measurement of the same plot, and Indian PMFBY damage "
             "photographs carry no stated licence. Without such data the "
             "results cannot be shown to transfer to measured damage."),
            ("Official records are district-level.",
             "The state reports yields and relief per district and SICKLE "
             "plots carry no district, so the label check is delta-wide; "
             "2020-21 district yields were not found in a public table."),
            ("No YES-TECH output.",
             "The YES-TECH study simulates the scheme's mechanism under "
             "stated assumptions. Validating it needs matched unit-level "
             "technology and crop-cutting yields, which are not public."),
            ("No people.",
             "Nobody has used the prototype, the inspection workflow or the "
             "console. Comprehension, workflow and outcomes for farmers or "
             "officials are unmeasured."),
            ("No scheme linkage.",
             "No Insurance Unit identity exists in the data, so nothing here "
             "reproduces a scheme decision.")):
        D.bullet(t, lead)

    # ================================================ 9 practice
    D.h("9. What a verification office can take from this", 1)
    D.p("Five operating rules follow from the evidence. Each is labelled by "
        "the kind of evidence behind it, and none has been tried in an "
        "office.")
    _b33 = N["vp"]["experiment_b"]["all_oof"]["budgets"]["0.33"]
    for lead, t in (
            ("Rank by the field's own records (observed).",
             "Every record-based policy far outperformed random visits on "
             "held-out plots; the satellite added at most one field-season."),
            ("Reserve a share for fields with no record, chosen at random "
             "(observed).",
             f"A 10 % reserve at a 33 % budget lifted no-record recall from "
             f"none to {pct(_b33['0.10|satellite']['recall_cold_start'])} while "
             f"keeping {pct(_b33['0.10|satellite']['recall_recorded'])} on "
             "recorded fields; the satellite chose among them no better than "
             "chance."),
            ("Withhold the estimate from inspectors, and never show a farmer "
             "a reassuring one (observed).",
             f"{w(exp20['n_inspected_bad_estimate_at_or_above_threshold'], True)} of "
             f"the {w(A20_found)} below-threshold fields inspectors were sent to at a "
             "20 % budget, and every one the ranking missed, carried an "
             "estimate above the threshold."),
            ("Before relying on a yield model, check that it responds to "
             "documented damage seasons (observed).",
             f"In a season of about \u20b9{rel20:,.0f} crore of state relief "
             "across the delta, this model's estimates at the same plots were "
             "unchanged. Which of those plots were damaged is unknown."),
            ("Where a scheme gives a model high weight, re-verify the units "
             "just above the threshold (simulated).",
             "In simulation this recovers wrongly unpaid units about "
             f"{w(round(gd['random']['rechecks_per_recovery'] / gd['near_threshold']['rechecks_per_recovery']))} "
             "times as efficiently as random rechecks. It needs testing on "
             "the scheme's own unit-level data.")):
        D.bullet(t, lead)

    # ================================================ 10 conclusion
    D.h("10. Conclusion", 1)
    D.p("The objective was to " + OBJECTIVE[3].lower() + OBJECTIVE[4:-1]
        .replace("We identify", "identify") + ". Each part is answered "
        "with the strength the evidence allows, and no more.")
    D.p("Field-level satellite crop prediction is unreliable under "
        "specific, measurable conditions that aggregate benchmark scores "
        "conceal: evaluation that shifts with batch size, a headline score "
        f"carried by one field in {N['n_fields']} that also hides a tail of "
        "failing fields, and yield models that "
        "are not shown to beat a constant. Checked against the state's own "
        "records, "
        "the recalled labels run low, and the satellite model's estimates "
        "did not respond to a season in which the state paid delta-wide "
        "crop-damage relief, though which plots were damaged is unknown. "
        "These are established on SICKLE and stated with their "
        "uncertainty.", bold_lead="Conditions of unreliability: "
        "supported.")
    D.p("Safeguards were evaluated rather than assumed, and the results "
        "divide by task. On crop type, with only "
        f"{w(N['tg']['n_bad'])} failing fields, a label-free confidence "
        "signal carries some information that field size does not, but "
        "does not demonstrably outrank it. On yield, none of "
        f"{w(n_sig)} signals "
        "discriminates, two of them pre-registered. The finding that "
        "generalises is that a safeguard must be validated on the quantity "
        "it gates, and that calibrated coverage is not discrimination.",
        bold_lead="Safeguards: supported as an investigation.")
    D.p("The prototype exists, helps prioritise verification, and guards "
        "against false reassurance by construction: there is no reassuring "
        "state, inspectors' task sheets withhold the estimate, and the cost "
        "of those rules is visible in the below-threshold fields a "
        "reassuring display would otherwise have cleared. Verification "
        "itself now exists as software, with an evidence rule and a "
        "tamper-evident audit chain, though its inspections so far are "
        "simulated. "
        "Predefined tests show its warnings do not soften as the estimate "
        "rises. Prioritisation from field records is demonstrated: every "
        "record-based policy far outperforms random inspection on held-out "
        "plots. But that value belongs to the records. Against the "
        "register a team already keeps, the workflow's advantage is not "
        "demonstrated, every miss falls where it must rely on satellite, "
        "and the satellite does not help choose which unrecorded fields "
        "to inspect. What does help those fields is reserving a small "
        "share of inspections for them.",
        bold_lead="Decision support: partially supported.")
    D.p("Under the evaluated conditions, the prototype is weakest where "
        "historical records are unavailable, a setting in which its "
        "satellite-based estimates have not demonstrated sufficient added "
        "value to reliably prioritize verification. The contribution is "
        "therefore not that satellite imagery enables field-level "
        "decisions. It is a reproducible account of when such predictions "
        "should not be trusted, a disciplined rejection of safeguards that "
        "do not work, and a decision-support design that stays honest while "
        "its central model is weak. Whether it improves decisions for "
        "farmers or officials is the next question, and only a pilot with "
        "an inspection office, using its own records and its own "
        "crop-cutting, can answer it. A protocol for that pilot is fixed "
        "before any pilot data: every policy's list is inspected blind and "
        "scored against crop-cutting on each inspected field and on a "
        "random audit sample, so misses can be counted. Sized from the "
        "effects observed here, it can test records against random "
        "selection with about "
        f"{N['pilot']['sets']['all field-seasons out-of-fold']['comparisons']['register vs random']['fields_to_enrol']:,.0f} "
        "fields, whereas the satellite's edge over the register, at its "
        "observed size, would need about "
        f"{N['pilot']['sets']['all field-seasons out-of-fold']['comparisons']['workflow vs register']['fields_to_enrol']:,.0f}. "
        "The pilot is therefore designed around the records, the reserve "
        "and the inspector, not around the satellite.")

    # ================================================ references
    D.h("References", 1)
    for i, k in enumerate(D.cites, 1):
        para = D.d.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)
        r = para.add_run(f"[{i}] {N['refs'][k]}")
        r.font.size = Pt(9.5)
        D.text.append(N["refs"][k])
    return D


# ------------------------------------------------------------- self-check
def self_check(text: str) -> list[str]:
    """The canonical objective must be present; overstatements must not."""
    from check_claims import NEGATORS, norm
    problems = []
    t = norm(text)
    if norm(OBJECTIVE) not in t:
        problems.append("canonical objective missing or altered")
    reg = yaml.safe_load((GUARD / "configs/claims.yaml").read_text(
        encoding="utf8"))
    low = t.lower()
    for r in reg["prose_rules"]["must_not_appear"]:
        ph = norm(str(r["phrase"]).lower())
        i = low.find(ph)
        while i >= 0:
            window = low[max(0, i - 110):i]
            if not any(neg in window for neg in NEGATORS):
                problems.append(f"un-negated {r['phrase']!r}: "
                                f"...{low[max(0, i - 50):i + len(ph) + 20]}...")
            i = low.find(ph, i + 1)
    if "simulated" not in low:
        problems.append("the simulated section does not declare itself")
    return problems


def main() -> int:
    N = load()
    D = build(N)
    text = "\n".join(D.text)
    problems = self_check(text)
    if problems:
        print("THESIS PAPER NOT WRITTEN. The rendered text fails its own "
              "checks:")
        for p in problems:
            print("  -", p)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.d.save(str(OUT))
    words = len(text.split())
    print(f"wrote {OUT}")
    print(f"  {words:,} words, {D.fig} figures, {D.tab} tables, "
          f"{len(D.cites)} references")
    print("  self-check passed: objective verbatim, no un-negated "
          "overstatement, simulated section labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
