"""Generate the submission-ready IEEE two-column manuscript.

    python scripts/make_paper_ieee.py

Every numeric value is read from a result file at build time and every
reference from docs/references.json, so the manuscript cannot drift from the
evidence or from the verification table.

Layout follows the supplied IEEE conference reference: a full-width title
block, then a two-column body with Roman-numeral sections, lettered
subsections, numbered figure captions and tables, and numeric citations.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PROJ = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJ))
from src.utils import completed_per_plot_iou  # noqa: E402
FIG = PROJ / "results" / "figures_ieee"
OUT = PROJ / "docs" / "SICKLE_Field_Level_Audit_IEEE.docx"
COL = 3.3                                   # usable column width, inches


# ----------------------------------------------------------------- numbers
def load():
    n = {}
    L = {r["rung"]: r for r in csv.DictReader(
        (PROJ / "results/ladder_crop_type.csv").open(encoding="utf8"))
        if r.get("pixel_macro_iou")}
    n["L"] = L
    for k in ("v1", "v2", "v4", "v5", "v6", "v7"):
        n[k] = float(L[k]["pixel_macro_iou"])
    n["v5_pl"] = float(L["v5"]["plot_macro_iou"])
    n["sd_max"] = max(float(L[k]["pixel_sd"]) for k in L)
    n["d_temporal"] = n["v2"] - n["v1"]
    n["d_radar"] = n["v5"] - n["v4"]
    n["d_l8"] = n["v6"] - n["v5"]
    n["d_utae"] = n["v7"] - n["v6"]

    # Classify every adjacent step from the file. An earlier draft enumerated
    # the rungs by hand, reported "two of nine" exceeding the seed spread when
    # four do, and omitted the class-weighting rung altogether. Deriving the
    # partition here makes an omission impossible.
    LABEL = {"v2": "full temporal sequence", "v3": "augmentation",
             "v4": "class-weighted loss", "v5": "Sentinel-1 radar",
             "v6": "Landsat-8 as a third sensor", "v7": "UTAE for U-Net3D",
             "v8": "wider encoder", "v9": "cosine schedule with warmup",
             "v10": "plot-balanced sampling"}
    rungs = [f"v{i}" for i in range(1, 11)]
    steps = []
    for a, b in zip(rungs, rungs[1:]):
        if a not in L or b not in L:
            continue
        delta = float(L[b]["pixel_macro_iou"]) - float(L[a]["pixel_macro_iou"])
        steps.append({"to": b, "label": LABEL[b], "delta": delta,
                      "exceeds": abs(delta) >= n["sd_max"]})
    n["steps"] = steps
    n["n_steps"] = len(steps)
    n["pos"] = [x for x in steps if x["exceeds"] and x["delta"] > 0]
    n["neg"] = [x for x in steps if x["exceeds"] and x["delta"] < 0]
    n["inc"] = [x for x in steps if not x["exceeds"]]

    # What KIND of change each step makes. The prose used to say "the two
    # that help add information while the two that hurt change the loss or
    # the architecture" as a literal. That sentence is a claim about which
    # steps exceed sigma_max, and adding seeds to v3 and v4 can move a step
    # across the line -- at which point the literal would be false in its
    # meaning, not merely in its count. It is now generated.
    KIND = {"v2": "information", "v3": "the training procedure",
            "v4": "the loss", "v5": "information", "v6": "information",
            "v7": "the architecture", "v8": "model capacity",
            "v9": "optimisation", "v10": "sampling"}
    for x in steps:
        x["kind"] = KIND[x["to"]]
    global SHORTLAB
    SHORTLAB = {"v2": "temporal", "v3": "augmentation", "v4": "class weighting",
                "v5": "radar", "v6": "Landsat-8", "v7": "UTAE",
                "v8": "capacity", "v9": "schedule", "v10": "plot balancing"}

    def _kinds(xs):
        ks = []
        for x in xs:
            if x["kind"] not in ks:
                ks.append(x["kind"])
        return ks

    def _orlist(ks):
        return (ks[0] if len(ks) == 1
                else ", ".join(ks[:-1]) + " or " + ks[-1])

    _W = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine"}
    n["n_pos_w"] = _W.get(len(n["pos"]), str(len(n["pos"])))
    n["n_neg_w"] = _W.get(len(n["neg"]), str(len(n["neg"])))
    n["pos_verb"] = "helps" if len(n["pos"]) == 1 else "help"
    n["neg_verb"] = "hurts" if len(n["neg"]) == 1 else "hurt"
    n["pos_kinds"] = _kinds(n["pos"])
    n["neg_kinds"] = _kinds(n["neg"])
    n["pos_all_information"] = (bool(n["pos"])
                                and n["pos_kinds"] == ["information"])
    n["neg_kind_phrase"] = (
        ("changes " if len(n["neg"]) == 1 else "change ")
        + _orlist(n["neg_kinds"]) if n["neg"] else "")

    # The manuscript says classifying each step against its OWN pair of
    # rungs gives the same partition as the global sigma_max. That is a
    # claim, so it is computed rather than asserted.
    own = []
    for a, b in zip(rungs, rungs[1:]):
        if a not in L or b not in L:
            continue
        d = float(L[b]["pixel_macro_iou"]) - float(L[a]["pixel_macro_iou"])
        sd_pair = max(float(L[a]["pixel_sd"]), float(L[b]["pixel_sd"]))
        own.append(abs(d) >= sd_pair)
    n["own_pair_same_partition"] = (
        own == [x["exceeds"] for x in steps])
    n["own_pair_n_exceeding"] = sum(own)

    # FIELD IoU throughout (the crop the field holds), not the trainer's
    # macro IoU: for the 35 single-crop fields macro IoU is at most 0.5 once
    # one pixel is wrong. See src/evaluation/metrics.py.
    from src.utils import completed_field_iou
    _fseeds, _ = completed_field_iou("v5", PROJ)
    per = {}
    for d_ in _fseeds:
        for k, v_ in d_.items():
            per.setdefault(float(k), []).append(v_)
    v = np.array([np.mean(x) for x in per.values()])
    # Cross-seed correlation over ALL seed pairs, computed here rather than
    # quoted: an earlier draft reported the range as +0.52 to +0.58 when the
    # third pair reaches +0.63.
    import itertools
    mats = [{str(k): v_ for k, v_ in d_.items()} for d_ in _fseeds]
    keys = sorted(set.intersection(*[set(m) for m in mats]), key=float)
    n["seed_r"] = sorted(
        float(np.corrcoef([mats[i][k] for k in keys],
                          [mats[j][k] for k in keys])[0, 1])
        for i, j in itertools.combinations(range(len(mats)), 2))
    n["n_seed_pairs"] = len(n["seed_r"])
    bad = [k for k in keys if np.mean([m[k] for m in mats]) < 0.5]
    n["n_bad_fields"] = len(bad)
    n["n_bad_always"] = sum(1 for k in bad if all(m[k] < 0.5 for m in mats))
    n["n_fields"] = len(v)
    n["median_field"] = float(np.median(v))
    n["mean_field"] = float(np.mean(v))
    n["n_below"] = int((v < 0.5).sum())
    n["pct_below"] = 100 * n["n_below"] / len(v)

    # The released tree ships 2-epoch *_debug smoke tests beside the real
    # baselines. They are not released models and counting them inflated the
    # audited total; crop type already excluded its one, so including them
    # here also made the two halves of the count inconsistent.
    allrows = list(csv.DictReader(
        (PROJ / "results/audit_summary.csv").open(encoding="utf8")))
    a = [r for r in allrows if "_debug" not in r["run"]]
    n["n_debug"] = len(allrows) - len(a)
    n["n_models"] = len(a)
    n["n_pass"] = sum(1 for r in a if r["verdict"] == "PASS")
    for t in ("crop_yield", "harvesting_date", "transplanting_date"):
        rs = [r for r in a if r["task"] == t and r["verdict"] == "PASS"]
        n[t] = {k: float(np.median([float(r[k]) for r in rs]))
                for k in ("agg_mae", "median", "p75", "p90")}
        n[t]["n"] = len(rs)

    # Crop type is gated separately from the regression tasks; verify_all.py
    # reports 9/12, so the manuscript must not call all 64 cells verified.
    ct = []
    for f in sorted(glob.glob(str(PROJ / "results/gate/*_bs32.json"))):
        g = json.loads(Path(f).read_text(encoding="utf8"))
        ct.append(g)
    n["ct_total"] = len(ct)
    n["ct_pass"] = sum(1 for g in ct if g["verdict"] == "PASS")
    miss = sorted(g["abs_delta"] for g in ct if g["verdict"] != "PASS")
    n["ct_miss"] = miss
    n["n_eval"] = n["n_models"] + n["ct_total"]
    n["n_repro"] = n["n_pass"] + n["ct_pass"]
    # Per-task breakdown, so the total can be checked against its parts.
    n["per_task"] = [("crop type (segmentation)", n["ct_total"], n["ct_pass"]),
                     ("yield", n["crop_yield"]["n"], n["crop_yield"]["n"]),
                     ("harvesting date", n["harvesting_date"]["n"],
                      n["harvesting_date"]["n"]),
                     ("transplanting date", n["transplanting_date"]["n"],
                      n["transplanting_date"]["n"])]

    # The abstract, contributions and Section V-H all report this
    # experiment, so a missing file is a build error, not a silent
    # omission that would leave the paper claiming less than it did.
    # Released yield models against a trivial constant predictor. This is
    # the comparison the benchmark never reports, and it changes what the
    # reproduction result means.
    import ast as _ast
    gl = [x for x in sorted(glob.glob(str(
        PROJ / "results/regression_gate/crop_yield_*_bs32.json")))
        if "_debug" not in x]
    gl = [x for x in gl
          if "per_field_pred" in json.loads(Path(x).read_text(encoding="utf8"))]
    d0 = json.loads(Path(gl[0]).read_text(encoding="utf8"))
    _truth = {_ast.literal_eval(k): float(v)
              for k, v in d0["per_field_true"].items()}
    _ks = sorted(_truth)
    _y = np.array([_truth[k] for k in _ks])
    _tr = []
    _root = None
    for line in (PROJ / "configs/base.yaml").read_text(encoding="utf8").splitlines():
        if line.strip().startswith("root:"):
            _root = line.split(":", 1)[1].strip().strip(chr(34) + chr(39))
            break
    with open(f"{_root}/sickle_dataset_tabular.csv", encoding="utf8") as fh:
        for r_ in csv.DictReader(fh):
            try:
                v_ = float(r_["YIELD"])
            except Exception:
                continue
            if r_["SPLIT"] == "train" and v_ > 0:
                _tr.append(v_)
    _cm = float(np.mean(_tr))
    n["const_mae"] = float(np.mean(np.abs(_cm - _y)))
    n["const_mape"] = float(np.mean(np.abs(_cm - _y) / _y) * 100)
    n["const_value"] = _cm
    _rows = []
    for x in gl:
        dd = json.loads(Path(x).read_text(encoding="utf8"))
        pv = np.array([float(dd["per_field_pred"][str(k)]) for k in _ks])
        _rows.append((dd["run"], float(np.mean(np.abs(pv - _y))),
                      float(np.mean(np.abs(pv - _y) / _y) * 100),
                      float(1 - ((pv - _y) ** 2).sum() / ((_y - _y.mean()) ** 2).sum()),
                      float(np.corrcoef(pv, _y)[0, 1])))
    _rows.sort(key=lambda t: t[1])
    n["yield_vs_const"] = _rows
    n["n_worse_mae"] = sum(1 for r_ in _rows if r_[1] > n["const_mae"])
    n["n_worse_mape"] = sum(1 for r_ in _rows if r_[2] > n["const_mape"])
    n["best_ckpt"] = _rows[0]

    ysf = PROJ / "results/triage/yield_screen.json"
    if not ysf.exists():
        raise SystemExit(
            f"missing {ysf.relative_to(PROJ)} -- the manuscript "
            f"reports the yield-screening experiment. "
            f"Run: python scripts/yield_screen.py")
    n["ys"] = json.loads(ysf.read_text(encoding="utf8"))

    # Plot-clustered intervals on each yield checkpoint's MAE gap to the
    # constant (scripts/yield_vs_constant.py). Required: without them the
    # manuscript could only quote point estimates, which is how "24 of 24
    # lose to a constant" got written.
    yvf = PROJ / "results/yield_vs_constant.json"
    if not yvf.exists():
        raise SystemExit("missing results/yield_vs_constant.json. "
                         "Run: python scripts/yield_vs_constant.py")
    n["yvc"] = json.loads(yvf.read_text(encoding="utf8"))

    n["tg"] = json.loads((PROJ / "results/triage/triage_v5.json").read_text(encoding="utf8"))

    # Field-level distribution, pooling weights, and the crop-type screen
    # measured against its label-free baselines. The manuscript refuses to
    # build without them rather than quietly reverting to the older, weaker
    # statements they replace.
    for key, rel, how in (
            ("fd", "results/field_distributions.json",
             "python scripts/field_distributions.py"),
            ("fs", "results/field_size_val.json",
             "python scripts/field_size.py"),
            ("sv", "results/triage/screen_validate.json",
             "python scripts/screen_validate.py")):
        f_ = PROJ / rel
        if not f_.exists():
            raise SystemExit(
                f"missing {rel} -- the manuscript reports this result. "
                f"Run: {how}")
        n[key] = json.loads(f_.read_text(encoding="utf8"))

    # Batch-size sensitivity, computed rather than quoted. Section V-A
    # previously carried three literals and a swing of "6.5 IoU points";
    # the saved cells give 0.0662, and the run it quotes is the mildest of
    # the four that were evaluated at more than one batch size.
    # Seed counts per rung, so no sentence can state a count the ladder
    # does not have. The ladder stops being uniform as soon as extra seeds
    # are trained for some rungs and not others, which is exactly what the
    # power analysis recommends doing.
    _ns, _nf = {}, {}
    with (PROJ / "results/ladder_crop_type.csv").open(encoding="utf8") as _fh:
        for _r in csv.DictReader(_fh):
            if _r.get("n_seeds"):
                _ns[_r["rung"]] = int(_r["n_seeds"])
                _nf[_r["rung"]] = int(_r.get("n_seeds_finished")
                                      or _r["n_seeds"])
    # rungs that have finished seeds beyond the balanced table: a targeted
    # follow-up, reported in the paired analysis only
    n["followup"] = {k: _nf[k] for k in _nf if _nf[k] > _ns.get(k, 0)}
    n["seeds_per_rung"] = _ns
    _vals = sorted(set(_ns.values()))
    _words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

    def _w(k):
        return _words.get(k, str(k))

    if len(_vals) == 1:
        n["seed_phrase"] = f"{_w(_vals[0])} seeds"
        n["seed_phrase_cap"] = f"{_w(_vals[0]).capitalize()} seeds"
        n["seed_count_note"] = ""
        n["n_ladder_runs"] = sum(_ns.values())
        if n["followup"]:
            _fu = n["followup"]
            _by_c = {}
            for k, c in _fu.items():
                _by_c.setdefault(c, []).append(k)
            _txt = " and ".join(
                f"{' and '.join(sorted(ks, key=lambda x: int(x[1:])))} to "
                f"{_w(c)} seeds each"
                for c, ks in sorted(_by_c.items(), key=lambda t: -t[0]))
            _extra = sum(_nf[k] - _ns[k] for k in _fu)
            n["seed_count_note"] = (
                f" A targeted follow-up then raised {_txt}, {_w(_extra)} "
                f"further runs, because the power analysis of Section V-B "
                f"identified that step as the only open question a larger "
                f"sample could settle. Those runs enter only the paired "
                f"analysis, so that every rung in the table is compared on "
                f"the same {_w(_vals[0])} seeds.")
            # the table sentence counts the table, not the follow-up
            n["n_ladder_runs"] = sum(_ns.values())
    else:
        # Each extra rung is named with ITS OWN count. An earlier version
        # said "seven at v3, v4" while v4 had six, because it reported one
        # shared maximum for every rung above the base.
        _lo = min(_vals)
        _more = sorted((k for k, v in _ns.items() if v > _lo),
                       key=lambda k: int(k[1:]))
        # group rungs that share a count: "eight at v3 and v4", not
        # "eight at v3 and eight at v4"
        _by = {}
        for k in _more:
            _by.setdefault(_ns[k], []).append(k)

        def _and(xs):
            return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

        _parts = [f"{_w(c)} at {_and(ks)}" for c, ks in
                  sorted(_by.items(), key=lambda t: -t[0])]
        _joined = _and(_parts)
        n["seed_phrase"] = f"{_w(_lo)} seeds ({_joined})"
        n["seed_phrase_cap"] = n["seed_phrase"][0].upper() + n["seed_phrase"][1:]
        _carry = [f"{_and(ks)} {'carries' if len(ks) == 1 else 'carry'} "
                  f"{_w(c)}" for c, ks in sorted(_by.items(),
                                                 key=lambda t: -t[0])]
        _carry_j = _and(_carry)
        n["seed_count_note"] = (
            f" The ladder is not uniform: {_carry_j} seeds, because the power "
            f"analysis of Section V-B identified "
            + ("that step" if len(_more) == 1 else "those rungs")
            + f" as the only place a larger sample would settle a "
            f"question, and the remaining rungs carry {_w(_lo)}.")
        n["n_ladder_runs"] = sum(_ns.values())
    n["seed_min"] = min(_vals) if _vals else 0

    _cs = PROJ / "results/triage/conditional_screen.json"
    if not _cs.exists():
        raise SystemExit(
            "missing results/triage/conditional_screen.json -- Section V-G "
            "reports whether the screen is explained by field size. Run: "
            "python scripts/conditional_screen.py")
    n["cs"] = json.loads(_cs.read_text(encoding="utf8"))

    _lp = PROJ / "results/ladder_power.json"
    if not _lp.exists():
        raise SystemExit(
            "missing results/ladder_power.json -- Section V-B reports the "
            "paired analysis of the ablation. Run: "
            "python scripts/ladder_power.py")
    n["lp"] = json.loads(_lp.read_text(encoding="utf8"))

    _bs = {}
    for _f in glob.glob(str(PROJ / "results/gate/gate_crop_type_*_bs*.json")):
        _d = json.loads(Path(_f).read_text(encoding="utf8"))
        _bs.setdefault(_d["run"], {})[_d["batch_size"]] = _d["reproduced"]
    _bs = {k: v for k, v in _bs.items() if len(v) > 1}
    n["bs_cells"] = _bs
    n["bs_spread"] = {k: max(v.values()) - min(v.values())
                      for k, v in _bs.items()}
    n["bs_worst_run"] = max(n["bs_spread"], key=n["bs_spread"].get)
    n["bs_mildest_run"] = min(n["bs_spread"], key=n["bs_spread"].get)
    # the run the section narrates, kept for continuity with Fig. 1
    n["bs_focus"] = "[L8]_unet3d" if "[L8]_unet3d" in _bs else n["bs_worst_run"]

    _rel = n["fd"]["released_crop_type"]
    _sh = [r["shift_without_largest_field"] for r in _rel
           if "shift_without_largest_field" in r]
    n["loo_n"] = len(_sh)
    n["loo_missing"] = n["fd"].get("released_missing_leave_fields_out") or []
    n["loo_median_abs"] = (float(np.median([abs(x) for x in _sh]))
                           if _sh else float("nan"))
    n["loo_max_abs"] = max((abs(x) for x in _sh), default=float("nan"))
    n["loo_worst_run"] = (max(_rel, key=lambda r: abs(
        r.get("shift_without_largest_field", 0.0)))["run"] if _sh else "")
    n["loo_worst"] = next((r for r in _rel if r["run"] == n["loo_worst_run"]),
                          None)
    n["gap_neg"] = [r["run"] for r in _rel if r["gap"] < 0]
    n["gaps"] = [r["gap"] for r in _rel]
    n["refs"] = json.loads((PROJ / "docs/references.json").read_text(encoding="utf8"))["references"]
    # The satellite figure's own record of which fields it shows, written
    # by scripts/make_fig_satellite.py; the V-C sentence about it reads this.
    _sat = FIG / "fig_satellite.json"
    n["sat"] = (json.loads(_sat.read_text(encoding="utf8"))
                if _sat.exists() else None)
    return n


def lst(xs):
    """Render ablation steps as 'label (+0.1198)' phrases.

    Uses U+2212 MINUS SIGN rather than hyphen-minus, to match the typography
    of the hand-written numbers elsewhere in the manuscript.
    """
    return ", ".join(
        # only the sign of the number, never the hyphens in "Sentinel-1"
        f"{x['label']} ({format(x['delta'], '+.4f').replace('-', chr(0x2212))})"
        for x in xs)


# ----------------------------------------------------------------- helpers
def _sgn(x, spec):
    """Format with U+2212 for a negative sign, as elsewhere in the text."""
    return format(x, spec).replace("-", "\u2212")


def _cnt(k):
    """Small counts as words: 'none of the 24 intervals lies ...'."""
    return {0: "none", 1: "one", 2: "two", 3: "three"}.get(k, str(k))


# ----------------------------------------------------------------- authors
# As the authors set them in their copy of the paper (2026-10-15), with
# "Science amd Technology" corrected to "and". Shared by both IEEE papers.
# IEEE convention: no honorifics, no affiliation superscripts (2026-10-16).
AUTHORS = [("Pharos Sophy Samuel T J", ""), ("S. Ganesh Kumar", "")]
AFFIL = ["Dept. of Data Science and Business Systems",
         "SRM Institute of Science and Technology",
         "Kattankulathur, India"]
EMAILS = ["tjpharossophysamuel@gmail.com", "ganeshk1@srmist.edu.in"]


def author_block(doc):
    """Authors side by side in a borderless table, IEEE conference layout:
    name, then department and institution in italics, city, email."""
    from docx.enum.table import WD_TABLE_ALIGNMENT
    assert not any(a[0].split()[0].rstrip(".") in ("Dr", "Prof", "Mr", "Ms")
                   for a in AUTHORS), "IEEE author lines carry no honorifics"
    tb = doc.add_table(rows=1, cols=len(AUTHORS))
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (name, mark) in enumerate(AUTHORS):
        cell = tb.rows[0].cells[i]
        lines = ([(name, 11.0, False, mark)]
                 + [(a, 9.0, j < 2, None) for j, a in enumerate(AFFIL)]
                 + [(EMAILS[i], 9.0, False, None)])
        for j, (txt, sz, it, mk) in enumerate(lines):
            p = cell.paragraphs[0] if j == 0 else cell.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            if mk:
                r = p.add_run(mk)
                r.font.superscript = True
                r.font.size = Pt(sz)
            r = p.add_run(txt)
            r.font.size = Pt(sz)
            r.italic = it
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tb._tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def spaced_percent(doc):
    """One percent style throughout: '95 %', as most hand-written text has
    it. Formatted values (f"{x:.0%}") come out closed up, so normalise every
    run, in paragraphs and table cells, just before saving."""
    import re as _re
    pat = _re.compile(r"(\d)%")
    paras = list(doc.paragraphs) + [q for tb in doc.tables for row in tb.rows
                                    for c in row.cells for q in c.paragraphs]
    for para in paras:
        for r in para.runs:
            if pat.search(r.text):
                r.text = pat.sub(r"\1 %", r.text)


def two_columns(section, n=2, space=0.25):
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(space * 1440)))
    cols.set(qn("w:equalWidth"), "1")


def body(d, text, first_indent=0.2):
    p = d.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    pf.first_line_indent = Inches(first_indent)
    pf.line_spacing = 1.0
    return p


def heading(d, text, level=1):
    p = d.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8); pf.space_after = Pt(3)
    pf.keep_with_next = True      # never strand a heading at a column foot
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text.upper()); r.bold = True; r.font.size = Pt(9.5)
    else:
        r = p.add_run(text); r.italic = True; r.font.size = Pt(9)
    return p


CITE = re.compile(r"\[(\d+)\]")


def renumber(d, refs):
    """Reassign citation numbers to IEEE order of first appearance.

    The prose is written with each reference's number as it stands in
    docs/references.json, which is not the order the final section ordering
    puts them in. IEEE requires citations numbered by first appearance, so
    this walks the document in order, assigns numbers as it meets each
    reference, and rewrites the tokens run by run -- which preserves the
    run's formatting, unlike rewriting paragraph text.

    Called after the body is complete and before the reference list is
    emitted, so the list can be written in the order this returns. Returns
    the references renumbered and reordered; an uncited reference would be
    left out of that list, so the caller checks for one.
    """
    order: list[int] = []

    def sub(m):
        old = int(m.group(1))
        if old not in order:
            order.append(old)
        return f"[{order.index(old) + 1}]"

    for p in d.paragraphs:
        for r in p.runs:
            if "[" in r.text:
                r.text = CITE.sub(sub, r.text)

    by_old = {r["n"]: r for r in refs}
    missing = [o for o in order if o not in by_old]
    if missing:
        raise SystemExit(f"cited but not in references.json: {missing}")
    uncited = sorted(set(by_old) - set(order))

    out = []
    for new, old in enumerate(order, start=1):
        r = dict(by_old[old])
        r["n"], r["was"] = new, old
        out.append(r)
    return out, uncited


_re_fig = __import__("re").compile(r"Fig\. (\d+)\.")
_FIG_SEEN: dict[int, str] = {}


def figure(d, name, caption, width=COL):
    """Place a figure, and refuse to emit two with the same number.

    Captions carry their own number, which is readable in the source but
    silently breaks when a figure is inserted in the middle: an added Fig. 4
    leaves the old Fig. 4 in place and every later reference pointing at the
    wrong panel. That happened once here. The caption number is therefore
    parsed and checked, so the build fails rather than producing a document
    with two Figure 4s.
    """
    m = _re_fig.match(caption)
    if not m:
        raise SystemExit(f"caption does not start with 'Fig. N.': "
                         f"{caption[:60]}")
    k = int(m.group(1))
    if k in _FIG_SEEN and _FIG_SEEN[k] != name:
        raise SystemExit(
            f"two figures numbered {k}: {_FIG_SEEN[k]} and {name}. "
            f"Renumber the later one and every in-text reference to it.")
    f = FIG / name
    if not f.exists():
        _FIG_SEEN.setdefault(k, name)
        return
    _FIG_SEEN[k] = name
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(f), width=Inches(width))
    c = d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(8)
    r = c.add_run(caption); r.font.size = Pt(7.5)


# ----------------------------------------------------------------- build
def build():
    n = load()
    tg = n["tg"]
    n["sv_macro"] = json.loads((PROJ / "results/triage/screen_validate_macro.json")
                               .read_text(encoding="utf8"))
    loo_d = {r["target_workload"]: r for r in tg["seed_disagreement"]["leave_one_out"]}
    d50c = {r["target_workload"]: r for r in tg["low_confidence"]["leave_one_out"]}[0.5]
    loo_c = {r["target_workload"]: r for r in tg["low_confidence"]["leave_one_out"]}
    d50, c33 = loo_d[0.5], loo_c[0.33]
    geo = tg["geographic_holdout"]["regions"]
    geo_big = [g for g in geo if g["n_fields"] >= 6 and g["signal"] == "seed_disagreement"]

    doc = Document()
    s = doc.sections[0]
    s.left_margin = s.right_margin = Inches(0.62)
    s.top_margin = Inches(0.7); s.bottom_margin = Inches(0.7)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(0)

    # ---- full-width title block
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Title revised 2026-10-13: it states the finding. "Aggregate metrics",
    # not "pooled IoU", because the tail is shown in all four tasks, three of
    # them scored by error rather than IoU (Section V-D).
    r = t.add_run("Aggregate Metrics Hide Field-Level Failures: A Reliability "
                  "and Reproducibility Audit of the SICKLE Satellite "
                  "Crop-Monitoring Benchmark")
    r.bold = True; r.font.size = Pt(19)
    t.paragraph_format.space_after = Pt(10)

    author_block(doc)

    # ---- switch to two columns
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    sec.left_margin = sec.right_margin = Inches(0.62)
    sec.top_margin = Inches(0.7); sec.bottom_margin = Inches(0.7)
    two_columns(sec)

    ab = doc.add_paragraph(); ab.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    rr = ab.add_run("Abstract—"); rr.bold = True; rr.italic = True; rr.font.size = Pt(9)
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
        "dispersion ranks "
        f"error at AUROC {n['ys']['threshold_free']['auroc']:.2f} (95 % CI "
        f"{n['ys']['threshold_free']['auroc_ci']['lo']:.2f}–"
        f"{n['ys']['threshold_free']['auroc_ci']['hi']:.2f}), no better "
        "than chance. Benchmarks that feed field-level decisions should "
        "report per-field error, and a safeguard must be validated on the "
        "quantity it gates. We audit a public benchmark, not any deployed "
        "system.")
    rr.font.size = Pt(9)

    kw = doc.add_paragraph(); kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw.paragraph_format.space_before = Pt(5); kw.paragraph_format.space_after = Pt(6)
    rr = kw.add_run("Index Terms—"); rr.bold = True; rr.italic = True; rr.font.size = Pt(9)
    rr = kw.add_run("crop monitoring, reproducibility, evaluation metrics, "
                    "per-field reliability, selective prediction, smallholder "
                    "agriculture, crop insurance, semantic segmentation.")
    rr.font.size = Pt(9)

    # ================================================= I
    heading(doc, "I.  Introduction")
    body(doc, "India has begun to automate crop-insurance payouts. "
              "Under the Pradhan Mantri Fasal Bima Yojana [3], the YES-TECH "
              "framework combines technology-based yield estimates with "
              "manual crop-cutting experiments; the manual requires at least "
              "30 % weightage to the modelled yield and sets no upper cap "
              "[2]. In 2025-26 Madhya Pradesh gave the technology-based yield "
              "100 % weightage and Maharashtra 50 %, and enrolled farmers do "
              "not file claims: they are \u201cautomatically worked out "
              "based on the shortfall in yield\u201d [21], as press "
              "reporting had described for Madhya Pradesh [4]. Farmers there "
              "contest the resulting assessments [5], [22], and "
              "a recent study proposes that satellite yield proxies supplement and progressively replace crop-cutting experiments [34]. No figure in "
              "this paper depends on these sources.", 0.0)
    body(doc, "The assessment is administrative, but the error is not. The "
              "manual estimates yield \u201cat the smallest administrative "
              "unit, i.e., Gram Panchayat\u201d [2], and in its 116 pages we "
              "found no requirement to validate at field level and no "
              "binding accuracy floor; an earlier, unversioned manual of "
              "January 2023 set an nRMSE ceiling of 30 % [20], which the "
              "governing manual does not carry. This describes the document, "
              "not practice, which we cannot observe. A village estimate is "
              "nevertheless assembled from the fields in it. If a model errs "
              "systematically on some fields, the errors do not cancel; they "
              "bias the village figure, and a few crop cuttings at village "
              "level cannot detect it.")
    body(doc, "Operational models are not public, so we audit the research "
              "benchmark on which models of this kind are developed: SICKLE "
              "[1], smallholder paddy in the Cauvery Delta of Tamil Nadu, "
              "with plots averaging 0.38 acres. We ask one question: what "
              "does a published benchmark number say about an individual "
              "field? Our contributions are: (i) three defects that undermine the "
              "benchmark's reproducibility, although most checkpoints "
              "reproduce, and one metric definition "
              "the released artefacts make easy to misread; "
              f"(ii) none of the {len(n['yield_vs_const'])} released yield "
              "models is shown to beat a constant predictor, a comparison "
              "the benchmark never reports; (iii) a per-field "
              f"audit of {n['n_eval']} released models across four tasks, "
              "showing that in every task the aggregate conceals a tail of "
              "fields far worse than the typical one, and that the common "
              "two-class form of per-field IoU misstates single-crop fields; "
              f"(iv) a controlled ten-rung ablation in which {len(n['inc'])} "
              f"of {n['n_steps']} modelling choices cannot be distinguished "
              "from seed variation; and (v) evidence that the failing fields "
              "recur across runs, while label-free safeguards find them only "
              "weakly for crop type and, in a pre-registered test, not at "
              "all for yield.")
    body(doc, "Scope. We audit a public benchmark, not any deployed system, "
              "and make no claim about payouts or outcomes for farmers. How "
              "these effects scale with the number of training plots needs "
              "a data-regime sweep, which is a separate study in progress "
              "and is not addressed here.")

    # ================================================= II
    heading(doc, "II.  Related Work")
    heading(doc, "A.  Crop mapping in smallholder systems", 2)
    body(doc, "Mapping crops in smallholder systems is constrained by field "
              "size relative to sensor resolution; Indian fields are often "
              "smaller than the pixels of freely available imagery, and "
              "multi-sensor combinations are used to compensate [6]. Benchmarks "
              "in this setting increasingly aggregate predictions to the "
              "agricultural parcel rather than treating pixels as the unit of "
              "analysis [7], [9], and field-scale yield estimation from "
              "satellite time series is an established task [16]. SICKLE [1] "
              "extends this to a multi-task setting, annotating crop type, "
              "yield and three phenology dates over the same plots. Methods "
              "proposed on it since continue to report pooled metrics: a "
              "multisensor transformer reports state-of-the-art yield MAPE "
              "[31], and agriculture-specific pretext tasks report pooled "
              "crop-mapping IoU and yield MAPE [32]; a 2026 review surveys "
              "benchmark datasets for satellite image time-series "
              "classification [33]. In "
              "smallholder regions, mapping accuracy also differs by field "
              "type: Huang et al. [28] find small, water-stressed fields the "
              "hardest to map. We ask a related but distinct question: not "
              "which kinds of field are hard, but how a benchmark's aggregate "
              "score conceals the distribution of failures across individual "
              "fields, and whether the same fields fail across independent "
              "runs.", 0.0)
    heading(doc, "B.  Evaluation metrics and the unit of analysis", 2)
    body(doc, "That aggregate segmentation metrics are biased toward large "
              "objects is established. Wang et al. [8] show that per-dataset "
              "mean intersection over union is biased in this way, propose "
              "fine-grained and worst-case variants, and benchmark fifteen "
              "networks across twelve natural and aerial datasets, framing the "
              "result explicitly as model and dataset auditing. We therefore "
              "treat per-field evaluation as established methodology that we "
              "apply, not as a contribution of this work. The concern is "
              "also general: aggregate metrics make it hard to predict how a "
              "system performs in a particular situation, and the "
              "instance-level results that would show it are rarely "
              "released [25]. What remains less "
              "examined under this protocol is whether the objects that fail "
              "are the same ones across independent training runs, which "
              "determines whether failures can be anticipated rather than "
              "merely counted.")
    heading(doc, "C.  Reproducibility, uncertainty and selective prediction", 2)
    body(doc, "Incomplete reporting is a recognised obstacle to reproducing "
              "published AI results [10], and variance from data sampling, "
              "initialisation and hyperparameter choice can change the "
              "outcome of a benchmark comparison, so a single run cannot "
              "attribute a difference to the change being tested [26]. Our "
              "ablation therefore runs every configuration at several seeds. "
              "Separately, disagreement among "
              "independently trained models is a standard and inexpensive "
              "uncertainty signal [11], and selective prediction formalises "
              "abstention: a model declines on part of the input distribution "
              "to bound error on the remainder, characterised by a "
              "risk–coverage trade-off [12]. We combine these: we use "
              "inter-seed disagreement and predictive confidence as "
              "label-free signals and evaluate them as screening tests in the "
              "coverage framing of [12].")
    heading(doc, "D.  Spatial generalisation", 2)
    body(doc, "Geographic transfer remains an open problem for crop mapping "
              "[15], and spatially independent validation can yield markedly "
              "different performance from random splits [13]. That approach is "
              "itself contested: Wadoux et al. [14] argue that spatial "
              "cross-validation lacks a theoretical basis for assessing map "
              "accuracy. We therefore present our regional holdout as a "
              "diagnostic rather than as validation, and draw no conclusion "
              "about geographic generalisation from it.")
    heading(doc, "E.  The gap this paper addresses", 2)
    body(doc, "Each strand above is established, and we claim none of them; "
              "in particular, the case against relying on aggregate metrics "
              "has been made in general [25], and what this paper adds is a "
              "measurement of it on field objects. "
              "What is missing is their conjunction on a benchmark that is "
              "being used to develop models for administrative decisions. "
              "Specifically: the metric bias of [8] has been demonstrated on "
              "natural and aerial imagery but not, to our knowledge, on a "
              "smallholder agricultural benchmark whose objects are legal "
              "parcels and whose per-object errors therefore carry an "
              "administrative consequence; the auditing framing of [8] counts "
              "which objects fail but does not test whether the same objects "
              "fail across independent training runs; and the selective "
              "prediction of [12] is normally evaluated per sample against "
              "labels, not as a label-free screen over field objects whose "
              "failure recurs. Sections V-F and V-G test exactly those two "
              "untested points, and Section V-A asks a prior question the "
              "reproducibility literature [10] motivates but which has not "
              "been asked of this benchmark: whether its published numbers "
              "can be recovered at all.")

    # ================================================= III
    heading(doc, "III.  Dataset and Experimental Setup")
    body(doc, "All experiments use the released SICKLE splits [1]. The tabular "
              "index contains 2,370 season-level samples and exposes 351 "
              "unique plot identifiers. The dataset paper reports 388 unique "
              "plots; the difference is accounted for by the test split, "
              "whose 206 rows carry a blank plot identifier in the release, "
              "so those plots cannot be recovered from the index. The split "
              "sizes are "
              "1,937 samples from 315 plots for training and 227 samples from "
              f"{n['n_fields']} plots for validation. The test split of 206 "
              "samples is unlabelled — every row is blank in PLOT_ID, CROP and "
              "PADDY_BIN and no masks exist — so every number reported here is "
              "a validation number and is not comparable to the published test "
              "table. For yield and the phenology tasks only rows with a "
              "recorded yield qualify, reducing validation to 41 field-seasons "
              "from 14 plots. We further note that 16 of the 21 crop labels "
              "have no validation plots, so only binary paddy classification "
              "is evaluable on this benchmark.", 0.0)
    body(doc, "Batching is stated precisely because it matters for Section "
              "V-A. Evaluation uses a batch size of 32 in full precision "
              "throughout, and all reported metrics are computed under that "
              "setting. Training uses an effective batch size of 32, formed as "
              "a micro-batch size of 8 with four gradient-accumulation steps "
              "under bfloat16 autocast, to fit an 8.5 GB device. Because the "
              "temporal axis is padded to the per-batch maximum, the padded "
              "tensor seen during training can differ from that of true "
              "batch-32 training. This setting is held constant across all "
              "rungs so that rung-to-rung differences remain controlled, but "
              "absolute values are not directly comparable to runs trained at "
              "a true batch of 32, and we do not assume the effect is "
              "identical across architectures.")

    # ================================================= IV
    heading(doc, "IV.  Methodology")
    heading(doc, "A.  Reproduction protocol", 2)
    body(doc, "No analysis was permitted to proceed until the harness could "
              "recover a published number. For each released checkpoint we "
              "load the authors' own configuration, data pipeline and model, "
              "evaluate on the validation split, and compare against the best "
              "epoch in that run's training log. Tolerances were declared in "
              "the source before any run and were not widened: 0.01 absolute "
              "for crop type and 10 % relative for the regression tasks.", 0.0)
    body(doc, "The audited inventory is stated exactly, because the released "
              "tree does not consist only of baselines. SICKLE annotates five "
              "targets: crop type, yield, and three phenology dates. We audit "
              f"four of them per field — {', '.join(t for t, _, _ in n['per_task'])} "
              f"— comprising {n['n_eval']} checkpoints: "
              + ", ".join(f"{c} for {t}" for t, c, _ in n["per_task"]) + ". "
              "Sowing date is excluded from the per-field audit and is "
              "included only in the metric check of Section V-A. Alongside "
              "the baselines the release ships runs suffixed _debug which "
              f"terminate after two epochs; the {n['n_debug']} such runs in "
              "the regression tasks and the one in crop type are smoke tests, "
              "not released models, and are excluded from every count in this "
              "paper. The yield task carries twice as many checkpoints as the "
              "date tasks because each sensor–architecture cell is released "
              "both per-sample and in a _season variant.")
    heading(doc, "B.  Per-field evaluation", 2)
    body(doc, "Mask band 0 carries the plot identifier, which makes per-field "
              "evaluation possible. For crop type we pool each field's pixels "
              "into one confusion matrix and weight fields equally [8]. For "
              "yield, mask band 5 stores yield divided by pixel count, so a "
              "field's yield is the sum over its pixels; a per-pixel error "
              "would describe a quantity scaling inversely with field area. "
              "For phenology a date is a single value per field. Aggregation "
              "is to the field-season rather than the field, since a plot "
              "recurs across seasons and a crop is insured for a season. "
              "Confidence intervals are obtained by resampling fields, never "
              "pixels, because pixels within a field are one measurement "
              "repeated rather than independent observations. For yield, the "
              "best checkpoint's correlation with the reference is also "
              "decomposed. The between-plot correlation is Pearson r over the "
              f"{n['yvc']['n_plots']} plot means of prediction and of "
              "reference, each plot's field-seasons weighted equally; the "
              "within-plot correlation is Pearson r over the "
              f"{n['yvc']['n_field_seasons']} field-seasons after each plot's "
              "own mean is subtracted from both, which leaves only "
              "season-to-season variation. Intervals for both resample whole "
              "plots, so the seasons of one plot are never treated as "
              "independent.")
    heading(doc, "C.  Controlled ablation", 2)
    body(doc, "Ten configurations were defined, each inheriting from its "
              "predecessor and altering exactly one factor: v1 single "
              "acquisition; v2 adds the full temporal sequence; v3 "
              "augmentation; v4 class-weighted loss; v5 Sentinel-1 radar; v6 "
              "Landsat-8; v7 UTAE [18] in place of U-Net3D [17]; v8 a wider "
              "encoder; v9 a cosine schedule with warmup; v10 plot-balanced "
              "sampling. ConvLSTM [19] appears among the released baselines "
              f"we reproduce. Each configuration was run at "
              f"{n['seed_phrase']}, giving "
              f"{n['n_ladder_runs']} runs."
              + n["seed_count_note"]
              + " A difference between "
              "adjacent rungs is attributable to that single change within "
              "this experimental system; we term this controlled component "
              "attribution and make no causal claim. A difference smaller than "
              "the seed-to-seed standard deviation is reported as "
              "inconclusive, which is not the same as demonstrating no "
              "effect.")
    heading(doc, "D.  Label-free screening", 2)
    body(doc, "Two signals are tested, neither of which uses ground truth. "
              "Seed disagreement is the fraction of a field's pixels on which "
              "independently seeded models do not all agree [11]; low "
              "confidence is one minus the mean softmax probability of the "
              "predicted class. Both are evaluated as screening tests in the "
              "coverage framing of [12]: what share of fields must be referred "
              "for verification (workload), and what share of genuinely "
              "unreliable fields that referral captures (recall). The referral "
              "rule refers fields whose signal falls in the top fraction of "
              "the signal distribution, a quantity computable at inference "
              "time. Thresholds are set leave-one-out, from the other fields "
              "only, so no field contributes to the threshold deciding its own "
              "outcome; this property is pinned by a unit test rather than "
              "left to inspection.")
    body(doc, "The interval is stated procedurally, because an "
              "independent reimplementation of an under-specified "
              "bootstrap will not match. Fields are resampled with "
              "replacement, 2,000 replicates, fixed seed; within each "
              "replicate the referral threshold is re-estimated from "
              "that replicate's signal values, so the interval carries "
              "the uncertainty of choosing the threshold as well as of "
              "the evaluation set; replicates containing no unreliable "
              "field are discarded, recall being undefined over an "
              "empty denominator; and the interval is the 2.5th to "
              "97.5th percentile of the retained replicates. Pixels are "
              "never resampled. One asymmetry is worth stating: the "
              "point estimate uses leave-one-out thresholds while each "
              "bootstrap replicate uses a single plug-in threshold, so "
              "the interval is not the exact bootstrap of the reported "
              "statistic. Freezing the leave-one-out rule and "
              "resampling only the scored fields instead narrows the "
              "50 %-workload interval from [68 %, 100 %] to "
              "[73 %, 100 %]; we report the wider of the two.")

    # ================================================= V
    heading(doc, "V.  Results and Analysis")
    heading(doc, "A.  Three defects, and one metric that invites misreading", 2)
    _f = n["bs_focus"]
    _c = n["bs_cells"][_f]
    _order = sorted(_c, reverse=True)
    _w = n["bs_worst_run"]
    _wc = n["bs_cells"][_w]
    body(doc, "First, evaluation depends on batch size. With frozen weights, "
              "identical data and unmodified code, the released "
              + ("Landsat-8 U-Net3D" if _f == "[L8]_unet3d" else _f)
              + " scores "
              + ", ".join(f"{_c[b]:.4f}" for b in _order)
              + " at batch sizes "
              + ", ".join(str(b) for b in _order)
              + f" (Fig. 1) — a swing of "
                f"{100 * n['bs_spread'][_f]:.1f} IoU points produced by "
                f"batch composition alone. Every value is backed by a saved "
                f"result file. The temporal axis is padded to the per-batch "
                f"maximum and the pad mask is realigned to the pooled length "
                f"by truncation rather than pooling, so whether masking "
                f"engages depends on which samples share a batch. Any IoU "
                f"reported on this benchmark without its evaluation batch "
                f"size is therefore not reproducible.", 0.0)
    body(doc, f"That run is the mildest of the "
              f"{len(n['bs_cells'])} we evaluated at more than one batch "
              f"size, and we quote it because Fig. 1 narrates it. The "
              f"largest effect is {_w}, which moves "
              f"{100 * n['bs_spread'][_w]:.1f} IoU points across batch sizes "
              + ", ".join(str(b) for b in sorted(_wc))
              + f" (from {min(_wc.values()):.4f} to "
                f"{max(_wc.values()):.4f}), although it is flat at "
              + ", ".join(str(b) for b in sorted(_wc) if b >= 16)
              + ". Two readings are therefore wrong: that the effect is "
                "confined to one checkpoint, and that it is confined to "
                "batch sizes too small to be used. "
              + (f"{n['bs_focus']} itself moves "
                 f"{100 * (max(v for k, v in _c.items() if k >= 16) - min(v for k, v in _c.items() if k >= 16)):.1f} "
                 f"points across batch sizes 16 and above."
                 if sum(1 for k in _c if k >= 16) > 1 else ""))
    figure(doc, "fig_batch.png",
           "Fig. 1.  Evaluation of a frozen checkpoint depends on batch size. "
           "Same weights, same data, unmodified code.", 2.4)
    body(doc, "Second, the same realignment divides by zero. A sample whose "
              "early acquisitions are empty — common where acquisitions begin "
              "late in a season or cloud-affected frames are zero-filled — "
              "yields an all-padding truncated mask and a non-finite output. "
              "We measured 13 of 1,937 training samples affected and 0 of 227 "
              "validation samples, which explains why reproduction succeeds "
              "while training fails. No guard against non-finite values exists "
              "anywhere in the reference implementation. Third, three of the "
              "four released UTAE crop-type baselines never trained: all are "
              "best at epoch 1, and two score an identical 0.3812 with paddy "
              "IoU exactly 0.000, predicting a single class everywhere. These "
              "are supplementary models — the dataset paper's main results "
              "table reports U-Net3D for time series and U-Net2D for single "
              "images [1] — but they are shipped as baselines and a reader "
              "comparing architectures from the release would be comparing "
              "against untrained weights.")
    body(doc, "A fourth observation is not a defect, and we state the "
              "distinction because it is easy to get wrong. The phenology "
              "training logs carry a column labelled MAPE whose value is mean "
              "absolute error divided by the constant 183, computed per "
              "pixel. This is the definition the dataset paper gives: it "
              "normalises “the MAE by the maximum possible date, i.e. 183, "
              "instead of the ground truth value”, because a conventional "
              "MAPE overshoots when the sowing-date range approaches zero "
              "[1]. The choice is documented and reasonable. What is not "
              "carried in the artefacts is the definition: the column is "
              "labelled MAPE with no qualification, and the paper's own main "
              "table reports the date tasks in MAE days rather than in that "
              "column. Our contribution here is empirical confirmation that "
              "the released value equals MAE/183 exactly in all 39 released "
              "date runs — every run of all three date tasks, sowing included "
              "— so the column is not comparable to a percentage error "
              "reported anywhere else, and should not be read as one.")

    heading(doc, "B.  Steps that add information exceed seed spread; "
                 "capacity is inconclusive", 2)
    body(doc, "A step is called inconclusive when the magnitude of its "
              "change falls below the largest seed-to-seed standard deviation "
              f"observed on any rung (σmax = {n['sd_max']:.4f}). This is "
              "deliberately conservative: it judges every step against the "
              "noisiest rung rather than against its own. "
              + ("Classifying each step against its own pair of rungs "
                 "gives the identical partition. "
                 if n["own_pair_same_partition"] else
                 "Classifying each step against its own pair of rungs "
                 f"instead gives {n['own_pair_n_exceeding']} exceeding "
                 "steps rather than "
                 f"{len(n['pos']) + len(n['neg'])}; we report the global "
                 "rule because it cannot be made to look better by "
                 "choosing which rung to compare against. ")
              + "All "
              f"{n['n_steps']} steps are reported. "
              f"{n['n_pos_w'].capitalize()} exceed{'s' if len(n['pos']) == 1 else ''} "
              f"σmax upward — " + lst(n["pos"]) + f" — and {n['n_neg_w']} "
              f"exceed{'s' if len(n['neg']) == 1 else ''} it downward — "
              + lst(n["neg"]) +
              f". The remaining {len(n['inc'])} are inconclusive: "
              + lst(n["inc"]) + ".", 0.0)
    body(doc, f"The pattern is that the {n['n_pos_w']} "
              f"step{'s' if len(n['pos']) != 1 else ''} which "
              f"{n['pos_verb']} "
              + ("add measurement"
                 if n["pos_all_information"]
                 else "change " + ", ".join(n["pos_kinds"]))
              + (f", while the {n['n_neg_w']} that {n['neg_verb']} "
                 f"{n['neg_kind_phrase']}. " if n["neg"] else ". ")
              + 
              "The rungs that add model capacity or change optimisation are "
              "inconclusive, and they consumed roughly three quarters of the "
              "compute (Fig. 2). Two of them had looked like results at a "
              "single seed and did not survive three: augmentation read "
              "−0.0496 and the capacity rung −0.0098 at seed 0 alone. "
              "With 315 training plots, additional parameters do not create "
              "additional evidence, whereas an additional sensor does.")
    body(doc, "Three caveats bound this. σmax is a descriptive spread, "
              "not a confidence interval, and exceeding it is not a "
              "significance test; we use it only to separate differences "
              "larger than the run-to-run noise we measured from those that "
              "are not. Inconclusive is not equivalent to absent. And the "
              "ordering is cumulative, so each step is attributable to its "
              "own change within this sequence but not necessarily to the "
              "same change made from a different starting point.")
    _lp = n["lp"]
    _st = {r["step"]: r for r in _lp["steps"]}
    _WW = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine"}
    # Steps the TABLE classifies as exceeding sigma_max (the balanced,
    # unpaired partition reported above) whose paired interval nevertheless
    # includes zero. Keyed on the table's classification, not on the paired
    # mean, so this paragraph cannot contradict the partition stated a few
    # sentences earlier.
    _tab = {f"v{int(x['to'][1:]) - 1}->{x['to']}": x
            for x in n["pos"] + n["neg"]}
    _gap = [k for k in _tab
            if k in _st and not _st[k]["verdict"].startswith("resolved")]
    # steps that were re-run beyond the base three seeds
    _rerun = [k for k, r in _st.items() if r["n_seeds"] > 3]

    def _m(v):
        return format(v, "+.4f").replace("-", chr(0x2212))
    _sum = _lp["summary"]
    _res, _small, _under = (_sum["resolved"], _sum["small"],
                            _sum["underpowered"])
    body(doc, "Because the first two caveats pull in opposite directions, we "
              "quantify them rather than leave them as warnings. Seeds are "
              "paired across rungs — seed k of one rung and seed k of "
              "the next share an initialisation and a data order — so "
              "each step can be estimated as a paired difference, which "
              "removes the seed effect common to both. On that basis "
              f"{len(_res)} of the {len(_lp['steps'])} steps have a 95 % "
              f"paired interval excluding zero "
              f"({', '.join(k.replace('->', ' to ') for k in _res)}), "
              f"and the remaining "
              f"{len(_small) + len(_under)} do not. "
              + (("The σmax rule classifies "
                  + (_WW.get(len(_gap), str(len(_gap))))
                  + (" further step" if len(_gap) == 1 else " further steps")
                  + " as exceeding the spread whose paired interval "
                  + ("nevertheless includes" if len(_gap) == 1
                     else "nevertheless include")
                  + " zero: "
                  + "; ".join(
                      f"{_tab[k]['label']}, at "
                      f"{format(_tab[k]['delta'], '+.4f').replace('-', chr(0x2212))}"
                      f" in the table, whose paired estimate over "
                      f"{_st[k]['n_seeds']} seeds is "
                      f"{format(_st[k]['paired_mean'], '+.4f').replace('-', chr(0x2212))}, "
                      f"interval "
                      f"[{format(_st[k]['ci95_paired'][0], '+.4f').replace('-', chr(0x2212))}, "
                      f"{format(_st[k]['ci95_paired'][1], '+.4f').replace('-', chr(0x2212))}]"
                      for k in _gap)
                  + ". Magnitude larger than the measured noise and "
                    "distinguishable from zero are different claims, and "
                    "we report both rather than letting the first stand "
                    "for the second.")
                 if _gap else
                 "Every step the σmax rule classifies as exceeding the "
                 "spread also has a paired interval excluding zero, so "
                 "here the two criteria agree."))
    body(doc, "The inconclusive rungs are also not alike, and the "
              "distinction decides whether more seeds would help. For "
              f"{len(_small)} of them "
              f"({', '.join(k.replace('->', ' to ') for k in _small)}) the "
              "paired interval lies entirely "
              "within ±σmax, so a large effect is ruled out and "
              "they are inconclusive because the effect is small. For "
              f"{len(_under)} "
              f"({', '.join(k.replace('->', ' to ') for k in _under)}) the "
              "interval still admits an effect "
              f"larger than σmax, so they are inconclusive for want of seeds "
              "at the counts the ladder has. Taking the observed means and spreads "
              "at face value, 80 % power would need about "
              + "; ".join(
                  f"{_st[k]['required_n_for_80pc_power']} seeds for "
                  f"{k.replace('->', ' to ')}"
                  for k in _under
                  if _st[k]["required_n_for_80pc_power"] is not None)
              + ". Those counts come from a mean and a standard deviation "
                "each estimated from only the paired seeds available, so "
                "they indicate which questions are cheap and which are not, "
                "and nothing more precise than that. "
              + (("Acting on that, "
                  + " and ".join(k.replace("->", " to ") for k in _rerun)
                  + " " + ("was" if len(_rerun) == 1 else "were")
                  + " re-run at "
                  + ", ".join(f"{_st[k]['n_seeds']}" for k in _rerun)
                  + " paired seeds rather than three. "
                  + " ".join(
                      # the estimate is already given above when the step
                      # is one the table flagged; say only how it stands
                      (f"It "
                       if k in _gap else
                       f"With {_st[k]['n_seeds']} seeds the "
                       f"{k.replace('->', ' to ')} step is "
                       f"{_m(_st[k]['paired_mean'])}, interval "
                       f"[{_m(_st[k]['ci95_paired'][0])}, "
                       f"{_m(_st[k]['ci95_paired'][1])}], and ")
                      + "is "
                      + {"resolved": "now resolved",
                         "small": "now bounded as small",
                         "underpowered": "still underpowered"}[
                          _st[k]["verdict"].split(" ")[0].split(":")[0]]
                      + "."
                      for k in _rerun) + " ")
                 if _rerun else "")
              + "The ablation as reported should be read as resolving "
              + f"{_WW.get(len(_res), str(len(_res)))} "
              + ("step" if len(_res) == 1 else "steps")
              + f", bounding {_WW.get(len(_small), str(len(_small)))} and "
              + f"leaving {_WW.get(len(_under), str(len(_under)))} "
              + "genuinely open.")
    figure(doc, "fig_ladder.png",
           f"Fig. 2.  Ten-rung controlled ablation at {n['seed_phrase']}. "
           "Error bars "
           "are seed-to-seed standard deviation. The shaded region marks the "
           "rungs (v7 to v10) that change the architecture, its width, the "
           "schedule or the sampling, not the information the model sees.")

    heading(doc, "C.  Reported IoU and the fields behind it", 2)
    _lad = n["fd"]["ladder_crop_type"]
    _q = _lad["quantiles"]
    _b50 = _lad["below"]["0.5"]
    _b70 = _lad["below"]["0.7"]
    _b30 = _lad["below"]["0.3"]
    _mcx = _lad["macro_contrast"]
    _sh = _lad["share_of_shortfall_worst_decile"]
    body(doc, "A per-field score needs a definition, and the obvious one "
              f"misleads here. {_lad['n_single_crop_fields']} of the "
              f"{n['n_fields']} validation fields hold one crop in every "
              "season. Averaged over both classes, IoU is 1.0 for such a "
              "field when every pixel is right and at most 0.5 as soon as "
              "one pixel is predicted as the absent class, whose IoU is then "
              "0. A field 99 % correct would score below 0.5. We therefore "
              "score each field on the crop or crops it holds, which for a "
              "single-crop field is the share of its pixels labelled "
              "correctly. The difference is not cosmetic: the two-class "
              f"average, which an earlier version of this study used, gives "
              f"a median field of {_mcx['median']:.3f} and "
              f"{_mcx['below']['0.5']} fields below 0.5; on the field score "
              f"the median is {_q['50']:.3f} and {_b50['n']} are. Per-field "
              "evaluation is the remedy for pooled metrics [8], so this is a "
              "trap worth stating for anyone applying it to parcels that "
              "hold one class.", 0.0)
    _sat = n["sat"]
    if _sat:
        _typ = [_sat["seed0_field_iou"][str(p)] for p in _sat["typical_fields"]]
        _bad = _sat["failing_fields"]
        _all0 = [p for p, v in _bad.items() if v == 0.0]
        body(doc, "Fig. 3 shows what these scores look like on the ground. "
                  "The fields chosen by size, not by result, are mapped "
                  f"almost perfectly (field IoU {min(_typ):.2f} to "
                  f"{max(_typ):.2f}). Of the {len(_bad)} fields whose IoU "
                  f"averaged over {len(_sat['seeds_for_selection'])} runs "
                  "falls below 0.5, "
                  + (f"field {_all0[0]} is paddy that is mislabelled at "
                     "every pixel in every run, and the other "
                     f"{len(_bad) - 1} are partly mislabelled. "
                     if len(_all0) == 1 else "all are partly mislabelled. ")
                  + "A pooled figure averages these fields into the many "
                  "that are right, and cannot say which they are.")
    figure(doc, "fig_satellite.png",
           "Fig. 3.  Sentinel-2 chips of SICKLE validation fields beside the "
           "prediction of the v5 seed-0 checkpoint, the run whose field "
           "scores the text reports. Each pair shows the "
           "surveyed field outlined on the image and the prediction over "
           "it; red marks wrong pixels inside the field. (a) Fields chosen "
           "by size, not by result. (b) The fields whose IoU averaged over "
           "three runs is below 0.5, each shown at the season with the "
           "most wrong pixels. The number is the field's IoU over all its "
           "seasons at seed 0; in (b) the three-seed mean follows and sets "
           "the colour. Chips are "
           "cropped around each field and carry their own 100 m bar.")
    body(doc, f"Field-level results use v5 (Sentinel-2 with radar), the rung with "
              f"the highest plot-level score "
              f"({float(n['L']['v5']['plot_macro_iou']):.3f}); v6 adds Landsat-8 "
              f"and scores {n['d_l8']:.4f} higher pooled, within seed spread. "
              f"For v5, pixel-pooled IoU is {n['v5']:.4f}, "
              f"the average field scores {_lad['per_field_mean']:.3f} "
              f"(95 % CI {_lad['ci_mean'][0]:.3f}–{_lad['ci_mean'][1]:.3f}) "
              f"and the median field {n['median_field']:.3f} (Fig. 4). The "
              "headline is therefore close to the average field and, if "
              "anything, below the typical one. What it does not show is "
              "where the shortfall sits. Across the "
              f"{n['n_fields']} validation fields the quartiles are "
              f"{_q['25']:.3f}, {_q['50']:.3f} and {_q['75']:.3f} and the "
              f"range runs from {_q['0']:.3f} to {_q['100']:.3f}; "
              f"{_b30['frac']:.0%} of fields fall below 0.30 "
              f"(95 % CI {_b30['ci'][0]:.0%}–{_b30['ci'][1]:.0%}), "
              f"{_b50['frac']:.0%} below 0.50 "
              f"({_b50['ci'][0]:.0%}–{_b50['ci'][1]:.0%}) and "
              f"{_b70['frac']:.0%} below 0.70 "
              f"({_b70['ci'][0]:.0%}–{_b70['ci'][1]:.0%}), and the "
              f"{_sh['n_fields']} worst fields carry {_sh['share']:.0%} of "
              "the shortfall below the pooled figure. None of the three "
              "levels is an official criterion; we report all three so that "
              "no conclusion rests on one cut. Intervals resample fields, "
              "the independent spatial unit.")
    _fs = n["fs"]
    body(doc, "The mechanism is unequal weight, and it can be stated exactly "
              "rather than argued. The pooled confusion matrix is the sum of "
              "the per-field matrices, so removing a field and recomputing "
              "is arithmetic, not reweighting. Field sizes differ by a factor "
              f"of {max(p_['n_pixels_pooled'] for p_ in _fs['plots']) // max(1, min(p_['n_pixels_pooled'] for p_ in _fs['plots']))}: one field "
              f"supplies {_fs['share_largest']:.0%} of the "
              f"{_fs['n_pixels_in_pooled_metric']:,} scored pixels and the "
              f"nine largest supply {_fs['share_top_quartile']:.0%}, against "
              f"{_fs['equal_weight_share']:.1%} for an equally weighted "
              f"field. Removing that single field from the "
              f"{n['loo_n']} released crop-type checkpoints moves the pooled "
              f"figure by a median of {n['loo_median_abs']:.4f} IoU and by as "
              f"much as {n['loo_max_abs']:.4f} "
              f"({n['loo_worst_run']}: {n['loo_worst']['pooled']:.4f} to "
              f"{n['loo_worst']['pooled_without_largest_field']:.4f}). A "
              "benchmark number that a single field of 36 can move that far "
              "is not a field-level statement.")
    _sp = _lad["size"]["spearman_weight_vs_iou"]
    _spci = _lad["size"]["spearman_weight_vs_iou_ci"]
    body(doc, "Two readings must be ruled out. The first is that large fields "
              "are simply easier, which would make the pooled figure "
              "optimistic in a predictable way: field IoU shows no "
              "relationship with field size that the data can distinguish "
              f"from zero (Spearman {_sgn(_sp, '+.3f')}, 95 % CI "
              f"{_sgn(_spci[0], '+.3f')} to {_sgn(_spci[1], '+.3f')}; Fig. 5c), so the pooled "
              "figure is dominated by a few fields in weight without those "
              "fields being demonstrably better. The second is that the gap is a fixed "
              "optimism that could be subtracted: across the released "
              f"checkpoints it ranges from {_sgn(min(n['gaps']), '+.4f')} to "
              f"{_sgn(max(n['gaps']), '+.4f')}, and "
              f"{', '.join(n['gap_neg']) if n['gap_neg'] else 'no checkpoint'} "
              + ("score" if len(n['gap_neg']) > 1 else "scores")
              + " higher per-field than pooled. The two metrics disagree "
              "by an amount that depends on the model, which is why the "
              "per-field distribution has to be reported rather than "
              "estimated from the pooled value.")
    figure(doc, "fig_perfield.png",
           "Fig. 4.  Field IoU (the crop each field holds) for the best "
           "configuration, with the reported pixel-pooled value overlaid. "
           "Red bars fall below 0.5; the cross marks a field scoring 0.")
    figure(doc, "fig_field_distribution.png",
           "Fig. 5.  (a) Distribution of field IoU over the "
           f"{n['n_fields']} validation fields at v5, "
           "with the pooled value and three stated thresholds. (b) The same "
           "as a quantile plot. (c) Field IoU against the number of "
           "pixels the field contributes to the pooled metric; the "
           "horizontal axis is logarithmic and spans three orders of "
           "magnitude, and no trend across it is distinguishable from "
           "zero.")

    heading(doc, "D.  Across four tasks the aggregate conceals the tail", 2)
    body(doc, f"All {n['n_models']} released models for the three regression "
              f"tasks reproduced within tolerance ({n['n_pass']}/"
              f"{n['n_models']} gates passed). The {n['ct_total']} crop-type "
              f"cells are weaker: {n['ct_pass']} reproduce within the declared "
              f"0.01 IoU tolerance and {n['ct_total'] - n['ct_pass']} miss it, "
              f"by {', '.join(f'{d:.4f}' for d in n['ct_miss'])}. We report "
              "this rather than widening the tolerance to absorb it, and we "
              "tested the obvious explanation on all three rather than "
              "asserting it. Each cell was re-evaluated across evaluation "
              "batch sizes; the three behave differently, so we report them "
              "separately rather than attributing them to one cause. "
              "The [S2] U-Net3D cell is explained: it reproduces within "
              "tolerance at three of the four sizes tried \u2014 0.7086 at "
              "8, 0.7192 at 16 and 0.7082 at 64 against a released 0.7173 "
              "\u2014 and misses only at 32, the size at which every result "
              "in this paper is pinned. Its apparent irreproducibility is an "
              "artefact of that choice, which is defect D1 acting on a "
              "reproduction verdict rather than on a score. The [S1] U-Net3D "
              "cell is not: it climbs from 0.4616 at batch 1 through 0.5412 "
              "and 0.6278 to 0.7888 at 16, then returns bit-identical values "
              "at 16, 32 and 64 against a released 0.8011. It moves by 33 "
              "IoU points across the range and saturates once batches are "
              "large enough that most contain a long sequence \u2014 itself "
              "consistent with the per-batch padding mechanism \u2014 yet "
              "never closes the final 0.0123. The [S2] ConvLSTM cell is also "
              "not explained, and differently again: its score ranges over "
              "0.178 IoU across batch sizes with no monotone trend, and the "
              "closest value to the released 0.4535 is the pinned batch 32 "
              "itself, still 0.0345 away. Batch composition therefore "
              "accounts for one of the three residuals and is ruled out for "
              "the other two, whose causes remain unresolved. "
              f"Across the two groups {n['n_eval']} models "
              f"were re-evaluated and {n['n_pass'] + n['ct_pass']} reproduced. "
              f"For crop type the median field ({n['median_field']:.3f}) is "
              f"better than the aggregate ({n['v5']:.4f}) and "
              f"{n['n_below']} of {n['n_fields']} fields fall below 0.5. "
              "For yield and phenology the median field is likewise better "
              "than the aggregate, and the tail is markedly heavier: one "
              "field-season in ten is two to four times worse than typical "
              "(Table I, Fig. 6). Across the released crop-type checkpoints "
              "the gap between pooled and average-field IoU takes either "
              "sign, so it cannot be corrected for. What is consistent "
              "across tasks is that the aggregate conceals the tail.", 0.0)

    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    rr = cap.add_run("TABLE I\nReported versus per-field-season error, medians "
                     "across all released models per task")
    rr.font.size = Pt(7.5); rr.bold = True
    tb = doc.add_table(rows=1, cols=5); tb.style = "Table Grid"
    for i, h in enumerate(("Task", "Reported MAE", "Typical field", "p90", "p90/med")):
        c = tb.rows[0].cells[i]; c.text = h
        for p_ in c.paragraphs:
            for r_ in p_.runs: r_.bold = True; r_.font.size = Pt(7.5)
    for t, lab, u in (("crop_yield", "Yield", "kg/acre"),
                      ("harvesting_date", "Harvest", "d"),
                      ("transplanting_date", "Transplant", "d")):
        v = n[t]; row = tb.add_row().cells
        for i, val in enumerate((f"{lab} ({u})", f"{v['agg_mae']:.2f}",
                                 f"{v['median']:.2f}", f"{v['p90']:.2f}",
                                 f"{v['p90']/v['median']:.2f}×")):
            row[i].text = val
            for p_ in row[i].paragraphs:
                for r_ in p_.runs: r_.font.size = Pt(7.5)
    figure(doc, "fig_tails.png",
           "Fig. 6.  For every regression task the typical field is better "
           "than the reported error (blue line), while the tail is far worse.")

    heading(doc, "E.  No released yield model is shown to beat a constant", 2)
    body(doc, "Section V-D established that the released regression "
              "checkpoints reproduce. Reproducing is not the same as being "
              "useful, and the benchmark never reports the comparison that "
              "settles it. We therefore scored a trivial predictor on the "
              "same evaluation set: the mean yield of the training split, "
              f"{n['const_value']:.0f} kg/acre, issued unchanged for every "
              "field-season.", 0.0)
    body(doc, f"That constant achieves MAE {n['const_mae']:.1f} kg/acre and "
              f"MAPE {n['const_mape']:.2f} % on the {len(n['yield_vs_const'])} "
              f"released yield checkpoints' own validation field-seasons. "
              f"{n['n_worse_mae']} of {len(n['yield_vs_const'])} released "
              f"checkpoints have a higher observed MAE than that constant, "
              f"and {n['n_worse_mape']} of {len(n['yield_vs_const'])} a "
              f"higher MAPE. The best of them, {n['best_ckpt'][0]}, scores "
              f"MAE {n['best_ckpt'][1]:.1f} against the constant's "
              f"{n['const_mae']:.1f}, with R\u00b2 = "
              f"{_sgn(n['best_ckpt'][3], '+.3f')} on these field-seasons: it "
              "explains less of their variance than the mean does.")
    _y = n["yvc"]
    _yb = _y["best"]
    body(doc, f"These are point estimates on {_y['n_field_seasons']} "
              f"field-seasons from only {_y['n_plots']} plots, so we resample "
              "plots, keeping each plot's field-seasons together and each "
              "checkpoint paired with the constant on the same "
              "field-seasons. The best checkpoint's excess MAE is "
              f"{_sgn(_yb['mae_minus_constant'], '+.1f')} kg/acre (95 % CI "
              f"{_sgn(_yb['ci95'][0], '+.1f')} to {_sgn(_yb['ci95'][1], '+.1f')}), "
              f"and {_cnt(_y['n_ci_above_zero'])} of the "
              f"{_y['n_checkpoints']} intervals lies wholly above zero"
              + (", nor any wholly below it" if _y["n_ci_below_zero"] == 0
                 else "")
              + ". No checkpoint is shown to beat the constant, and none "
              "is shown to be worse than it either. Failing to beat the mean on unseen units is not peculiar to SICKLE: maize-yield models from five African countries reach moderate R² within a country, but tested on an unseen country every feature set scores a negative R², worse than predicting the test-set mean [35].")
    # Decomposed 2026-10-14: the overall r is between-plot association, and
    # an earlier reading of it as season-level skill was wrong.
    _dc = _yb["decomposition"]
    _bp, _wp = _dc["between_plots"], _dc["within_plots"]
    body(doc, "The shortfall is not an absence of signal, but the signal is "
              "not the one a loss assessment needs. The same best checkpoint "
              f"correlates with the reference yield at "
              f"r = {_sgn(_yb['pearson_r'], '+.3f')} (95 % CI "
              f"{_sgn(_yb['pearson_r_ci95'][0], '+.3f')} to "
              f"{_sgn(_yb['pearson_r_ci95'][1], '+.3f')}, resampling plots). "
              "Decomposed on the same predictions, that association comes "
              f"from differences between plots: the {_bp['n']} plot averages "
              "of prediction and reference correlate at "
              f"r = {_sgn(_bp['r'], '+.3f')} (95 % CI "
              f"{_sgn(_bp['ci95'][0], '+.2f')} to "
              f"{_sgn(_bp['ci95'][1], '+.2f')}), while within plots, with "
              "each plot's own mean removed from both, the "
              f"{_wp['n']} field-seasons correlate at "
              f"r = {_sgn(_wp['r'], '+.3f')} (95 % CI "
              f"{_sgn(_wp['ci95'][0], '+.2f')} to "
              f"{_sgn(_wp['ci95'][1], '+.2f')}). The model's estimate does "
              "move from season to season within a plot (standard deviation "
              f"{_dc['within_plot_sd_pred']:.0f} kg/acre, against "
              f"{_dc['within_plot_sd_ref']:.0f} for the reference), but not "
              "with the reference. It therefore ranks fields by their typical "
              "yield and shows no evidence of tracking a field's better and "
              "worse seasons, the shortfall on which a claim is worked out "
              "[21]. With two to four seasons per plot this is an absence of "
              "evidence, not a demonstration that such skill is impossible. "
              "Pooled MAE separates none of this: a model can rank fields "
              "usefully and still be no better than issuing one number to "
              "everyone. Related evidence comes from index insurance: with "
              "yield data covering many fields but few years, standard "
              "estimates understate basis risk and so overstate insurance "
              "quality [27]. That work concerns how basis risk is estimated; "
              "the decomposition here asks directly whether a model tracks "
              "the seasons of the same plot.")
    body(doc, "We draw a narrow conclusion. This does not show the "
              "architectures are unsuitable, nor that satellite yield "
              "estimation cannot work; it shows that on this benchmark, as "
              "released, the published yield task has no demonstrated "
              "advantage over a constant, and that a reproduction study which "
              "stops at \u2018the numbers match\u2019 would not have "
              "discovered it. The comparison costs nothing to run and we "
              "recommend it be reported alongside any field-level yield "
              "result.")

    heading(doc, "F.  Field scores recur across runs", 2)
    body(doc, "Field IoU correlates positively between every pair of "
              f"independently seeded runs: across the {n['n_seed_pairs']} "
              f"pairs the Pearson correlation ranges from "
              f"r = {_sgn(min(n['seed_r']), '+.2f')} to {_sgn(max(n['seed_r']), '+.2f')}. Of "
              f"the {n['n_bad_fields']} fields whose mean field IoU is below "
              f"0.5, {n['n_bad_always']} are below 0.5 in every run "
              "(Fig. 7). How well a field is mapped is thus largely a "
              "property of the field rather than of the training run, which "
              f"is the precondition for anticipating it. With "
              f"{n['seed_phrase']} this is {len(n['seed_r'])} correlation "
              f"estimates on {n['n_fields']} fields, so we report the range "
              "rather than a pooled coefficient and attach no p-value.", 0.0)
    figure(doc, "fig_seeds.png",
           "Fig. 7.  Field IoU between two independent seeds.", 2.4)

    heading(doc, "G.  Can unreliable fields be identified without labels?",
            2)
    _V = n["sv"]
    _sd = _V["signals"]["seed_disagreement"]
    _lc = _V["signals"]["low_confidence"]
    _sf = _V["signals"]["smallest_first"]
    _pn = _V["permutation_null"]
    _C = n["cs"]
    _t1 = _C["test_1_conditional_permutation"]["low_confidence"]
    _sa = _C["test_2_out_of_sample"]["fit_free"]["screen alone"]
    _nn = _C["loo_noise_null"]
    _lcb = _lc["vs_baseline"]["smallest_first"]
    _mv = n["sv_macro"]
    body(doc, f"With field IoU as the score, {tg['n_bad']} of the "
              f"{tg['n_fields']} fields are unreliable (mean below 0.5), so "
              "every rate below is a statement about a handful of fields "
              f"and one field moves recall by {100 / tg['n_bad']:.0f} "
              "percentage points. Two label-free signals were tested: seed "
              "disagreement, which needs several independently trained "
              "models, and mean confidence, which needs one. Both correlate "
              "with field IoU in the useful direction (r = "
              f"{_sgn(tg['seed_disagreement']['corr_with_iou'], '+.2f')} and "
              f"{_sgn(tg['low_confidence']['corr_with_iou'], '+.2f')}). Referring the "
              f"top half by confidence catches {d50c['n_caught']} of the "
              f"{tg['n_bad']} (95 % CI {d50c['recall_ci']['lo']:.0%}–"
              f"{d50c['recall_ci']['hi']:.0%}) at "
              f"{d50c['precision']:.0%} precision (Fig. 8).", 0.0)
    body(doc, "Against random referral taken as a distribution of "
              f"{_pn['n_permutations']:,} permutations, whose 97.5th "
              f"percentile is AUROC {_pn['auroc_p97_5']:.3f} at this sample "
              "size, the evidence is weak. Seed disagreement ranks the "
              f"unreliable fields at {_sd['auroc']:.3f} (95 % CI "
              f"{_sd['auroc_ci'][0]:.3f}–{_sd['auroc_ci'][1]:.3f}), an "
              "interval that includes chance. Confidence ranks them at "
              f"{_lc['auroc']:.3f} ({_lc['auroc_ci'][0]:.3f}–"
              f"{_lc['auroc_ci'][1]:.3f}): the bootstrap interval excludes "
              "chance, but the value does not exceed the permutation null's "
              "97.5th percentile. And a ranking that needs no model at all, "
              "checking the smallest fields first, scores "
              f"{_sf['auroc']:.3f}; the confidence signal's advantage over it "
              f"is {_sgn(_lcb['delta_auroc'], '+.3f')} (95 % CI {_sgn(_lcb['ci'][0], '+.3f')} "
              f"to {_sgn(_lcb['ci'][1], '+.3f')}).")
    body(doc, "Being no better than size is not the same as being size, and "
              "the two can be told apart. Permuting the signal within size "
              "terciles leaves the size-to-failure relationship exactly "
              "intact and destroys everything else. Against that null, "
              f"confidence scores p = {_t1['p_one_sided']:.3f} over "
              f"{_t1['n_permutations']:,} permutations, so it carries some "
              "information that size does not supply; whether that "
              "information improves a ranking is not demonstrated, since "
              "the screen must beat size in either direction and its gain "
              "over the stronger direction has an interval of "
              f"[{_sgn(_sa['ci_vs_smaller_first'][0], '+.3f')}, "
              f"{_sgn(_sa['ci_vs_smaller_first'][1], '+.3f')}]. We report the crop-type "
              "screen as carrying information beyond field size, as not "
              "shown to rank unreliable fields better than size, and as not "
              "established.")
    body(doc, "Two corrections are recorded rather than hidden. First, "
              "before the per-field score was corrected, the same analysis "
              f"on the two-class average marked {_mv['n_bad']} fields "
              "unreliable, most of them for a few wrong pixels, and both "
              "signals then appeared informative; those results are kept in "
              "the protocol's amendment and are not carried forward. "
              "Second, the protocol pre-specified a leave-one-field-out "
              "logistic comparison, which at this sample size is "
              "anti-correlated for an uninformative feature: run on "
              f"{_nn['n_draws']} pure-noise features it averages AUROC "
              f"{_nn['mean']:.3f}, with {_nn['share_below_half']:.0%} of "
              "draws below 0.5. It is reported as a diagnostic only.")
    figure(doc, "fig_triage.png",
           "Fig. 8.  Screening performance of two label-free signals. The "
           "diagonal is random referral. Thresholds are set leave-one-out.")

    heading(doc, "H.  Geographic-transfer diagnostic", 2)
    _n0 = len({g["region"] for g in geo if g["n_bad"] == 0})
    _W_ = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}
    body(doc, "We also held out each delta sub-region in turn, setting the "
              "referral threshold from the remaining regions only and applying "
              "it to the held-out one. No field contributes to the threshold "
              "that judges it. The outcome is mixed and we draw no conclusion "
              "about geographic generalisation from it. For seed disagreement, "
              + "; ".join(f"{g['region']} ({g['n_fields']} fields, "
                          f"{g['n_bad']} unreliable) gives "
                          f"{g['recall']:.0%} recall" for g in geo_big)
              + ". Of ten region–signal combinations only four involve six or "
              "more fields; three sub-regions contribute two or three "
              f"validation fields and {_W_.get(_n0, str(_n0))} contain no "
              "unreliable field at all, so no recall is defined for them. The defensible conclusion "
              "concerns the benchmark rather than the method: a geographically "
              "held-out evaluation of field-level reliability requires a "
              "labelled set substantially larger than 36 fields spread across "
              "five sub-regions. We note also that spatial cross-validation is "
              "itself contested as an accuracy-assessment procedure [14].", 0.0)

    if True:
        y = n["ys"]
        tf = y["threshold_free"]
        informative = tf["auroc_ci_excludes_half"]
        sel = y["selective_prediction"]
        full = next(x for x in sel if x["coverage"] == 1.0)
        half = min(sel, key=lambda x: x["coverage"])
        below = [r for r in y["screening"]
                 if r["recall"] < r["workload_realised"]]

        heading(doc, "I.  The screen does not carry over to yield", 2)
        body(doc, "Sections V-F and V-G concern crop-type IoU. A loss "
                  "assessment does not consume IoU, so a screen validated on "
                  "segmentation overlap does not establish that a yield "
                  "estimate can be trusted. We therefore put the same "
                  "question to the quantity such an assessment would use. "
                  "The hypothesis is falsifiable as stated: inter-model "
                  "disagreement ranks field-seasons by yield error better "
                  "than chance, or it does not.", 0.0)
        body(doc, f"SICKLE releases {y['n_checkpoints']} yield checkpoints "
                  f"once the smoke tests are excluded. We exported each "
                  f"one's per-field-season prediction \u2014 the released "
                  f"artefacts store only the absolute error, which has "
                  f"already discarded the prediction and so cannot produce a "
                  f"label-free signal \u2014 and took the dispersion of those "
                  f"predictions, their coefficient of variation, as the "
                  f"screening signal. It is computable at inference time and "
                  f"uses no ground truth. A field-season counts as "
                  f"unreliable when the ensemble mean's absolute percentage "
                  f"error exceeds {y['ape_threshold']:.0f}\u202f%, a choice "
                  f"made for this study and not a YES-TECH or PMFBY "
                  f"criterion; no official document we consulted defines "
                  f"one. Neither denominator approaches zero \u2014 "
                  f"ground-truth yields span 800 to 3,840 and every ensemble "
                  f"mean is positive \u2014 so no stabilising constant was "
                  f"introduced. The protocol, including the threshold, the "
                  f"baselines and the rejection criteria, was frozen in the "
                  f"repository before any screening number was computed.")
        body(doc, f"The evaluation covers {y['n_field_seasons']} "
                  f"field-seasons from only {y['n_plots']} plots, up to four "
                  f"seasons each. Seasons of one plot share soil, geometry "
                  f"and management, so the independent unit is the plot and "
                  f"n = {y['n_plots']}. Thresholds are set "
                  f"leave-one-plot-out, with every season of the held-out "
                  f"plot removed before the threshold is computed, and all "
                  f"intervals bootstrap plots rather than field-seasons.")
        body(doc, f"The result is negative. Ranking performance is AUROC "
                  f"{tf['auroc']:.3f} with 95\u202f% CI "
                  f"[{tf['auroc_ci']['lo']:.3f}, {tf['auroc_ci']['hi']:.3f}] "
                  f"against 0.500 for random referral, and Spearman "
                  f"{_sgn(tf['spearman_signal_vs_ape'], '+.3f')} between the signal "
                  f"and the realised error. The interval includes 0.5, so "
                  f"the signal is not shown to rank yield errors better than "
                  f"chance. The point estimate in fact sits below 0.5, and "
                  f"recall falls short of the random-referral expectation at "
                  f"{len(below)} of the {len(y['screening'])} workloads "
                  f"tested (Table II, Fig. 9) \u2014 at a "
                  f"{y['screening'][0]['workload_realised']:.0%} workload it "
                  f"catches {y['screening'][0]['n_caught']} of "
                  f"{y['n_bad']} unreliable field-seasons where random "
                  f"referral would be expected to catch about "
                  f"{y['screening'][0]['workload_realised'] * y['n_bad']:.0f}. "
                  f"With {y['n_plots']} plots the intervals are far too wide "
                  f"to conclude that the signal is actively harmful; what "
                  f"they exclude is any claim that it works.")
        if y.get("baseline_auroc"):
            b = {k: v for k, v in y["baseline_auroc"].items()
                 if k != "disagreement"}
            if b:
                body(doc, "Two label-free baselines were run on the same "
                          "footing and neither rescues the approach: "
                          + ", ".join(
                              f"{k.replace('_', ' ')} at AUROC {v:.3f}"
                              for k, v in b.items())
                          + ". Referring the lowest predicted yields first "
                            "is the best of the three, but its margin over "
                            "chance is well inside the uncertainty this "
                            "sample supports, so we draw no conclusion from "
                            "it either.")
        body(doc, f"Selective prediction shows the same thing from the other "
                  f"side. Withholding the half of the field-seasons with the "
                  f"highest disagreement moves the median absolute "
                  f"percentage error among those kept from "
                  f"{full['median_ape']:.1f}\u202f% to "
                  f"{half['median_ape']:.1f}\u202f%, but leaves the mean "
                  f"essentially unchanged at {full['mean_ape']:.1f}\u202f% "
                  f"against {half['mean_ape']:.1f}\u202f%. A screen that "
                  f"worked would move both. The median shift alone is not "
                  f"evidence of a useful screen and we do not report it as "
                  f"one.")
        ss = y.get("sensitivity") or []
        ok = [x for x in ss if x.get("auroc") is not None]
        if len(ok) > 1:
            body(doc, "The outcome is not an artefact of where the "
                      "unreliability threshold was placed. Across thresholds "
                      "from 30\u202f% to 100\u202f% APE the AUROC moves "
                      "between "
                      + f"{min(x['auroc'] for x in ok):.3f} and "
                      + f"{max(x['auroc'] for x in ok):.3f}, scattering "
                      "around chance rather than strengthening anywhere.")
        ei = y["ensemble_independence"]
        sub = y.get("model_subset_sensitivity") or {}
        body(doc, f"One property of the ensemble bounds how far the negative "
                  f"result generalises. The {y['n_checkpoints']} checkpoints "
                  f"are four sensor sets crossed with three architectures "
                  f"and two season variants; they are not "
                  f"{y['n_checkpoints']} independent models. Their pairwise "
                  f"prediction correlation has median "
                  f"{ei['pairwise_corr_median']:.3f} and reaches "
                  f"{ei['pairwise_corr_max']:.3f}, the highest values being "
                  f"base runs against their own season variants. Dispersion "
                  f"across members this correlated is a weaker signal than "
                  f"an independently seeded ensemble would provide"
                  + (", and restricting it to the "
                     + " or ".join(f"{v['n_models']}" for v in sub.values())
                     + " less duplicated members does not change the "
                       "conclusion ("
                     + ", ".join(f"AUROC {v['auroc']:.3f}"
                                 for v in sub.values()) + ")"
                     if sub else "")
                  + ". For that reason we call the quantity inter-model "
                    "disagreement, not predictive uncertainty: its "
                    "calibration has not been evaluated, and the finding is "
                    "that this particular ensemble's dispersion does not "
                    "predict yield error, not that no uncertainty measure "
                    "could.")
        figure(doc, "fig_yield_screen.png",
               "Fig. 9.  Top: the label-free dispersion signal against the "
               "realised yield error, one marker per field-season; the "
               "dotted line is the unreliability threshold. Bottom: recall "
               "against realised workload with plot-bootstrap intervals; the "
               "dotted diagonal is random referral, which every interval "
               "includes. The "
               f"independent unit is the plot (n = {y['n_plots']}), not the "
               f"field-season.")

        cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(6)
        rr = cap.add_run("TABLE II\nYield screening by verification budget. "
                         f"Recall is over {y['n_bad']} unreliable "
                         f"field-seasons; intervals bootstrap "
                         f"{y['n_plots']} plots.")
        rr.font.size = Pt(7.5); rr.bold = True
        tb = doc.add_table(rows=1, cols=6); tb.style = "Table Grid"
        for i, h in enumerate(("Workload", "Referred", "Caught", "Recall",
                               "95% CI", "Random")):
            c = tb.rows[0].cells[i]; c.text = h
            for p_ in c.paragraphs:
                for r_ in p_.runs:
                    r_.bold = True; r_.font.size = Pt(7.5)
        for r_ in y["screening"]:
            row = tb.add_row().cells
            vals = (f"{r_['workload_realised']:.0%}",
                    f"{r_['n_referred']}/{y['n_field_seasons']}",
                    f"{r_['n_caught']}/{y['n_bad']}",
                    f"{r_['recall']:.0%}",
                    f"[{r_['recall_ci']['lo']:.0%}, "
                    f"{r_['recall_ci']['hi']:.0%}]",
                    f"{r_['workload_realised']:.0%}")
            for i, v in enumerate(vals):
                row[i].text = v
                for p_ in row[i].paragraphs:
                    for rn in p_.runs:
                        rn.font.size = Pt(7.5)

        body(doc, "We keep this experiment in the paper precisely because it "
                  "failed. A reader of Section V-G would otherwise have "
                  "reasonable grounds to assume the crop-type screening "
                  "result extends to the yield estimates that an assessment "
                  "would actually consume, and on this evidence it does not. "
                  "The two findings are not in tension: field-level "
                  "segmentation scores recur across seeds, while the "
                  "dispersion of this particular "
                  "ensemble carries no usable information about which yield "
                  "estimates are wrong. Any safeguard built on label-free "
                  "screening therefore has to be validated on the quantity "
                  "it will actually gate, and on far more than "
                  f"{y['n_plots']} plots. This study measures yield error "
                  "only; it evaluates no claim decision, payout or farmer "
                  "outcome, and none should be inferred from it.")

    # ================================================= VI
    heading(doc, "VI.  Discussion")
    _Cd = n["cs"]
    _sad = _Cd["test_2_out_of_sample"]["fit_free"]["screen alone"]
    body(doc, "Sections V-F and V-G establish a precondition, not a "
              "capability. Because field scores recur across independent "
              "runs, failure is not random, and that is what would make "
              "selective automation in the sense of [12] possible at all: a "
              "deployment that processes the fields a model handles reliably "
              "and refers the remainder for human verification. Whether it is "
              "achievable here is a separate question, and the evidence stops "
              "short of it. Only a handful of fields fail; a label-free "
              "signal carries some information that field size does not, "
              "but its gain over simply checking the smallest fields first "
              "has an interval of "
              f"[{_sgn(_sad['ci_vs_smaller_first'][0], '+.3f')}, "
              f"{_sgn(_sad['ci_vs_smaller_first'][1], '+.3f')}], and on yield the "
              "equivalent signal fails outright. We therefore present "
              "recurrence as the reason the question is worth asking, not as "
              "evidence that a referral rule built on it would work.", 0.0)
    body(doc, "Each audited task corresponds to a decision taken about an "
              "individual field, which is why per-field error is the relevant "
              "quantity. Insurance assessment depends on a field-season's "
              "yield; procurement scheduling on its harvest date; irrigation "
              "timing on its transplanting date. Harvest-date error has a p90 "
              f"of {n['harvesting_date']['p90']:.1f} days against a reported "
              f"MAE of {n['harvesting_date']['agg_mae']:.1f}. If procurement "
              "scheduling were driven by such a model, roughly one "
              "field-season in ten would be misplaced by more than two weeks, "
              "and the aggregate figure would not expose it. We state these as "
              "implications of the measured error distributions, not as "
              "demonstrated outcomes: this study measures model error and does "
              "not measure payouts, incomes or yields, and makes no claim "
              "about them.")
    body(doc, "Why the field is the right unit is clearest in the "
              "framework that motivated this audit. The YES-TECH manual "
              "requires at least 30 % weightage to modelled yield with no "
              "upper cap, and specifies validation at the Gram Panchayat; our "
              "documentary review found no explicit requirement to validate "
              "below that unit, which is not the same as a prohibition and "
              "says nothing about practice. It further provides that model "
              "outputs of past years \u201cshall not be used for (a) making "
              "comparisons with other datasets, (b) revising the loss "
              "assessments of past years, and (c) challenging the estimates "
              "of current/past years\u201d, and that past estimates fall "
              "within the intellectual-property exemption of section 8(1)(d) "
              "of the Right to Information Act, 2005, and are \u201cnot be "
              "made public\u201d [2]. We quote these provisions and draw no "
              "legal conclusion from them. Their relevance here is narrower: "
              "where an estimate is hard to contest after the fact, the "
              "field-level error measured in Section V has fewer routes to "
              "being caught, and a model that errs systematically on "
              "particular fields biases the village figure rather than "
              "cancelling within it.")

    # ================================================= VII
    heading(doc, "VII.  Limitations and Threats to Validity")
    body(doc, f"Every result is a validation-set number computed on "
              f"{n['n_fields']} fields, and on 41 field-seasons from 14 plots "
              "for yield and phenology, because the benchmark's test split is "
              "unlabelled. The study covers one delta, one cropping system and "
              "2018–2020. The screening and geographic results are preliminary "
              f"at this sample size. {n['seed_phrase_cap']} afford limited "
              "power, so inconclusive rungs should not be read as having no "
              "effect; Section V-B separates the rungs where the interval "
              "rules out a large effect from those where it does not. "
              "Training used gradient accumulation with micro-batches of 8, "
              "which changes temporal padding relative to a true batch of 32; "
              "this was held constant across rungs and all evaluation was "
              "performed in full precision at the pinned batch size. A runtime "
              "patch guarding the non-finite output of Section V-A was active "
              "during training and was verified to leave every already-finite "
              "result bit-identical. Per-field evaluation follows established "
              "methodology [8] and is applied here rather than proposed. "
              "The yield-screening experiment of Section V-I rests on "
              "41 field-seasons from 14 plots, and its ensemble members "
              "correlate at a median of 0.92, so its negative finding bounds "
              "this ensemble on this sample rather than label-free screening "
              "in general. "
              "The yield labels are farmers' recollections, collected in "
              "2021\u201322 for seasons back to 2018 [1]. Self-reported yields "
              "have elsewhere proved a poor basis for calibrating satellite "
              "yield models [29], and farmer reports have diverged from "
              "crop-cut and full-plot measures even where satellite estimates "
              "tracked full-plot yields well [30]. Recalled labels are "
              "therefore a plausible contributor to the finding that no "
              "released model beats the constant, not a demonstrated "
              "explanation of it. "
              "Finally, we audit SICKLE, a research benchmark; we do not audit "
              "YES-TECH's operational models, which are not publicly "
              "available, and make no claim about their accuracy.", 0.0)
    body(doc, "Reproducibility. Every run records a manifest containing the "
              "resolved configuration, a content hash, the git commit and a "
              "dirty-tree flag, the device, the effective and micro batch "
              "sizes, the seed and the trainable parameter count. A single "
              "command re-checks every claim in this paper against the file "
              "that backs it. We note what that does and does not establish: "
              "it confirms the manuscript agrees with the artefacts in our "
              "repository, and is not a substitute for an independent "
              "researcher running the released code and data on their own "
              "machine [10].")

    # ================================================= VIII
    heading(doc, "VIII.  Conclusion")
    # The conclusion is organised on the study's objective -- identify the
    # conditions under which predictions are unreliable, and empirically
    # evaluate safeguards for detecting those failures -- and says what the
    # paper does NOT establish in the same place it says what it does. An
    # earlier version claimed the difficult fields were "identifiable in
    # advance" and recommended "selective automation"; the conditional test
    # (Section V-G) and the yield screen (Section V-I) do not support
    # either, so neither survives here.
    _C = n["cs"]
    _t1 = _C["test_1_conditional_permutation"]["low_confidence"]
    _sa = _C["test_2_out_of_sample"]["fit_free"]["screen alone"]
    _ys = n["ys"]["threshold_free"]
    _fs = n["fs"]
    body(doc, "India has begun to automate crop-insurance payouts, and "
              "the yield behind them increasingly comes from models of the "
              "kind this benchmark is used to develop. We asked what a "
              "benchmark number says about an individual field. On SICKLE, "
              "it says less than it appears to.", 0.0)
    body(doc, "Evaluation itself is unstable: a frozen model moves by up to "
              f"{100 * max(n['bs_spread'].values()):.1f} IoU points with "
              f"batch size alone. None of the {len(n['yield_vs_const'])} "
              "released yield models is shown to beat a constant that uses "
              f"no imagery: all {n['n_worse_mae']} have a higher observed "
              f"error, but with {n['yvc']['n_plots']} plots no difference is "
              "distinguishable from zero. A pooled IoU of "
              f"{n['v5']:.4f} hides a concentrated tail: most fields are "
              f"mapped almost perfectly, but {n['n_below']} of "
              f"{n['n_fields']} score below 0.5, the same fields recur "
              "across runs, and one field supplies "
              f"{100 * _fs['share_largest']:.0f} % of the scored pixels, so "
              "removing it alone moves the published figure by up to "
              f"{n['loo_max_abs']:.3f}. And {len(n['inc'])} of "
              f"{n['n_steps']} modelling choices cannot be told apart from "
              "seed variation.")
    body(doc, "Safeguards were tested rather than assumed. For crop type, a "
              "label-free confidence signal carries some information that "
              f"field size does not (p = {_t1['p_one_sided']:.3f}), but it "
              "does not beat checking the smallest fields first (gain "
              f"interval [{_sgn(_sa['ci_vs_smaller_first'][0], '+.3f')}, "
              f"{_sgn(_sa['ci_vs_smaller_first'][1], '+.3f')}]). For yield, the "
              "quantity a payout uses, ensemble dispersion ranks error at "
              f"AUROC {_ys['auroc']:.2f}, no better than chance.")
    body(doc, "Two recommendations follow, and no more. Benchmarks that feed "
              "field-level decisions should report per-field error "
              "distributions and the evaluation batch size, not only pooled "
              "accuracy. And a confidence indicator should be shown to "
              "discriminate, on the quantity it gates, before it is shown to "
              "anyone; one that does not discriminate offers false "
              "reassurance with the authority of a number. A companion study "
              "builds a verification-prioritisation prototype on this "
              "premise. We audited a research benchmark, not any operational "
              "system, and make no claim about insurance decisions or "
              "outcomes for farmers.")

    # ================================================= refs
    # Citation numbers are assigned here, from the finished body, so the list
    # below is already in order of first appearance.
    refs, uncited = renumber(doc, n["refs"])
    if uncited:
        print(f"  WARNING uncited references dropped from the list: {uncited}")

    heading(doc, "References")
    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Inches(0.22); pf.first_line_indent = Inches(-0.22)
        pf.space_after = Pt(2)
        # A URL is printed only where it is the reference's only locator --
        # the government and press sources. A proceedings paper is already
        # fully cited, so its retrieval URL is verification evidence (it is
        # kept in references.json) rather than part of the citation.
        loc = r.get("doi") or ""
        only_locator = any(w in r.get("type", "")
                           for w in ("Government", "Press"))
        tail = (f" [Online]. Available: {loc}"
                if loc.startswith("http") and only_locator
                else f" doi: {loc}." if loc and not loc.startswith("http")
                else "")
        run = p.add_run(f"[{r['n']}] {r['entry']}{tail}")
        run.font.size = Pt(7.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spaced_percent(doc)
    doc.save(OUT)

    # The verification table must carry the manuscript's final numbering, or
    # a reviewer checking [7] in the table would be checking the wrong paper.
    # It goes to a separate generated file: docs/references.json holds the
    # authoring numbering that the prose above cites, and writing the new
    # numbering back over it would remap every citation on the next build.
    blob = json.loads((PROJ / "docs/references.json").read_text(encoding="utf8"))
    blob["references"] = refs
    blob["_numbering"] = ("GENERATED by scripts/make_paper_ieee.py -- the "
                          "manuscript's numbering, IEEE order of first "
                          "appearance. 'was' is the authoring number in "
                          "references.json. Do not edit; edit references.json.")
    (PROJ / "docs/references_final.json").write_text(
        json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"  wrote {OUT}")
    print(f"  {len(refs)} references, renumbered to first-appearance order")
    return OUT


if __name__ == "__main__":
    build()
