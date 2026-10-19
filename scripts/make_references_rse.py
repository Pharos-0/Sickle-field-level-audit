"""Elsevier (Remote Sensing of Environment) versions of the references.

    python scripts/make_references_rse.py

Writes docs/references_rse.json: for every entry of docs/references.json,
the in-text citation key (author part and year) and the full reference in
the journal's name-date style. Author lists come from the Crossref record of
the DOI wherever there is one, so no name is typed by hand; the remaining
entries (proceedings without DOIs, preprints, government and press sources)
are written out below from the records already verified in
docs/references.json. Journal names follow ISO 4 / LTWA abbreviations.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "sickle-audit/1.0 (reference formatting; mailto:none@example.org)"}

# authoring number -> (venue string in name-date style, given after the title)
VENUE = {
    1: "in: Proceedings of the IEEE/CVF Winter Conference on Applications of "
       "Computer Vision (WACV), pp. 5983–5992",
    6: "Remote Sens. 13, 1870",
    7: "Remote Sens. 10, 911",
    9: "Sci. Data 12, 664",
    10: "Proc. AAAI Conf. Artif. Intell. 32 (1)",
    13: "Nat. Commun. 11, 4540",
    14: "Ecol. Model. 457, 109692",
    15: "in: IGARSS 2024 – IEEE International Geoscience and Remote "
        "Sensing Symposium, pp. 1540–1545",
    16: "Remote Sens. Environ. 233, 111410",
    17: "in: Medical Image Computing and Computer-Assisted Intervention "
        "– MICCAI 2016, Lect. Notes Comput. Sci. 9901, pp. 424–432",
    18: "in: Proceedings of the IEEE/CVF International Conference on "
        "Computer Vision (ICCV), pp. 4852–4861",
    25: "Science 380 (6641), 136–138",
    28: "Int. J. Appl. Earth Obs. Geoinf. 139, 104559",
    29: "Front. Sustain. Food Syst. 4, 25",
    30: "Am. J. Agric. Econ. 102 (1), 202–219",
    31: "IEEE Geosci. Remote Sens. Lett. 22, 1–5",
    32: "Proc. AAAI Conf. Artif. Intell. 40 (45), 38533–38541",
    33: "Remote Sens. 18, 1581",
    34: "Inf. Process. Agric., in press",
    36: "in: International Conference on Knowledge Engineering and "
        "Communication Systems (ICKECS), pp. 1–5",
    37: "in: International Conference on Computer Networks and Inventive "
        "Communication Technologies (ICCNCT), pp. 1452–1459",
    38: "in: Proceedings of the International Conference on Artificial "
        "Intelligence and Machine Learning for Agro-Ecosystems (ICAIML-AE "
        "2026), Adv. Intell. Syst. Res., Atlantis Press, pp. 455–466",
}

# Entries without a Crossref author list, written from their verified
# records (see "verification" in docs/references.json for each).
MANUAL = {
    2: ("MNCFC", "2023a",
        "MNCFC (Mahalanobis National Crop Forecast Centre), 2023a. YES-TECH "
        "Manual 2023: Yield Estimation System based on Technology under "
        "PMFBY. Department of Agriculture and Farmers Welfare, Government of "
        "India, New Delhi. https://www.ncfc.gov.in/downloads/YESTECH_Manual.pdf"),
    20: ("MNCFC", "2023b",
         "MNCFC (Mahalanobis National Crop Forecast Centre), 2023b. Yield "
         "Estimation System based on Technology (YES-TECH) under PMFBY: "
         "Manual for Implementation, January 2023. Department of Agriculture "
         "and Farmers Welfare, Government of India, New Delhi. "
         "https://apsac.ap.gov.in/wp-content/uploads/2023/01/Document2.pdf"),
    3: ("DAFW", "2020",
        "DAFW (Department of Agriculture and Farmers Welfare), 2020. Pradhan "
        "Mantri Fasal Bima Yojana: Revised Operational Guidelines. "
        "Government of India, New Delhi. "
        "https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf"),
    21: ("MoAFW", "2026",
         "MoAFW (Ministry of Agriculture and Farmers Welfare), 2026. Pending "
         "claims of PMFBY. Lok Sabha Starred Question No. 25, answered 21 "
         "July 2026. Government of India, New Delhi. https://sansad.in/getFile/"
         "lsapps/loksabhaquestions/annex/188/AS25_tpFUNy.pdf"),
    4: ("Mahaur", "2026a",
        "Mahaur, C., 2026a. Can technology make India’s crop insurance "
        "payouts more accurate? Down To Earth, 27 May 2026. https://www."
        "downtoearth.org.in/agriculture/can-technology-make-indias-crop-"
        "insurance-payouts-more-accurate"),
    22: ("Mahaur", "2026b",
         "Mahaur, C., 2026b. Implementation of Yield Estimation System based "
         "on Technology (YES-TECH) in Madhya Pradesh: Learnings from select "
         "districts towards improved claim settlement against crop losses as "
         "part of PMFBY. Presentation, Centre for Science and Environment, "
         "New Delhi, May 2026. https://cdn.cseindia.org/attachments/"
         "0.70726700_1779095588_yestech.pdf"),
    5: ("Mahaur and Khurana", "2026",
        "Mahaur, C., Khurana, A., 2026. Beyond the lens: Madhya Pradesh "
        "farmers struggle with satellite-based crop insurance under PMFBY. "
        "Down To Earth, 20 August 2026. https://www.downtoearth.org.in/"
        "agriculture/beyond-the-lens"),
    8: ("Wang et al.", "2023",
        None),     # DOI 10.52202/075280-2630 -- filled from Crossref below
    11: ("Lakshminarayanan et al.", "2017",
         "Lakshminarayanan, B., Pritzel, A., Blundell, C., 2017. Simple and "
         "scalable predictive uncertainty estimation using deep ensembles, "
         "in: Advances in Neural Information Processing Systems 30 (NIPS "
         "2017)."),
    12: ("Geifman and El-Yaniv", "2017",
         "Geifman, Y., El-Yaniv, R., 2017. Selective classification for deep "
         "neural networks, in: Advances in Neural Information Processing "
         "Systems 30 (NIPS 2017)."),
    19: ("Shi et al.", "2015",
         "Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., Woo, W.-C., "
         "2015. Convolutional LSTM network: A machine learning approach for "
         "precipitation nowcasting, in: Advances in Neural Information "
         "Processing Systems 28 (NIPS 2015)."),
    26: ("Bouthillier et al.", "2021",
         "Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., "
         "Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., "
         "Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., "
         "Pal, C., Varoquaux, G., Vincent, P., 2021. Accounting for variance "
         "in machine learning benchmarks, in: Proceedings of Machine Learning "
         "and Systems (MLSys) 3."),
    37: ("Merphin R et al.", "2026",
         "Merphin R, Patan Usman Khan, Cherukuri Nandakumar, Ajith V, Bandaru "
         "Venkatesh, Ram Abishek S, 2026. An Enhanced AI-based Agricultural "
         "Risk & Insurance Trigger System, in: International Conference on "
         "Computer Networks and Inventive Communication Technologies "
         "(ICCNCT), pp. 1452–1459. "
         "https://doi.org/10.1109/ICCNCT68477.2026.11590693"),
    27: ("Stigler et al.", "2022",
         "Stigler, M., Dey, A., Hobbs, A., Lobell, D., 2022. With big data "
         "come big problems: Pitfalls in measuring basis risk for crop index "
         "insurance. arXiv preprint arXiv:2209.14611. "
         "https://doi.org/10.48550/arXiv.2209.14611"),
    35: ("Adjei", "2026",
         "Adjei, Y.O., 2026. Do foundation model embeddings improve "
         "cross-country crop yield generalisation? A leave-one-country-out "
         "evaluation in Sub-Saharan Africa. arXiv preprint arXiv:2605.08113. "
         "https://doi.org/10.48550/arXiv.2605.08113"),
}
# NeurIPS/NIPS online proceedings carry no page numbers, so none are given.
# Bouthillier et al.: 16 authors in this order, read from
# proceedings.mlsys.org/paper_files/paper/2021 on 2026-10-20.

# Crossref splits a few names differently from the authors' own records:
# SICKLE's CVF BibTeX gives "Devshali, Charu Chandra"; Wang et al.'s
# co-author is Luc Van Gool; Garnot and Landrieu's own (CVF) BibTeX gives
# "Garnot, Vivien Sainte Fare".
AUTHOR_FIX = {
    1: [("Chandra Devshali, C.", "Devshali, C.C.")],
    8: [("Gool, L.V.", "Van Gool, L.")],
    18: [("Fare Garnot, V.S.", "Garnot, V.S.F.")],
}
KEY_FIX = {18: "Garnot and Landrieu"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=60) as r:
        return r.read()


def initials(given: str) -> str:
    out = []
    for part in given.replace(".", ". ").split():
        if "-" in part:
            out.append("-".join(p[0] + "." for p in part.split("-") if p))
        elif part:
            out.append(part[0] + ".")
    return "".join(out)


def author_list(auth):
    names = []
    for a in auth:
        fam, giv = a.get("family", "").strip(), a.get("given", "").strip()
        names.append(f"{fam}, {initials(giv)}" if giv else fam)
    return ", ".join(names)


def cite_key(auth):
    fams = [a.get("family", "") for a in auth]
    if len(fams) == 1:
        return fams[0]
    if len(fams) == 2:
        return f"{fams[0]} and {fams[1]}"
    return f"{fams[0]} et al."


def sentence_case_ok(title):
    return title  # titles are kept as registered


def mlsys_authors():
    t = get("https://proceedings.mlsys.org/paper_files/paper/2021").decode("utf8", "replace")
    import re
    plain = " ".join(re.sub(r"<[^>]+>", " ", t).split())
    i = plain.find("Accounting for Variance in Machine Learning Benchmarks")
    seg = plain[i + len("Accounting for Variance in Machine Learning Benchmarks"):][:900]
    # the author list runs until the next paper title; cut at the first
    # run of words that is not a comma-separated name list
    names = []
    for chunk in seg.split(","):
        c = chunk.strip()
        if not c or len(c.split()) > 4 or any(ch.isdigit() for ch in c):
            break
        names.append(c)
    return names


def main():
    refs = json.loads((PROJ / "docs/references.json").read_text(encoding="utf8"))["references"]
    refs += json.loads((PROJ / "docs/references_thesis.json").read_text(encoding="utf8"))["references"]
    out = {}
    for r in refs:
        n = r["n"]
        if n in MANUAL and MANUAL[n][2]:
            key, yr, full = MANUAL[n]
            out[n] = {"key": key, "year": yr, "full": full}
            continue
        doi = r.get("doi") or ""
        if n == 8:
            doi = "10.52202/075280-2630"
        if False:
            names = mlsys_authors()
            fams = [x.split()[-1] for x in names]
            au = ", ".join(f"{x.split()[-1]}, {initials(' '.join(x.split()[:-1]))}"
                           for x in names)
            out[n] = {"key": "Bouthillier et al.", "year": "2021",
                      "full": f"{au}, 2021. Accounting for variance in machine "
                              "learning benchmarks, in: Proceedings of Machine "
                              "Learning and Systems (MLSys) 3.",
                      "n_authors": len(names), "authors_raw": names}
            continue
        if not doi or doi.startswith("http"):
            out[n] = {"key": None, "year": None, "full": None,
                      "note": "no DOI and no manual entry"}
            continue
        time.sleep(0.5)
        m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
        auth = m.get("author", [])
        dp = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0]
        yr = str(dp[0])
        title = (m.get("title") or [""])[0].replace("&amp;", "&")
        venue = VENUE.get(n)
        if n == 8:
            venue = ("in: Advances in Neural Information Processing Systems 36 "
                     "(NeurIPS 2023), Datasets and Benchmarks Track")
        if venue is None:
            out[n] = {"key": cite_key(auth), "year": yr, "full": None,
                      "note": f"no venue string for n={n}"}
            continue
        if title.endswith(("?", "!")):
            sep = " " if not venue.startswith("in:") else ", "
        else:
            sep = ", " if venue.startswith("in:") else ". "
        authors = author_list(auth)
        for a, b in AUTHOR_FIX.get(n, []):
            assert a in authors, (n, a, authors)
            authors = authors.replace(a, b)
        full = f"{authors}, {yr}. {title}{sep}{venue}. https://doi.org/{doi}"
        out[n] = {"key": KEY_FIX.get(n, cite_key(auth)), "year": yr,
                  "full": full, "n_authors": len(auth)}
    (PROJ / "docs/references_rse.json").write_text(
        json.dumps({"_note": "GENERATED by scripts/make_references_rse.py from "
                             "Crossref records and the verified entries of "
                             "references.json. Do not edit by hand.",
                    "references": {str(k): v for k, v in sorted(out.items())}},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    for k, v in sorted(out.items()):
        print(k, "|", v["key"], v["year"], "|", (v["full"] or v.get("note"))[:400])


if __name__ == "__main__":
    main()
