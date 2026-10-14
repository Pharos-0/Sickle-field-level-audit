"""Export a .docx to PDF through Word, and report the page count.

    python scripts/export_pdf.py docs/SICKLE_Field_Level_Audit_IEEE.docx

Word is used rather than a converter because the manuscript's two-column
section breaks and figure anchoring are Word features; a third-party
converter can reflow them, which would mean the PDF a reviewer reads is not
the document that was checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

WD_FORMAT_PDF = 17


def export(docx: Path) -> Path:
    import win32com.client as win32

    docx = docx.resolve()
    pdf = docx.with_suffix(".pdf")
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    doc = None
    try:
        doc = word.Documents.Open(str(docx), ReadOnly=False)
        doc.Repaginate()
        pages = doc.ComputeStatistics(2)        # wdStatisticPages
        words = doc.ComputeStatistics(0)        # wdStatisticWords
        doc.SaveAs2(str(pdf), FileFormat=WD_FORMAT_PDF)
        print(f"  {docx.name}: {pages} pages, {words} words")
        print(f"  wrote {pdf}")
    finally:
        if doc is not None:
            doc.Close(SaveChanges=0)
        word.Quit()
    return pdf


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    export(Path(sys.argv[1]))
