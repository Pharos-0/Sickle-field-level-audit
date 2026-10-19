"""Save every figure as a vector PDF beside its PNG.

Importing this module patches matplotlib's Figure.savefig: a call that
writes ``name.png`` also writes ``name.pdf`` with the same arguments (minus
the raster resolution). The PNGs stay the source for the Word manuscripts;
the PDFs are the vector artwork a journal asks for (Remote Sensing of
Environment: vector EPS/PDF for charts, or 1000 dpi bitmaps).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.figure as _mf

_ORIG = _mf.Figure.savefig


def _savefig_with_pdf(self, fname, *args, **kwargs):
    out = _ORIG(self, fname, *args, **kwargs)
    try:
        p = Path(fname)
    except TypeError:            # a file object, not a path: leave it alone
        return out
    if p.suffix.lower() == ".png":
        kw = {k: v for k, v in kwargs.items() if k != "dpi"}
        _ORIG(self, p.with_suffix(".pdf"), *args, **kw)
    return out


if getattr(_mf.Figure.savefig, "__name__", "") != "_savefig_with_pdf":
    _mf.Figure.savefig = _savefig_with_pdf
