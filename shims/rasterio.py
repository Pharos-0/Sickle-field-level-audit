"""A minimal `rasterio` stand-in backed by tifffile.

WHY THIS EXISTS
---------------
Windows Smart App Control on this machine blocks rasterio's GDAL DLLs:

    ImportError: DLL load failed while importing _base:
    An Application Control policy has blocked this file.

This is consistent, not intermittent — three consecutive imports failed.

The reference SICKLE implementation (`reference_impl/utils/dataset.py`) uses
exactly one rasterio API:

    with rasterio.open(path, 'r') as fp:
        mask = fp.read()

`rasterio.open(...).read()` returns a band-first array of shape
(bands, H, W). `tifffile.imread` returns the same shape and dtype for these
files, verified against the SICKLE masks:

    masks/10m/0.tif -> (6, 33, 33) float64
    masks/30m/0.tif -> (6, 12, 12) float64
    masks/3m/0.tif  -> (6, 108, 108) float64

This shim therefore lets the authors' code run **unmodified**, which is what a
reproduction requires. Put this directory first on sys.path.

It implements only what that code path needs. Anything else raises rather than
silently returning something plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

__version__ = "0.0-shim-tifffile"
__shim__ = True


class _Dataset:
    """The subset of a rasterio DatasetReader that the reference code uses."""

    def __init__(self, path: str | Path) -> None:
        self.name = str(path)
        self._arr: np.ndarray | None = None

    # ---- lazily read so that open() is cheap -----------------------------
    @property
    def _data(self) -> np.ndarray:
        if self._arr is None:
            a = tifffile.imread(self.name)
            if a.ndim == 2:            # single band -> (1, H, W), as rasterio does
                a = a[None, ...]
            self._arr = a
        return self._arr

    def read(self, indexes=None):
        """Band-first array. `indexes` is 1-based, matching rasterio."""
        a = self._data
        if indexes is None:
            return a
        if isinstance(indexes, int):
            return a[indexes - 1]
        return a[[i - 1 for i in indexes]]

    # ---- metadata the code may touch -------------------------------------
    @property
    def count(self) -> int:
        return int(self._data.shape[0])

    @property
    def height(self) -> int:
        return int(self._data.shape[1])

    @property
    def width(self) -> int:
        return int(self._data.shape[2])

    @property
    def dtypes(self):
        return tuple([str(self._data.dtype)] * self.count)

    @property
    def shape(self):
        return (self.height, self.width)

    def close(self) -> None:
        self._arr = None

    def __enter__(self) -> "_Dataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<rasterio-shim Dataset {self.name!r}>"


def open(fp, mode: str = "r", **kwargs):  # noqa: A001 - mirrors rasterio.open
    if mode not in ("r", "rb"):
        raise NotImplementedError(
            f"rasterio shim supports read-only access; got mode={mode!r}. "
            "Writing is not implemented and must not be silently faked.")
    return _Dataset(fp)


def __getattr__(name: str):
    raise AttributeError(
        f"rasterio.{name} is not provided by this tifffile-backed shim. "
        "The shim implements only `open(path).read()`, which is the sole "
        "rasterio API the SICKLE reference implementation uses. If new code "
        "needs more, implement it explicitly rather than widening the shim.")
