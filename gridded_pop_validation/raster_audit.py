"""Raster grid alignment auditing and array anomaly checks for population grids.

Uses plain 6-element affine tuples `(a, b, c, d, e, f)` following GDAL/rasterio
north-up convention:
    x = c + col * a + row * b
    y = f + col * d + row * e
where `a > 0` (pixel width), `e < 0` (negative pixel height), `b = d = 0` for
unrotated north-up grids.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _unpack_north_up(transform: Sequence[float]) -> tuple[float, float, float, float]:
    vals = tuple(float(v) for v in transform[:6])
    if len(vals) != 6:
        raise ValueError("transform must have at least 6 affine elements (a, b, c, d, e, f)")
    a, b, c, d, e, f = vals
    if abs(b) > 1e-12 or abs(d) > 1e-12:
        raise ValueError("rotated rasters (non-zero b or d) are not supported")
    if a <= 0.0 or e >= 0.0:
        raise ValueError("expected north-up transform with a > 0 and e < 0")
    return a, c, e, f


def check_window_alignment(
    transform: Sequence[float],
    bounds: tuple[float, float, float, float],
    *,
    tol_pixels: float = 1e-6,
) -> dict[str, float | bool]:
    """Detect fractional-pixel offsets between a requested clipping box and raster origin.

    Clipping with non-integer pixel windows without updating the affine origin can
    introduce silent sub-pixel shifts in small-area zonal population totals.

    Parameters
    ----------
    transform : sequence of 6 floats
        Affine transform `(a, b, c, d, e, f)` (compatible with `rasterio.Affine`).
    bounds : (minx, miny, maxx, maxy)
        Requested clipping bounding box in the same CRS as `transform`.
    tol_pixels : float
        Maximum fractional pixel offset considered aligned.
    """
    a, c, e, f = _unpack_north_up(transform)
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if minx >= maxx or miny >= maxy:
        raise ValueError("bounds must satisfy minx < maxx and miny < maxy")

    col_left = (minx - c) / a
    col_right = (maxx - c) / a
    row_top = (maxy - f) / e
    row_bottom = (miny - f) / e

    def frac_offset(val: float) -> float:
        return val - round(val)

    dx_left_px = frac_offset(col_left)
    dx_right_px = frac_offset(col_right)
    dy_top_px = frac_offset(row_top)
    dy_bottom_px = frac_offset(row_bottom)

    max_abs_offset = max(
        abs(dx_left_px),
        abs(dx_right_px),
        abs(dy_top_px),
        abs(dy_bottom_px),
    )

    return {
        "is_pixel_aligned": bool(max_abs_offset <= tol_pixels),
        "dx_left_pixels": float(dx_left_px),
        "dx_right_pixels": float(dx_right_px),
        "dy_top_pixels": float(dy_top_px),
        "dy_bottom_pixels": float(dy_bottom_px),
        "dx_left_units": float(dx_left_px * a),
        "dy_top_units": float(dy_top_px * abs(e)),
        "max_abs_offset_pixels": float(max_abs_offset),
    }


def snap_bounds_to_grid(
    transform: Sequence[float],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    """Snap a bounding box outward to exact integer pixel boundaries.

    Returns
    -------
    snapped_bounds : (minx, miny, maxx, maxy)
        Outward-snapped coordinates aligned with the parent raster grid.
    window : (col_off, row_off, width, height)
        Exact integer pixel window corresponding to `snapped_bounds`.
    """
    a, c, e, f = _unpack_north_up(transform)
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if minx >= maxx or miny >= maxy:
        raise ValueError("bounds must satisfy minx < maxx and miny < maxy")

    col_start = math.floor((minx - c) / a + 1e-12)
    col_stop = math.ceil((maxx - c) / a - 1e-12)
    row_start = math.floor((maxy - f) / e + 1e-12)
    row_stop = math.ceil((miny - f) / e - 1e-12)

    snapped = (
        c + col_start * a,
        f + row_stop * e,
        c + col_stop * a,
        f + row_start * e,
    )
    window = (col_start, row_start, col_stop - col_start, row_stop - row_start)
    return snapped, window


def check_raster_anomalies(
    array: np.ndarray,
    *,
    nodata: float | None = None,
    constant_tile_threshold: int = 16,
) -> dict[str, Any]:
    """Audit a 2D population raster array for common data integrity anomalies.

    Checks for non-finite values, unexpected negative counts, zero-population share,
    and large runs of identical positive values (a frequent sign of uniform-fill tiles).
    """
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {arr.shape}")

    total_cells = int(arr.size)
    if nodata is not None and not np.isnan(nodata):
        valid_mask = arr != float(nodata)
    else:
        valid_mask = np.ones_like(arr, dtype=bool)

    nan_count = int(np.sum(np.isnan(arr) & valid_mask))
    inf_count = int(np.sum(np.isinf(arr) & valid_mask))
    finite_valid = valid_mask & np.isfinite(arr)
    vals = arr[finite_valid]

    neg_count = int(np.sum(vals < 0.0))
    pos_vals = vals[vals > 0.0]

    max_constant_run = 0
    if pos_vals.size > 0:
        uniq, counts = np.unique(np.round(pos_vals, 6), return_counts=True)
        max_constant_run = int(np.max(counts))

    return {
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "total_cells": total_cells,
        "valid_finite_cells": int(vals.size),
        "nan_cells": nan_count,
        "inf_cells": inf_count,
        "negative_cells": neg_count,
        "positive_cells": int(pos_vals.size),
        "sum_positive": float(np.sum(pos_vals)) if pos_vals.size > 0 else 0.0,
        "max_identical_positive_cells": max_constant_run,
        "has_constant_fill_warning": bool(max_constant_run >= constant_tile_threshold),
    }
