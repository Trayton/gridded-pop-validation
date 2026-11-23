"""Census-anchored proportional calibration, log-linear bias adjustment,
and group-based spatial cross-validation splits.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def proportional_calibrate(
    grid_values: Sequence[float],
    unit_ids: Sequence[Any],
    census_totals: Mapping[Any, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rescale grid cell values so each administrative unit sums exactly to its census total.

    If a unit has positive census population but zero grid sum, population is distributed
    uniformly across the cells assigned to that unit and recorded in diagnostics.
    """
    vals = np.asarray(grid_values, dtype=np.float64)
    units = np.asarray(unit_ids)
    if vals.ndim != 1 or units.ndim != 1 or vals.size != units.size:
        raise ValueError("grid_values and unit_ids must be 1D sequences of identical length")
    if np.any(vals < 0) or not np.all(np.isfinite(vals)):
        raise ValueError("grid_values must be finite and non-negative")

    calibrated = vals.copy()
    zero_grid_units: list[Any] = []
    missing_census_units: list[Any] = []

    for u in dict.fromkeys(units.tolist()):
        mask = units == u
        if u not in census_totals:
            missing_census_units.append(u)
            continue
        target = float(census_totals[u])
        if target < 0.0 or not np.isfinite(target):
            raise ValueError(f"invalid census total for unit {u!r}: {target}")
        cur_sum = float(np.sum(vals[mask]))
        if cur_sum > 0.0:
            calibrated[mask] = vals[mask] * (target / cur_sum)
        else:
            if target > 0.0:
                calibrated[mask] = target / int(np.sum(mask))
                zero_grid_units.append(u)
            else:
                calibrated[mask] = 0.0

    diagnostics = {
        "n_cells": int(vals.size),
        "n_units": len(set(units.tolist())),
        "zero_grid_fallback_units": zero_grid_units,
        "missing_census_units": missing_census_units,
    }
    return calibrated, diagnostics


def log_linear_calibrate(
    train_predicted: Sequence[float],
    train_observed: Sequence[float],
    test_predicted: Sequence[float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit log1p(observed) = alpha + beta * log1p(predicted) via OLS and apply to test data."""
    x_tr = np.log1p(np.asarray(train_predicted, dtype=np.float64))
    y_tr = np.log1p(np.asarray(train_observed, dtype=np.float64))
    x_te = np.log1p(np.asarray(test_predicted, dtype=np.float64))

    if x_tr.size != y_tr.size or x_tr.size < 2:
        raise ValueError("train_predicted and train_observed must match and have length >= 2")

    design = np.column_stack([np.ones_like(x_tr), x_tr])
    coef, _, _, _ = np.linalg.lstsq(design, y_tr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])

    # Duan smearing factor on log1p residuals to reduce retransformation bias
    resid = y_tr - (alpha + beta * x_tr)
    smearing = float(np.mean(np.exp(resid)))

    pred_log = alpha + beta * x_te
    calibrated = np.maximum(0.0, np.expm1(pred_log) * smearing)
    return calibrated, {"alpha": alpha, "beta": beta, "smearing_factor": smearing}


def group_kfold_indices(
    groups: Sequence[Any],
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic group K-fold splits to prevent spatial leakage across contiguous regions."""
    grp = np.asarray(groups)
    unique_groups = sorted({str(g) for g in grp})
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits must be between 2 and number of unique groups ({len(unique_groups)})"
        )

    # Sort groups deterministically by descending size then name, and assign round-robin
    counts = {g: int(np.sum([str(x) == g for x in grp])) for g in unique_groups}
    ordered = sorted(unique_groups, key=lambda g: (-counts[g], g))
    fold_of: dict[str, int] = {}
    fold_sizes = [0] * n_splits
    for g in ordered:
        f = int(np.argmin(fold_sizes))
        fold_of[g] = f
        fold_sizes[f] += counts[g]

    all_idx = np.arange(grp.size)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        val_mask = np.array([fold_of[str(g)] == f for g in grp], dtype=bool)
        splits.append((all_idx[~val_mask], all_idx[val_mask]))
    return splits
