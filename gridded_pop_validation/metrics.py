"""Accuracy metrics, Theil MSE decomposition, and change-tracking diagnostics
for comparing gridded population aggregates against census benchmarks.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _to_1d_float(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-tie rank transform implemented in pure NumPy."""
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    n = arr.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def theil_mse_decomposition(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    """Decompose MSE into bias, variance, and covariance (unsystematic) components.

    Identity (using population standard deviations, ddof=0):
        MSE = (mean_pred - mean_obs)^2
            + (std_pred - std_obs)^2
            + 2 * (1 - r) * std_pred * std_obs
    """
    obs = _to_1d_float(observed, "observed")
    pred = _to_1d_float(predicted, "predicted")
    if obs.size != pred.size:
        raise ValueError("observed and predicted must have identical length")

    diff = pred - obs
    mse = float(np.mean(diff * diff))
    bias_comp = float((np.mean(pred) - np.mean(obs)) ** 2)
    std_obs = float(np.std(obs, ddof=0))
    std_pred = float(np.std(pred, ddof=0))
    var_comp = float((std_pred - std_obs) ** 2)
    r = _pearson(obs, pred)
    if np.isnan(r):
        cov_comp = max(0.0, mse - bias_comp - var_comp)
    else:
        cov_comp = float(2.0 * (1.0 - r) * std_pred * std_obs)

    if mse == 0.0:
        bias_share = var_share = cov_share = 0.0
    else:
        bias_share = bias_comp / mse
        var_share = var_comp / mse
        cov_share = cov_comp / mse

    return {
        "mse": mse,
        "bias_component": bias_comp,
        "variance_component": var_comp,
        "covariance_component": cov_comp,
        "bias_share": bias_share,
        "variance_share": var_share,
        "covariance_share": cov_share,
    }


def accuracy_summary(
    observed: Sequence[float],
    predicted: Sequence[float],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute core level-accuracy metrics for gridded vs. census population counts."""
    obs = _to_1d_float(observed, "observed")
    pred = _to_1d_float(predicted, "predicted")
    if obs.size != pred.size:
        raise ValueError("observed and predicted must have identical length")
    if np.any(obs < 0) or np.any(pred < 0):
        raise ValueError("population counts must be non-negative")

    err = pred - obs
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err * err)))

    pos = obs > eps
    if np.any(pos):
        ape = abs_err[pos] / obs[pos]
        mape = float(np.mean(ape) * 100.0)
        medape = float(np.median(ape) * 100.0)
    else:
        mape = float("nan")
        medape = float("nan")

    denom_smape = np.abs(obs) + np.abs(pred)
    valid_smape = denom_smape > eps
    smape = (
        float(np.mean(2.0 * abs_err[valid_smape] / denom_smape[valid_smape]) * 100.0)
        if np.any(valid_smape)
        else float("nan")
    )

    log_err = np.log1p(pred) - np.log1p(obs)
    mle = float(np.mean(log_err))
    rmsle = float(np.sqrt(np.mean(log_err * log_err)))

    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / ss_tot) if ss_tot > 0.0 else float("nan")

    obs_sum = float(np.sum(obs))
    total_rel_bias = (
        float((np.sum(pred) - obs_sum) / obs_sum * 100.0) if obs_sum > eps else float("nan")
    )

    decomp = theil_mse_decomposition(obs, pred)
    return {
        "n": int(obs.size),
        "obs_total": obs_sum,
        "pred_total": float(np.sum(pred)),
        "total_relative_bias_pct": total_rel_bias,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "medape_pct": medape,
        "smape_pct": smape,
        "mean_log_error": mle,
        "rmsle": rmsle,
        "r2": r2,
        "pearson_r": _pearson(obs, pred),
        "spearman_rho": _spearman(obs, pred),
        "mse_bias_share": decomp["bias_share"],
        "mse_variance_share": decomp["variance_share"],
        "mse_covariance_share": decomp["covariance_share"],
    }


def evaluate_by_group(
    observed: Sequence[float],
    predicted: Sequence[float],
    groups: Sequence[Any],
) -> dict[str, dict[str, float]]:
    """Compute accuracy summaries separately for each stratum (e.g., urban/rural, shrinking/growing)."""
    obs = _to_1d_float(observed, "observed")
    pred = _to_1d_float(predicted, "predicted")
    grp = np.asarray(groups)
    if not (obs.size == pred.size == grp.size):
        raise ValueError("observed, predicted, and groups must have identical length")

    results: dict[str, dict[str, float]] = {}
    for label in sorted({str(g) for g in grp}):
        mask = np.array([str(g) == label for g in grp], dtype=bool)
        results[label] = accuracy_summary(obs[mask], pred[mask])
    return results


def evaluate_population_change(
    obs_t0: Sequence[float],
    obs_t1: Sequence[float],
    pred_t0: Sequence[float],
    pred_t1: Sequence[float],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Evaluate how faithfully gridded population products track intercensal population change.

    Compares both absolute differences (dP) and log growth rates (log(P_t1 / P_t0))
    and measures directional agreement (growth vs. shrinkage classification).
    """
    o0 = _to_1d_float(obs_t0, "obs_t0")
    o1 = _to_1d_float(obs_t1, "obs_t1")
    p0 = _to_1d_float(pred_t0, "pred_t0")
    p1 = _to_1d_float(pred_t1, "pred_t1")
    if not (o0.size == o1.size == p0.size == p1.size):
        raise ValueError("all four sequences must have identical length")

    d_obs = o1 - o0
    d_pred = p1 - p0
    err = d_pred - d_obs

    valid_log = (o0 > eps) & (o1 > eps) & (p0 > eps) & (p1 > eps)
    if np.any(valid_log):
        g_obs = np.log(o1[valid_log] / o0[valid_log])
        g_pred = np.log(p1[valid_log] / p0[valid_log])
        growth_r = _pearson(g_obs, g_pred)
        growth_rho = _spearman(g_obs, g_pred)
        growth_mae = float(np.mean(np.abs(g_pred - g_obs)))
        growth_rmse = float(np.sqrt(np.mean((g_pred - g_obs) ** 2)))
    else:
        growth_r = growth_rho = growth_mae = growth_rmse = float("nan")

    nonzero = np.abs(d_obs) > eps
    direction_agreement = (
        float(np.mean(np.sign(d_obs[nonzero]) == np.sign(d_pred[nonzero])) * 100.0)
        if np.any(nonzero)
        else float("nan")
    )

    shrinking_obs = d_obs < -eps
    shrinking_pred = d_pred < -eps
    tp = int(np.sum(shrinking_obs & shrinking_pred))
    fn = int(np.sum(shrinking_obs & (~shrinking_pred)))
    fp = int(np.sum((~shrinking_obs) & shrinking_pred))
    shrink_recall = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else float("nan")
    shrink_precision = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else float("nan")

    return {
        "n": int(o0.size),
        "change_mae": float(np.mean(np.abs(err))),
        "change_rmse": float(np.sqrt(np.mean(err * err))),
        "change_pearson_r": _pearson(d_obs, d_pred),
        "change_spearman_rho": _spearman(d_obs, d_pred),
        "log_growth_pearson_r": growth_r,
        "log_growth_spearman_rho": growth_rho,
        "log_growth_mae": growth_mae,
        "log_growth_rmse": growth_rmse,
        "direction_agreement_pct": direction_agreement,
        "shrinkage_recall_pct": shrink_recall,
        "shrinkage_precision_pct": shrink_precision,
    }
