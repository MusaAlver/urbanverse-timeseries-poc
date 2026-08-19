
from __future__ import annotations

import numpy as np


def _as_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values.")
    return arr


def pearson_level_correlation(y_true, y_pred) -> float:
    """Correlation between forecast levels and ground-truth levels."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    if np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def first_difference_correlation(y_true, y_pred) -> float:
    """Correlation between one-step changes in truth and prediction."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    dy = np.diff(y)
    dp = np.diff(p)
    if np.std(dy) == 0 or np.std(dp) == 0:
        return float("nan")
    return float(np.corrcoef(dy, dp)[0, 1])


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of non-flat ground-truth changes whose direction is matched."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    dy = np.diff(y)
    dp = np.diff(p)
    mask = np.sign(dy) != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(dp[mask]) == np.sign(dy[mask])))


def variability_ratio(y_true, y_pred) -> float:
    """Forecast standard deviation divided by ground-truth standard deviation."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    denom = np.std(y, ddof=1)
    if denom == 0:
        return float("nan")
    return float(np.std(p, ddof=1) / denom)


def range_ratio(y_true, y_pred) -> float:
    """Forecast range divided by ground-truth range."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    denom = float(np.max(y) - np.min(y))
    if denom == 0:
        return float("nan")
    return float((np.max(p) - np.min(p)) / denom)


def mean_absolute_change_ratio(y_true, y_pred) -> float:
    """Average absolute forecast movement divided by average absolute true movement."""
    y = _as_1d(y_true)
    p = _as_1d(y_pred)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length.")
    denom = float(np.mean(np.abs(np.diff(y))))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(np.diff(p))) / denom)


def shape_tracking_metrics(y_true, y_pred) -> dict[str, float]:
    """Diagnostic metrics for whether a point forecast tracks trajectory shape.

    These metrics complement MAE/RMSE. They are not replacements for primary
    forecast-accuracy metrics and should be interpreted together.
    """
    return {
        "level_correlation": pearson_level_correlation(y_true, y_pred),
        "first_difference_correlation": first_difference_correlation(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "variability_ratio": variability_ratio(y_true, y_pred),
        "range_ratio": range_ratio(y_true, y_pred),
        "mean_absolute_change_ratio": mean_absolute_change_ratio(y_true, y_pred),
    }


def masked_shape_tracking_metrics(y_true, y_pred) -> dict[str, float]:
    """Shape diagnostics while respecting invalid target points.

    Level/range/variability metrics use all valid target-prediction pairs.
    Change/direction metrics use only *consecutive* pairs for which both target
    timestamps are valid, avoiding artificial differences across missing gaps.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    valid = np.isfinite(y) & (y > 0) & np.isfinite(p)
    if valid.sum() < 2:
        return {
            "level_correlation": float("nan"),
            "first_difference_correlation": float("nan"),
            "directional_accuracy": float("nan"),
            "variability_ratio": float("nan"),
            "range_ratio": float("nan"),
            "mean_absolute_change_ratio": float("nan"),
        }

    yv, pv = y[valid], p[valid]
    level = pearson_level_correlation(yv, pv)
    var_ratio = variability_ratio(yv, pv)
    r_ratio = range_ratio(yv, pv)

    pair_valid = valid[1:] & valid[:-1]
    dy = np.diff(y)[pair_valid]
    dp = np.diff(p)[pair_valid]

    if len(dy) < 2 or np.std(dy) == 0 or np.std(dp) == 0:
        diff_corr = float("nan")
    else:
        diff_corr = float(np.corrcoef(dy, dp)[0, 1])

    nonflat = np.sign(dy) != 0
    direction = float(np.mean(np.sign(dp[nonflat]) == np.sign(dy[nonflat]))) if np.any(nonflat) else float("nan")

    denom = float(np.mean(np.abs(dy))) if len(dy) else 0.0
    change_ratio = float(np.mean(np.abs(dp)) / denom) if denom > 0 else float("nan")

    return {
        "level_correlation": level,
        "first_difference_correlation": diff_corr,
        "directional_accuracy": direction,
        "variability_ratio": var_ratio,
        "range_ratio": r_ratio,
        "mean_absolute_change_ratio": change_ratio,
    }
def pooled_within_window_shape_metrics(windows) -> dict[str, float]:
    """Pool shape information across windows without creating fake boundaries.

    First differences are computed inside each window and then pooled.
    Variability is pooled after centering each valid window around its own mean.
    """
    pooled_dy, pooled_dp, centered_y, centered_p = [], [], [], []
    for y_true, y_pred in windows:
        y = np.asarray(y_true, dtype=float).reshape(-1)
        p = np.asarray(y_pred, dtype=float).reshape(-1)
        if y.shape != p.shape:
            raise ValueError("Each y_true/y_pred window pair must have the same shape.")
        valid = np.isfinite(y) & (y > 0) & np.isfinite(p)
        if valid.sum() >= 2:
            yv, pv = y[valid], p[valid]
            centered_y.append(yv - np.mean(yv))
            centered_p.append(pv - np.mean(pv))
        pair_valid = valid[1:] & valid[:-1]
        if np.any(pair_valid):
            pooled_dy.append(np.diff(y)[pair_valid])
            pooled_dp.append(np.diff(p)[pair_valid])
    if not pooled_dy or not centered_y:
        return {
            "first_difference_correlation": float("nan"),
            "directional_accuracy": float("nan"),
            "variability_ratio": float("nan"),
            "mean_absolute_change_ratio": float("nan"),
            "n_valid_differences": 0,
            "n_centered_level_pairs": 0,
        }
    dy, dp = np.concatenate(pooled_dy), np.concatenate(pooled_dp)
    cy, cp = np.concatenate(centered_y), np.concatenate(centered_p)
    diff_corr = float(np.corrcoef(dy, dp)[0, 1]) if len(dy) >= 2 and np.std(dy) > 0 and np.std(dp) > 0 else float("nan")
    nonflat = np.sign(dy) != 0
    direction = float(np.mean(np.sign(dp[nonflat]) == np.sign(dy[nonflat]))) if np.any(nonflat) else float("nan")
    true_std = np.std(cy, ddof=1) if len(cy) >= 2 else 0.0
    pred_std = np.std(cp, ddof=1) if len(cp) >= 2 else 0.0
    var_ratio = float(pred_std / true_std) if true_std > 0 else float("nan")
    denom = float(np.mean(np.abs(dy))) if len(dy) else 0.0
    change_ratio = float(np.mean(np.abs(dp)) / denom) if denom > 0 else float("nan")
    return {
        "first_difference_correlation": diff_corr,
        "directional_accuracy": direction,
        "variability_ratio": var_ratio,
        "mean_absolute_change_ratio": change_ratio,
        "n_valid_differences": int(len(dy)),
        "n_centered_level_pairs": int(len(cy)),
    }

