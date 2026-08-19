"""Evaluation metrics shared by TimesFM and Moirai experiments."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error on all supplied values."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error on all supplied values."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def valid_target_mask(y_true, *, zero_is_invalid: bool = True) -> np.ndarray:
    """Boolean mask for valid target observations.

    METR-LA/DCRNN-style evaluation commonly treats target value 0 as null.  The
    revised PoC therefore masks invalid target points instead of deleting an
    entire forecast window because one point is invalid.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mask = np.isfinite(y_true)
    if zero_is_invalid:
        mask &= y_true > 0
    return mask


def masked_mae(y_true, y_pred, *, zero_is_invalid: bool = True) -> float:
    """MAE using only valid target points."""
    y_true, y_pred = _arrays(y_true, y_pred)
    mask = valid_target_mask(y_true, zero_is_invalid=zero_is_invalid) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def masked_rmse(y_true, y_pred, *, zero_is_invalid: bool = True) -> float:
    """RMSE using only valid target points."""
    y_true, y_pred = _arrays(y_true, y_pred)
    mask = valid_target_mask(y_true, zero_is_invalid=zero_is_invalid) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
