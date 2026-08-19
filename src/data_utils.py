"""Data utilities for the UrbanVerse time-series PoC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowQuality:
    context_invalid_fraction: float
    target_valid_fraction: float
    context_invalid_points: int
    target_invalid_points: int


def load_metr_la(path: str | Path) -> pd.DataFrame:
    """Load the METR-LA HDF5 DataFrame and validate its time index."""
    frame = pd.read_hdf(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Expected METR-LA rows to use a pandas DatetimeIndex.")

    # Some legacy METR-LA HDF5 files carry an old pandas ``freq`` metadata
    # attribute that modern pandas may deserialize as ``numpy.bytes_``.  The
    # timestamps themselves are valid, but resampling can then fail while
    # trying to inspect the broken frequency metadata.  Rebuilding the index
    # from its timestamp values strips that stale metadata without changing
    # any timestamp.
    frame.index = pd.DatetimeIndex(frame.index.to_numpy(copy=True), name=frame.index.name)

    if frame.index.has_duplicates:
        raise ValueError("METR-LA index contains duplicate timestamps.")
    return frame.sort_index()


def get_sensor_series(frame: pd.DataFrame, sensor_id: str) -> pd.Series:
    """Return one sensor as a float time series."""
    if sensor_id not in frame.columns.astype(str):
        raise KeyError(f"Sensor {sensor_id!r} was not found in the dataset.")
    column = next(col for col in frame.columns if str(col) == sensor_id)
    series = frame[column].astype(float).copy()
    series.name = sensor_id
    return series


def invalid_speed_mask(series: pd.Series) -> pd.Series:
    """Mask NaN/non-finite/non-positive readings as invalid for this PoC."""
    values = pd.to_numeric(series, errors="coerce")
    return values.isna() | ~np.isfinite(values.to_numpy()) | (values <= 0)


def prepare_context_target(
    series: pd.Series,
    end_position: int,
    context: int,
    horizon: int,
    *,
    max_context_invalid_fraction: float = 0.05,
    min_target_valid_fraction: float = 0.80,
):
    """Prepare one forecast window without whole-window zero deletion.

    Context invalid values are imputed *inside the observed context only* using
    time interpolation, then forward/backward fill if needed.  The future target
    is never imputed for scoring: invalid target points remain invalid and are
    masked by the evaluation metrics.

    This keeps a window with a small number of null measurements instead of
    discarding the whole example.
    """
    start = end_position - context
    stop = end_position + horizon
    if start < 0 or stop > len(series):
        raise IndexError("Requested forecast window lies outside the series.")

    history_raw = series.iloc[start:end_position].astype(float).copy()
    target = series.iloc[end_position:stop].astype(float).copy()

    history_invalid = invalid_speed_mask(history_raw)
    target_invalid = invalid_speed_mask(target)

    context_invalid_fraction = float(history_invalid.mean())
    target_valid_fraction = float((~target_invalid).mean())

    quality = WindowQuality(
        context_invalid_fraction=context_invalid_fraction,
        target_valid_fraction=target_valid_fraction,
        context_invalid_points=int(history_invalid.sum()),
        target_invalid_points=int(target_invalid.sum()),
    )

    if context_invalid_fraction > max_context_invalid_fraction:
        raise ValueError(
            f"Context invalid fraction {context_invalid_fraction:.3f} exceeds "
            f"limit {max_context_invalid_fraction:.3f}."
        )
    if target_valid_fraction < min_target_valid_fraction:
        raise ValueError(
            f"Target valid fraction {target_valid_fraction:.3f} is below "
            f"minimum {min_target_valid_fraction:.3f}."
        )

    history = history_raw.mask(history_invalid)
    if history.isna().any():
        # Every value used here is already in the observed pre-forecast context.
        if isinstance(history.index, pd.DatetimeIndex):
            history = history.interpolate(method="time", limit_direction="both")
        else:
            history = history.interpolate(limit_direction="both")
        history = history.ffill().bfill()
    if history.isna().any():
        raise ValueError("Context could not be imputed from observed context values.")

    return history, target, quality


def clean_window(series: pd.Series, end_position: int, context: int, horizon: int):
    """Compatibility wrapper returning context and target.

    Revised behavior: a small number of invalid points no longer deletes the
    entire window.  Use :func:`prepare_context_target` when quality metadata is
    needed.
    """
    history, target, _ = prepare_context_target(series, end_position, context, horizon)
    return history, target
