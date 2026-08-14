"""Data utilities for the UrbanVerse time-series PoC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metr_la(path: str | Path) -> pd.DataFrame:
    """Load the METR-LA HDF5 DataFrame and validate its time index."""
    frame = pd.read_hdf(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Expected METR-LA rows to use a pandas DatetimeIndex.")
    return frame.sort_index()


def get_sensor_series(frame: pd.DataFrame, sensor_id: str) -> pd.Series:
    """Return one sensor as a float time series."""
    if sensor_id not in frame.columns.astype(str):
        raise KeyError(f"Sensor {sensor_id!r} was not found in the dataset.")

    # Preserve compatibility when HDF5 columns are not stored as strings.
    column = next(col for col in frame.columns if str(col) == sensor_id)
    series = frame[column].astype(float).copy()
    series.name = sensor_id
    return series


def clean_window(series: pd.Series, end_position: int, context: int, horizon: int):
    """Return a clean context/target pair ending at ``end_position``.

    Windows containing missing values or non-positive traffic-speed values are
    rejected rather than silently imputed in the primary PoC.
    """
    start = end_position - context
    stop = end_position + horizon
    if start < 0 or stop > len(series):
        raise IndexError("Requested forecast window lies outside the series.")

    window = series.iloc[start:stop]
    if window.isna().any() or (window <= 0).any():
        raise ValueError("Forecast window contains missing or non-positive values.")

    history = window.iloc[:context]
    target = window.iloc[context:]
    return history, target
