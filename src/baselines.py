"""Simple forecasting baselines used to judge whether foundation models add value."""

from __future__ import annotations

import numpy as np


def persistence_forecast(last_observation: float, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the whole forecast horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return np.full(int(horizon), float(last_observation), dtype=float)


def seasonal_naive_forecast(history, horizon: int, seasonality: int) -> np.ndarray:
    """Repeat values from one seasonal cycle ago.

    Example for 5-minute traffic with daily seasonality: seasonality=288.
    The history must contain at least ``seasonality`` observations.
    """
    history = np.asarray(history, dtype=float)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    if len(history) < seasonality:
        raise ValueError(
            f"history has {len(history)} points but seasonality={seasonality}; "
            "not enough history for seasonal naive"
        )
    cycle = history[-seasonality:]
    reps = int(np.ceil(horizon / seasonality))
    return np.tile(cycle, reps)[:horizon].astype(float)
