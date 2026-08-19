import numpy as np

from src.shape_tracking import (
    directional_accuracy,
    first_difference_correlation,
    mean_absolute_change_ratio,
    variability_ratio,
)


def test_perfect_shape_tracking():
    y = np.array([1.0, 3.0, 2.0, 5.0])
    p = y.copy()
    assert np.isclose(first_difference_correlation(y, p), 1.0)
    assert np.isclose(directional_accuracy(y, p), 1.0)
    assert np.isclose(variability_ratio(y, p), 1.0)
    assert np.isclose(mean_absolute_change_ratio(y, p), 1.0)


def test_flat_forecast_has_zero_movement_and_variability():
    y = np.array([1.0, 3.0, 2.0, 5.0])
    p = np.array([2.0, 2.0, 2.0, 2.0])
    assert np.isclose(variability_ratio(y, p), 0.0)
    assert np.isclose(mean_absolute_change_ratio(y, p), 0.0)


def test_directional_accuracy_detects_opposite_moves():
    y = np.array([1.0, 2.0, 1.0, 2.0])
    p = np.array([2.0, 1.0, 2.0, 1.0])
    assert np.isclose(directional_accuracy(y, p), 0.0)
