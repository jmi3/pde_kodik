"""Tests for array utility helpers."""

import numpy as np

from pde_framework.utils import gaussian_1d


def test_gaussian_1d_shape_and_nonnegativity() -> None:
    """Gaussian helper returns the expected shape and non-negative values."""
    x = np.linspace(-1.0, 1.0, 101)
    values = gaussian_1d(x, center=0.0, sigma=0.2)

    assert values.shape == x.shape
    assert np.all(values >= 0.0)


def test_gaussian_1d_peak_near_center() -> None:
    """Gaussian helper peaks near the requested center."""
    x = np.linspace(-1.0, 1.0, 101)
    values = gaussian_1d(x, center=0.0, sigma=0.2)

    peak_index = int(np.argmax(values))
    assert x[peak_index] == 0.0


def test_gaussian_1d_is_symmetric_about_center() -> None:
    """Gaussian helper is symmetric on a centered grid."""
    x = np.linspace(-1.0, 1.0, 11)
    values = gaussian_1d(x, center=0.0, sigma=0.3)

    assert np.allclose(values, values[::-1])
