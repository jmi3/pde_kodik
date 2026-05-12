"""Tests for Laplacian kernels."""

import numpy as np

from pde_framework.calculation_core import laplacian_1d


def test_laplacian_constant_field_is_zero_interior() -> None:
    """Constant fields have zero Laplacian in the interior."""
    u = np.ones(7)
    result = laplacian_1d(u, 0.5)

    assert np.allclose(result[1:-1], 0.0)
    assert np.allclose(result[[0, -1]], 0.0)


def test_laplacian_linear_field_is_zero_interior() -> None:
    """Linear fields have zero Laplacian in the interior."""
    x = np.linspace(-1.0, 1.0, 9)
    result = laplacian_1d(x, x[1] - x[0])

    assert np.allclose(result[1:-1], 0.0, atol=1e-12)


def test_laplacian_quadratic_field_is_two_interior() -> None:
    """The quadratic field u=x^2 yields Laplacian approximately 2."""
    x = np.linspace(-1.0, 1.0, 9)
    u = x**2
    result = laplacian_1d(u, x[1] - x[0])

    assert np.allclose(result[1:-1], 2.0, atol=1e-12)


def test_laplacian_output_shape_matches_input() -> None:
    """Output shape matches the input shape."""
    u = np.array([1.0, 2.0, 3.0, 4.0])
    result = laplacian_1d(u, 1.0)

    assert result.shape == u.shape
