"""Tests for divergence kernels."""

import numpy as np

from pde_framework.calculation_core import divergence_1d


def test_divergence_constant_vector_field_is_zero() -> None:
    """Divergence of constant vector field is zero."""
    v = np.full(11, 2.0)

    result = divergence_1d(v, 0.1)

    assert np.allclose(result, 0.0)


def test_divergence_of_v_equals_x_is_one_interior() -> None:
    """For v=x, dv/dx is approximately 1 in interior."""
    x = np.linspace(-1.0, 1.0, 11)

    result = divergence_1d(x, x[1] - x[0])

    assert np.allclose(result[1:-1], 1.0, atol=1e-12)
    assert result.shape == x.shape
