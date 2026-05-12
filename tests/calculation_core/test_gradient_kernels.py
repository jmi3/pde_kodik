"""Tests for gradient kernels."""

import numpy as np

from pde_framework.calculation_core import gradient_1d_backward, gradient_1d_central, gradient_1d_forward


def test_gradient_constant_is_zero() -> None:
    """Gradient of a constant field is zero."""
    u = np.full(9, 3.0)

    result = gradient_1d_central(u, 0.25)

    assert np.allclose(result, 0.0)


def test_gradient_linear_is_constant_derivative() -> None:
    """Gradient of u=x is 1 for central differences."""
    x = np.linspace(-1.0, 1.0, 9)

    result = gradient_1d_central(x, x[1] - x[0])

    assert np.allclose(result[1:-1], 1.0, atol=1e-12)


def test_forward_and_backward_have_expected_shape() -> None:
    """Forward/backward gradient kernels preserve input shape."""
    x = np.linspace(0.0, 1.0, 6)
    u = x**2

    forward = gradient_1d_forward(u, x[1] - x[0])
    backward = gradient_1d_backward(u, x[1] - x[0])

    assert forward.shape == u.shape
    assert backward.shape == u.shape
