"""Tests for time stepping kernels."""

import numpy as np

from pde_framework.calculation_core import euler_step_1d


def test_euler_step_1d_returns_new_state() -> None:
    """Euler kernel returns u + dt * rhs without mutating input."""
    u = np.array([1.0, 2.0, 3.0])
    rhs = np.array([0.5, -1.0, 2.0])

    result = euler_step_1d(u, rhs, 0.2)

    assert np.allclose(result, np.array([1.1, 1.8, 3.4]))
    assert np.allclose(u, np.array([1.0, 2.0, 3.0]))


def test_euler_step_1d_constant_rhs() -> None:
    """Euler kernel works for constant RHS fields."""
    u = np.zeros(4)
    rhs = np.ones(4)

    result = euler_step_1d(u, rhs, 0.5)

    assert np.allclose(result, 0.5)
