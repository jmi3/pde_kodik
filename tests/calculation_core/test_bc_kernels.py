"""Tests for boundary condition kernels."""

import numpy as np

from pde_framework.calculation_core import apply_dirichlet_1d


def test_apply_dirichlet_1d_modifies_boundaries_in_place() -> None:
    """Dirichlet kernel modifies only the boundaries in place."""
    u = np.array([1.0, 2.0, 3.0, 4.0])
    inner_before = u[1:-1].copy()

    apply_dirichlet_1d(u, left_value=-1.0, right_value=10.0)

    assert np.allclose(u[0], -1.0)
    assert np.allclose(u[-1], 10.0)
    assert np.allclose(u[1:-1], inner_before)


def test_apply_dirichlet_1d_handles_single_value_array() -> None:
    """Dirichlet kernel handles a degenerate single-point array."""
    u = np.array([2.5])

    apply_dirichlet_1d(u, left_value=1.0, right_value=3.0)

    assert np.allclose(u[0], 1.0)
