"""Tests for the DivergenceOperator implementation."""

import numpy as np

from pde_framework.field import ScalarField, VectorField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients import DivergenceOperator


def test_divergence_operator_returns_scalar_field() -> None:
    """DivergenceOperator returns ScalarField on same grid."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=9)
    vector = VectorField(grid, [grid.x.copy()])

    operator = DivergenceOperator()
    result = operator.apply(vector)

    assert isinstance(result, ScalarField)
    assert result.grid is grid
    assert np.allclose(result.data[1:-1], 1.0, atol=1e-12)


def test_divergence_operator_repr() -> None:
    """repr() is stable and informative."""
    assert repr(DivergenceOperator()) == "DivergenceOperator()"
