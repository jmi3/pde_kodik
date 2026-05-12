"""Tests for the Laplacian operator implementation."""

import numpy as np

from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients import LaplacianOperator


def test_laplacian_operator_returns_new_field() -> None:
    """LaplacianOperator returns a new ScalarField on the same grid."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=9)
    field = ScalarField(grid)
    field.apply_function(lambda x: x**2)
    operator = LaplacianOperator()

    result = operator.apply(field)

    assert result.grid is field.grid
    assert result is not field
    assert np.allclose(result.data[1:-1], 2.0, atol=1e-12)
    assert np.allclose(field.data, grid.x**2)


def test_laplacian_operator_repr() -> None:
    """repr() is stable and informative."""
    assert repr(LaplacianOperator()) == "LaplacianOperator()"
