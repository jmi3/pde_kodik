"""Tests for the GradientOperator implementation."""

import numpy as np

from pde_framework.field import ScalarField, VectorField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients import GradientOperator


def test_gradient_operator_returns_scalar_field_on_1d_grid() -> None:
    """GradientOperator returns VectorField with one 1D component."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=9)
    field = ScalarField(grid)
    field.apply_function(lambda x: x)

    operator = GradientOperator(scheme="central")
    result = operator.apply(field)

    assert isinstance(result, ScalarField)
    assert result.grid is field.grid
    assert np.allclose(result.data, 1.0, atol=1e-12)


def test_gradient_operator_repr_contains_scheme() -> None:
    """repr() exposes selected finite-difference scheme."""
    assert repr(GradientOperator(scheme="forward")) == "GradientOperator(scheme='forward')"
