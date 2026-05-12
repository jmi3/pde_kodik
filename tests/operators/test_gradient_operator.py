"""Tests for the GradientOperator implementation."""

import numpy as np

from pde_framework.field import ScalarField, VectorField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients import GradientOperator


def test_gradient_operator_returns_vector_field() -> None:
    """GradientOperator returns VectorField with one 1D component."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=9)
    field = ScalarField(grid)
    field.apply_function(lambda x: x)

    operator = GradientOperator(scheme="central")
    result = operator.apply(field)

    assert isinstance(result, VectorField)
    assert result.grid is field.grid
    assert len(result.components) == 1
    assert np.allclose(result.components[0][1:-1], 1.0, atol=1e-12)


def test_gradient_operator_repr_contains_scheme() -> None:
    """repr() exposes selected finite-difference scheme."""
    assert repr(GradientOperator(scheme="forward")) == "GradientOperator(scheme='forward')"
