"""Tests for the Dirichlet boundary-condition implementation."""

import numpy as np

from pde_framework.boundary_conditions.clients import DirichletBC
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D


def test_dirichlet_bc_applies_boundaries_in_place() -> None:
    """DirichletBC modifies only the boundary values."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    field = ScalarField(grid, data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    bc = DirichletBC(left_value=-1.0, right_value=10.0)

    bc.apply(field.data, field.grid)

    assert np.isclose(field.data[0], -1.0)
    assert np.isclose(field.data[-1], 10.0)
    assert np.allclose(field.data[1:-1], np.array([2.0, 3.0, 4.0]))


def test_dirichlet_bc_repr_contains_values() -> None:
    """repr() includes the boundary values."""
    bc = DirichletBC(left_value=0.0, right_value=1.5)

    assert "0.0" in repr(bc)
    assert "1.5" in repr(bc)
