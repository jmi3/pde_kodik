"""Tests for the VectorField data structure."""

import numpy as np
import pytest

from pde_framework.field import VectorField
from pde_framework.grid import Grid1D


def test_vector_field_1d_single_component() -> None:
    """In 1D, VectorField accepts exactly one component."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    vector = VectorField(grid, [np.ones(grid.shape)])

    assert vector.grid is grid
    assert len(vector.components) == 1
    assert vector.components[0].shape == grid.shape


def test_vector_field_rejects_wrong_number_of_components() -> None:
    """1D VectorField rejects multi-component input."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)

    with pytest.raises(ValueError, match="exactly 1 component"):
        VectorField(grid, [np.ones(grid.shape), np.ones(grid.shape)])


def test_vector_field_rejects_shape_mismatch() -> None:
    """Each component must match grid shape."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)

    with pytest.raises(ValueError, match="does not match grid shape"):
        VectorField(grid, [np.ones(4)])


def test_vector_field_copy_is_deep_and_norm_positive() -> None:
    """copy() makes deep copy and norm() aggregates components."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    vector = VectorField(grid, [np.linspace(0.0, 1.0, grid.n_points)])

    copied = vector.copy()
    copied.components[0][0] = 123.0

    assert not np.isclose(vector.components[0][0], copied.components[0][0])
    assert vector.norm() > 0.0
