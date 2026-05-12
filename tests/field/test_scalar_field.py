"""Tests for the ScalarField data structure."""

import numpy as np
import pytest

from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D


def test_scalar_field_defaults_to_zeros() -> None:
    """ScalarField initializes with zeros when no data is provided."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=4)
    field = ScalarField(grid)

    assert field.grid is grid
    assert field.data.shape == grid.shape
    assert np.allclose(field.data, 0.0)


def test_scalar_field_rejects_shape_mismatch() -> None:
    """ScalarField rejects data with an incompatible shape."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=4)

    with pytest.raises(ValueError, match="does not match grid shape"):
        ScalarField(grid, data=np.zeros(3))


def test_scalar_field_copy_is_deep() -> None:
    """ScalarField.copy returns a deep copy of the data."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=4)
    field = ScalarField(grid)
    field.fill(2.0)

    copied = field.copy()
    copied.fill(5.0)

    assert np.allclose(field.data, 2.0)
    assert np.allclose(copied.data, 5.0)


def test_scalar_field_apply_function() -> None:
    """ScalarField.apply_function evaluates a function on grid coordinates."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=5)
    field = ScalarField(grid)

    field.apply_function(lambda x: x**2)

    assert np.allclose(field.data, grid.x**2)


def test_scalar_field_arithmetic_and_scalar_multiply() -> None:
    """ScalarField arithmetic returns new compatible fields."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=4)
    left = ScalarField(grid, data=np.ones(grid.shape))
    right = ScalarField(grid, data=np.full(grid.shape, 2.0))

    summed = left + right
    diff = right - left
    scaled = 3.0 * left

    assert np.allclose(summed.data, 3.0)
    assert np.allclose(diff.data, 1.0)
    assert np.allclose(scaled.data, 3.0)
    assert summed.grid is grid


def test_scalar_field_norm_positive_for_nonzero_data() -> None:
    """ScalarField.norm returns a positive norm for non-zero data."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=4)
    field = ScalarField(grid, data=np.array([0.0, 1.0, 0.0, 0.0]))

    assert field.norm() > 0.0


def test_scalar_field_add_rejects_different_grid_shapes() -> None:
    """Binary addition should fail for fields on different grid shapes."""
    grid_left = Grid1D(x_min=0.0, x_max=1.0, n_points=4)
    grid_right = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    left = ScalarField(grid_left, data=np.ones(grid_left.shape))
    right = ScalarField(grid_right, data=np.ones(grid_right.shape))

    with pytest.raises(ValueError, match="left shape"):
        _ = left + right


def test_scalar_field_sub_rejects_different_grid_coordinates() -> None:
    """Binary subtraction should fail for fields on different coordinates."""
    grid_left = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    grid_right = Grid1D(x_min=0.0, x_max=2.0, n_points=5)
    left = ScalarField(grid_left, data=np.ones(grid_left.shape))
    right = ScalarField(grid_right, data=np.ones(grid_right.shape))

    with pytest.raises(ValueError, match="coordinates differ"):
        _ = left - right
