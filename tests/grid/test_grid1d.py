"""Tests for the Grid1D data structure."""

import numpy as np
import pytest

from pde_framework.grid import Grid1D


def test_grid1d_constructs_uniform_coordinates() -> None:
    """Grid1D creates uniformly spaced coordinates with the expected shape."""
    grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=5)

    assert grid.shape == (5,)
    assert grid.n_points == 5
    assert np.allclose(grid.x, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert grid.dx == pytest.approx(0.5)


def test_grid1d_interior_slice() -> None:
    """Grid1D exposes the expected interior slice."""
    grid = Grid1D(x_min=0.0, x_max=2.0, n_points=4)

    assert grid.interior_slice == slice(1, -1)


def test_grid1d_rejects_too_few_points() -> None:
    """Grid1D rejects grids with fewer than two points."""
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        Grid1D(x_min=0.0, x_max=1.0, n_points=1)


def test_grid1d_rejects_non_increasing_domain() -> None:
    """Grid1D rejects domains where x_max is not greater than x_min."""
    with pytest.raises(ValueError, match="x_max must be greater than x_min"):
        Grid1D(x_min=1.0, x_max=1.0, n_points=2)
