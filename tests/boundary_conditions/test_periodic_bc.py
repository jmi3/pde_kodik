import numpy as np

from pde_framework.boundary_conditions.clients import PeriodicBC
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D


def test_periodic_copies_neighbors():
    grid = Grid1D(0.0, 1.0, 6)
    u = ScalarField(grid)
    # make interior values distinct
    vals = np.arange(grid.n_points, dtype=float)
    u._data = vals.copy()

    bc = PeriodicBC()
    bc.apply(u.data, grid)

    # after applying, left boundary should equal second-to-last, right boundary equals second
    assert np.allclose(u.data[0], u.data[-2])
    assert np.allclose(u.data[-1], u.data[1])

