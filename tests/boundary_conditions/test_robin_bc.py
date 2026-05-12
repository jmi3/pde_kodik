import numpy as np

from pde_framework.boundary_conditions.clients import RobinBC
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D


def test_robin_reduces_to_dirichlet_and_neumann():
    grid = Grid1D(0.0, 1.0, 5)
    u = ScalarField(grid)
    # set interior to known values
    u._data = np.linspace(0.0, 1.0, grid.n_points)

    dx = grid.dx

    # Dirichlet equivalence on left: b=0 -> a*u = c -> u0 = c/a
    bc_dir = RobinBC(left_a=1.0, left_b=0.0, left_c=2.0, right_a=1.0, right_b=0.0, right_c=3.0)
    bc_dir.apply(u.data, grid)
    assert np.allclose(u.data[0], 2.0)

    # Neumann equivalence (left): a=0, b=1, c=0 -> zero gradient -> u0 == u1
    u._data = np.linspace(0.0, 1.0, grid.n_points)
    bc_neu = RobinBC(left_a=0.0, left_b=1.0, left_c=0.0, right_a=1.0, right_b=0.0, right_c=0.0)
    bc_neu.apply(u.data, grid)
    assert np.allclose(u.data[0], u.data[1])

