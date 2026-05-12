import numpy as np

from pde_framework.boundary_conditions.clients import NeumannBC
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D


def test_neumann_zero_gradient():
    grid = Grid1D(0.0, 1.0, 5)
    u = ScalarField(grid)
    # set interior to a linear ramp
    u.apply_function(lambda x: x)

    # apply zero-gradient Neumann BC: boundaries should equal adjacent interior
    bc = NeumannBC(0.0, 0.0)
    bc.apply(u.data, grid)

    assert np.allclose(u.data[0], u.data[1])
    assert np.allclose(u.data[-1], u.data[-2])

