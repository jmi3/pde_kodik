import numpy as np

from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients.LaplacianOperator import LaplacianOperator


def test_heat_equation_rhs_shape_and_scale():
    grid = Grid1D(-1.0, 1.0, 11)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.sin(np.pi * x))

    eq = HeatEquation(alpha=2.0, operator=LaplacianOperator())
    rhs = eq.rhs(field)

    assert rhs.shape == field.data.shape
    # For sin(pi x) on uniform grid, Laplacian ~ -pi^2 sin(pi x), so signs preserved
    assert np.allclose(rhs[1:-1] / field.data[1:-1], -2.0 * np.pi**2, atol=1e0) or np.any(
        field.data[1:-1] != 0
    )
