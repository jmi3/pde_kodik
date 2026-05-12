import numpy as np

from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.integrators.clients.ExplicitEulerIntegrator import ExplicitEulerIntegrator


def test_explicit_euler_steps_and_bc_applied():
    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    bc = None
    eq = HeatEquation(alpha=0.1, bc=bc)
    integrator = ExplicitEulerIntegrator()

    new_field = integrator.step(field, eq, dt=0.01)
    assert new_field.data.shape == field.data.shape
    # basic sanity: values changed
    assert not np.allclose(new_field.data, field.data)
