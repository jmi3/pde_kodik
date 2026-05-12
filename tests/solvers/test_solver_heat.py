import numpy as np

from pde_framework.boundary_conditions.clients.DirichletBC import DirichletBC
from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.integrators.clients.ExplicitEulerIntegrator import ExplicitEulerIntegrator
from pde_framework.solvers.Solver import Solver


def test_solver_runs_and_enforces_dirichlet_bc():
    grid = Grid1D(-1.0, 1.0, 31)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-5.0 * x**2))

    bc = DirichletBC(0.0, 0.0)
    eq = HeatEquation(alpha=0.05, bc=bc)
    integrator = ExplicitEulerIntegrator()
    solver = Solver(integrator, eq)

    snapshots = solver.run(field, dt=0.005, n_steps=10, snapshot_stride=5)
    assert len(snapshots) >= 2
    # final snapshot should have zero boundaries
    last = snapshots[-1]
    assert np.isclose(last.data[0], 0.0)
    assert np.isclose(last.data[-1], 0.0)
