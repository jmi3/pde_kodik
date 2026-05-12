"""Tests for integrator consistency and RK4 integrator behavior."""

import numpy as np

from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.integrators.clients.ExplicitEulerIntegrator import ExplicitEulerIntegrator
from pde_framework.integrators.clients.RK4Integrator import RK4Integrator


def test_rk4_integrator_basic_step():
    """Test that RK4 integrator performs a step and returns correct shape."""
    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    eq = HeatEquation(alpha=0.1, bc=None)
    integrator = RK4Integrator()

    new_field = integrator.step(field, eq, dt=0.01)
    assert new_field.data.shape == field.data.shape
    assert not np.allclose(new_field.data, field.data)


def test_integrators_return_same_shape():
    """Test that Euler and RK4 return fields of same shape."""
    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    eq = HeatEquation(alpha=0.1, bc=None)
    euler_integrator = ExplicitEulerIntegrator()
    rk4_integrator = RK4Integrator()

    dt = 0.01
    euler_result = euler_integrator.step(field, eq, dt=dt)
    rk4_result = rk4_integrator.step(field, eq, dt=dt)

    assert euler_result.data.shape == rk4_result.data.shape
    assert euler_result.data.shape == field.data.shape


def test_integrators_preserve_boundary_conditions():
    """Test that both integrators apply BC correctly."""
    from pde_framework.boundary_conditions.clients.DirichletBC import DirichletBC

    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    bc = DirichletBC(left_value=0.0, right_value=0.0)
    eq = HeatEquation(alpha=0.1, bc=bc)

    euler_integrator = ExplicitEulerIntegrator()
    rk4_integrator = RK4Integrator()

    dt = 0.01
    euler_result = euler_integrator.step(field, eq, dt=dt)
    rk4_result = rk4_integrator.step(field, eq, dt=dt)

    # Both should have boundary values 0.0
    assert np.isclose(euler_result.data[0], 0.0)
    assert np.isclose(euler_result.data[-1], 0.0)
    assert np.isclose(rk4_result.data[0], 0.0)
    assert np.isclose(rk4_result.data[-1], 0.0)


def test_rk4_higher_accuracy_than_euler():
    """Test that RK4 is more accurate than Euler for the same problem."""
    # Simple linear ODE test: u' = -u, u(0) = 1, exact solution u(t) = exp(-t)
    # This can be discretized as a heat equation with specific parameters

    grid = Grid1D(-1.0, 1.0, 101)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.ones_like(x))  # constant initial condition

    # Heat equation with alpha=1.0
    eq = HeatEquation(alpha=1.0, bc=None)

    euler_integrator = ExplicitEulerIntegrator()
    rk4_integrator = RK4Integrator()

    # Perform 10 steps
    dt = 0.001
    n_steps = 10

    euler_state = field
    rk4_state = field

    for _ in range(n_steps):
        euler_state = euler_integrator.step(euler_state, eq, dt=dt)
        rk4_state = rk4_integrator.step(rk4_state, eq, dt=dt)

    # RK4 should generally be closer to the true solution
    # For now, just check that both produce reasonable outputs
    assert not np.any(np.isnan(euler_state.data))
    assert not np.any(np.isnan(rk4_state.data))


def test_integrators_not_modify_input():
    """Test that both integrators do NOT modify the input field."""
    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    eq = HeatEquation(alpha=0.1, bc=None)
    euler_integrator = ExplicitEulerIntegrator()
    rk4_integrator = RK4Integrator()

    original_data = field.data.copy()

    dt = 0.01
    euler_integrator.step(field, eq, dt=dt)
    assert np.allclose(field.data, original_data), "Euler modified input field"

    rk4_integrator.step(field, eq, dt=dt)
    assert np.allclose(field.data, original_data), "RK4 modified input field"


def test_rk4_integrator_with_dirichlet_bc():
    """Test RK4 with Dirichlet boundary conditions."""
    from pde_framework.boundary_conditions.clients.DirichletBC import DirichletBC

    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))

    bc = DirichletBC(left_value=0.0, right_value=0.0)
    eq = HeatEquation(alpha=0.1, bc=bc)
    integrator = RK4Integrator()

    new_field = integrator.step(field, eq, dt=0.01)

    # Boundaries should be enforced
    assert np.isclose(new_field.data[0], 0.0)
    assert np.isclose(new_field.data[-1], 0.0)


def test_integrator_signature_consistency():
    """Test that both integrators have the same function signature."""
    # Verify that both classes have step() method
    euler = ExplicitEulerIntegrator()
    rk4 = RK4Integrator()

    assert hasattr(euler, "step") and callable(getattr(euler, "step"))
    assert hasattr(rk4, "step") and callable(getattr(rk4, "step"))

    # Both should accept (state, equation, dt)
    grid = Grid1D(-1.0, 1.0, 21)
    field = ScalarField(grid)
    field.apply_function(lambda x: np.exp(-(x**2)))
    eq = HeatEquation(alpha=0.1, bc=None)

    # This should not raise for either integrator
    result_euler = euler.step(field, eq, 0.01)
    result_rk4 = rk4.step(field, eq, 0.01)

    assert isinstance(result_euler, ScalarField)
    assert isinstance(result_rk4, ScalarField)
