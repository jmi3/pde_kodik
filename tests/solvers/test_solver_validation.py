"""Tests for Solver validation behavior introduced in Epic 6."""

from __future__ import annotations

import numpy as np
import pytest

from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.integrators.clients.ExplicitEulerIntegrator import ExplicitEulerIntegrator
from pde_framework.integrators.IIntegrator import IIntegrator
from pde_framework.operators.clients.LaplacianOperator import LaplacianOperator
from pde_framework.solvers.Solver import Solver


class EchoEquation(IEquation):
    """Equation that returns zero RHS and records hook calls."""

    def __init__(self) -> None:
        self.hook_called = False

    def rhs(self, state: ScalarField) -> np.ndarray:
        return np.zeros_like(state.data)

    def validate_time_step(self, state: ScalarField, dt: float) -> bool:
        self.hook_called = True
        return True


class CopyIntegrator(IIntegrator):
    """Integrator that returns a copy without evolving state."""

    def step(self, state: ScalarField, equation: IEquation, dt: float) -> ScalarField:
        return state.copy()


def _make_state() -> ScalarField:
    grid = Grid1D(0.0, 1.0, 5)
    return ScalarField(grid, data=np.linspace(0.0, 1.0, grid.n_points))


def test_solver_rejects_non_positive_dt() -> None:
    """Solver should reject dt <= 0."""
    solver = Solver(CopyIntegrator(), EchoEquation())

    with pytest.raises(ValueError, match="dt must be positive"):
        solver.run(_make_state(), dt=0.0, n_steps=1)


def test_solver_rejects_non_positive_t_end() -> None:
    """Solver should reject t_end <= 0."""
    solver = Solver(CopyIntegrator(), EchoEquation())

    with pytest.raises(ValueError, match="t_end must be positive"):
        solver.run(_make_state(), dt=0.1, t_end=0.0)


def test_solver_rejects_invalid_save_every() -> None:
    """Solver should reject save_every < 1."""
    solver = Solver(CopyIntegrator(), EchoEquation())

    with pytest.raises(ValueError, match="save_every must be at least 1"):
        solver.run(_make_state(), dt=0.1, n_steps=1, save_every=0)


def test_solver_invokes_equation_validation_hook_when_available() -> None:
    """Solver should call equation.validate_time_step if present."""
    equation = EchoEquation()
    solver = Solver(CopyIntegrator(), equation)

    snapshots = solver.run(_make_state(), dt=0.1, n_steps=2, save_every=1)

    assert equation.hook_called is True
    assert len(snapshots) == 3


def test_solver_can_derive_n_steps_from_t_end() -> None:
    """Solver should support t_end-driven run configuration."""
    solver = Solver(CopyIntegrator(), EchoEquation())

    snapshots = solver.run(_make_state(), dt=0.1, t_end=0.25, save_every=1)

    # n_steps = ceil(0.25/0.1) = 3, plus initial snapshot
    assert len(snapshots) == 4


def test_solver_rejects_unstable_heat_equation_time_step() -> None:
    """Solver should reject dt values violating HeatEquation CFL validation."""
    grid = Grid1D(0.0, 1.0, 11)
    state = ScalarField(grid, data=np.sin(np.pi * grid.x))
    equation = HeatEquation(alpha=0.1, operator=LaplacianOperator())
    solver = Solver(ExplicitEulerIntegrator(), equation)

    with pytest.raises(ValueError, match="violates CFL condition"):
        solver.run(state, dt=0.2, n_steps=1)
