"""Explicit Euler integrator using calculation_core time kernel."""

from __future__ import annotations

from pde_framework.calculation_core import euler_step_1d
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.integrators.IIntegrator import IIntegrator


class ExplicitEulerIntegrator(IIntegrator):
    def step(self, state: ScalarField, equation: IEquation, dt: float) -> ScalarField:
        rhs = equation.rhs(state)
        new_data = euler_step_1d(state.data, rhs, dt)
        new_field = ScalarField(state.grid, data=new_data)

        # Apply boundary conditions if the equation provides them
        bc = getattr(equation, "bc", None)
        if bc is not None:
            bc.apply(new_field.data, new_field.grid)

        return new_field
