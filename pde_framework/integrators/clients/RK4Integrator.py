"""Runge-Kutta 4 integrator for explicit time stepping."""

from __future__ import annotations

from pde_framework.calculation_core import rk4_step_1d
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.integrators.IIntegrator import IIntegrator


class RK4Integrator(IIntegrator):
    """4-stage explicit Runge-Kutta integrator for PDEs.

    Uses the classical RK4 formula with 4 evaluations of the right-hand side.
    Provides higher accuracy than explicit Euler at the cost of more RHS evaluations.

    Attributes
    ----------
    None (stateless).
    """

    def step(self, state: ScalarField, equation: IEquation, dt: float) -> ScalarField:
        """Advance state by one time step using RK4.

        Parameters
        ----------
        state : ScalarField
            Current solution field. Not modified.
        equation : IEquation
            PDE equation providing the rhs method.
        dt : float
            Time step size.

        Returns
        -------
        ScalarField
            New solution field on the same grid.

        Notes
        -----
        The integrator internally evaluates equation.rhs 4 times, so for stiff
        problems or tight coupling, it may be significantly more expensive than Euler.
        """

        # Closure: rhs evaluation via equation
        def rhs_fn(state_data):
            # Create temporary field for RHS evaluation
            temp_field = ScalarField(state.grid, data=state_data)
            return equation.rhs(temp_field)

        # Perform RK4 step on data
        new_data = rk4_step_1d(state.data, rhs_fn, dt)
        new_field = ScalarField(state.grid, data=new_data)

        # Apply boundary conditions if provided
        bc = getattr(equation, "bc", None)
        if bc is not None:
            bc.apply(new_field.data, new_field.grid)

        return new_field
