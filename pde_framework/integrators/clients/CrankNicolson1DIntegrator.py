"""Explicit Euler integrator using calculation_core time kernel."""

from __future__ import annotations

from pde_framework.calculation_core import cn_step_1d
from pde_framework.equations.IDiffusionEquation import IDiffusionEquation
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.integrators.IIntegrator import IIntegrator


class CrankNicolson1DIntegrator(IIntegrator):
    def __init__(self, type: str):
        if type not in ["diffusion"]:
            raise NotImplementedError()
        self.type = type

    def _diffusion_step(
        self, state: ScalarField, equation: IDiffusionEquation, dt: float
    ) -> ScalarField:

        subdiag, maindiag, superdiag = equation.get_tridiagonal_form(state.grid)
        new_data = cn_step_1d(
            u=state.data, subdiag=subdiag, maindiag=maindiag, superdiag=superdiag, dt=dt
        )
        return ScalarField(grid=state.grid, data=new_data)

    def step(self, state: ScalarField, equation: IEquation, dt: float) -> ScalarField:
        match self.type:
            case "diffusion":
                if not isinstance(equation, IDiffusionEquation):
                    raise TypeError("Equation is not a diffusion one.")
                new_state = self._diffusion_step(state, equation, dt)
            case _:
                raise TypeError("This is an unknown equation type to this integrator.")
        return new_state
