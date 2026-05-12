import numpy as np

from pde_framework.boundary_conditions.clients.DirichletBC import DirichletBC
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.operators.clients.LaplacianOperator import LaplacianOperator
from pde_framework.utils.validators import validate_heat_cfl_1d


class HeatEquation(IEquation):
    """Simple heat/diffusion equation wrapper.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient.
    operator : LaplacianOperator | None
        Operator used to compute spatial Laplacian. If None, `LaplacianOperator()` is used.
    bc : DirichletBC | None
        Optional boundary condition object. Solver is responsible for applying BCs after time steps.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        operator: LaplacianOperator | None = None,
        bc: DirichletBC | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.operator = operator or LaplacianOperator()
        self.bc = bc

    def rhs(self, state: ScalarField) -> np.ndarray:
        """Compute RHS = alpha * Laplacian(state)."""

        lap_field = self.operator.apply(state)
        return self.alpha * lap_field.data

    def validate_time_step(
        self,
        state: ScalarField,
        dt: float,
        *,
        safety_factor: float = 0.5,
        mode: str = "raise",
    ) -> bool:
        """Validate explicit CFL-like stability for the given state/grid.

        Parameters
        ----------
        state : ScalarField
            Current scalar field whose grid spacing is used.
        dt : float
            Candidate time step.
        safety_factor : float, default=0.5
            Maximum stable ratio ``alpha * dt / dx**2``.
        mode : {"raise", "warn"}, default="raise"
            Validation behavior for unstable choices.
        """

        return validate_heat_cfl_1d(
            alpha=self.alpha,
            dt=dt,
            dx=state.grid.dx,
            safety_factor=safety_factor,
            mode=mode,
        )
