import numpy as np

from pde_framework.boundary_conditions.IBoundaryCondition import IBoundaryCondition
from pde_framework.boundary_conditions.clients.DirichletBC import DirichletBC
from pde_framework.equations.IDiffusionEquation import IDiffusionEquation
from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.operators.clients.LaplacianOperator import LaplacianOperator


class SchrodingerEquation(IDiffusionEquation):
    """
    SE:
    d/dt psi = alpha i Laplace(psi) - i V(x) psi

    """

    def __init__(
        self,
        alpha: float = 1.0,
        operator: LaplacianOperator | None = None,
        bc: IBoundaryCondition | None = None ,
        potential: ScalarField | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.operator = operator or LaplacianOperator()
        self.bc = bc or DirichletBC(left_value=0, right_value=0)
        self.potential = potential

    def set_potential(self, potential: ScalarField):
        self.potential = potential

    def rhs(self, state: ScalarField) -> np.ndarray:
        """Right-hand side of the Schrödinger equation.

        Returns a NumPy array (not a ScalarField) representing the time
        derivative of the state: d/dt psi = i*(alpha*Laplace(psi) - V*psi).
        """
        lap_field = self.operator.apply(state)
        lap = lap_field.data

        if self.potential is None:
            return 1j * (self.alpha * lap)

        if not self.potential.grid == state.grid:
            raise ValueError("The state must be defined on the same grid as the potential.")

        pot = self.potential.data
        psi = state.data
        result = (1j * (complex(self.alpha) * lap - pot * psi))

        self.bc.apply(field_data=result, grid=state.grid)

        return result

    def get_tridiagonal_form(self, grid: Grid1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_shape = np.ones_like(grid.x)
        coefficient = pow(grid.dx, -2) * self.alpha * 1j

        if self.potential is None:
            main_diagonal = -base_shape * 2 * coefficient

        elif self.potential.grid == grid:
            main_diagonal = -base_shape * 2 * coefficient - self.potential.data * 1j

        else:
            raise ValueError("The state must be defined on the same grid as the potential.")

        off_diagonal = base_shape * coefficient

        return off_diagonal, main_diagonal, off_diagonal
