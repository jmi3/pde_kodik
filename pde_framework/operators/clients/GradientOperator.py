"""1D gradient operator implementation."""

from __future__ import annotations

from pde_framework.calculation_core import (
    gradient_1d_backward,
    gradient_1d_central,
    gradient_1d_forward,
)
from pde_framework.field import ScalarField, VectorField
from pde_framework.grid import Grid1D
from pde_framework.operators.IOperator import IOperator


class GradientOperator(IOperator):
    """Discrete gradient operator.

    Parameters
    ----------
    scheme : str
        Finite-difference scheme: "central", "forward", or "backward".
    """

    def __init__(self, scheme: str = "central") -> None:
        normalized = scheme.lower().strip()
        if normalized not in {"central", "forward", "backward"}:
            raise ValueError(
                f"Unsupported gradient scheme '{scheme}'. Expected 'central', 'forward', or 'backward'."
            )
        self.scheme = normalized

    def apply(self, field: ScalarField) ->  ScalarField:
        """Apply the gradient to a scalar field and return a vector field."""

        if self.scheme == "central":
            gradient = gradient_1d_central(field.data, field.grid.dx)
        elif self.scheme == "forward":
            gradient = gradient_1d_forward(field.data, field.grid.dx)
        else:
            gradient = gradient_1d_backward(field.data, field.grid.dx)

        return ScalarField(field.grid, gradient)
    

    def __repr__(self) -> str:
        return f"GradientOperator(scheme='{self.scheme}')"
