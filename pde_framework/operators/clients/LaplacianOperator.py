"""1D Laplacian operator implementation."""

from __future__ import annotations

from pde_framework.calculation_core import laplacian_1d
from pde_framework.field import ScalarField
from pde_framework.operators.IOperator import IOperator


class LaplacianOperator(IOperator):
    """Discrete 1D Laplacian operator.

    The operator returns a new ScalarField and leaves the input unchanged.
    """

    def apply(self, field: ScalarField) -> ScalarField:
        """Apply the discrete Laplacian to a scalar field."""

        laplacian = laplacian_1d(field.data, field.grid.dx)
        return ScalarField(field.grid, data=laplacian)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""

        return "LaplacianOperator()"
