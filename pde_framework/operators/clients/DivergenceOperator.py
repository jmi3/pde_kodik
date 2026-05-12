"""1D divergence operator implementation."""

from __future__ import annotations

from pde_framework.calculation_core import divergence_1d
from pde_framework.field import ScalarField, VectorField
from pde_framework.operators.IOperator import IOperator


class DivergenceOperator(IOperator):
    """Discrete 1D divergence operator."""

    def apply(self, vector_field: VectorField) -> ScalarField:
        """Apply divergence to a vector field and return a scalar field."""

        if len(vector_field.components) != 1:
            raise ValueError(
                f"1D DivergenceOperator expects 1 component, got {len(vector_field.components)}"
            )

        divergence = divergence_1d(vector_field.components[0], vector_field.grid.dx)
        return ScalarField(vector_field.grid, data=divergence)

    def __repr__(self) -> str:
        return "DivergenceOperator()"
