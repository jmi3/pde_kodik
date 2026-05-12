"""Dirichlet boundary-condition implementation."""

from __future__ import annotations

import numpy as np

from pde_framework.boundary_conditions.IBoundaryCondition import IBoundaryCondition
from pde_framework.calculation_core import apply_dirichlet_1d


class DirichletBC(IBoundaryCondition):
    """Fixed-value boundary condition for one-dimensional fields.

    Parameters
    ----------
    left_value : float
        Boundary value at the left edge.
    right_value : float
        Boundary value at the right edge.
    """

    def __init__(self, left_value: float, right_value: float) -> None:
        self.left_value = float(left_value)
        self.right_value = float(right_value)

    def apply(self, field_data: np.ndarray, grid) -> None:
        """Apply the boundary condition in place."""

        apply_dirichlet_1d(field_data, self.left_value, self.right_value)

    def __repr__(self) -> str:
        """Return a string representation."""

        return f"DirichletBC(left_value={self.left_value}, right_value={self.right_value})"
