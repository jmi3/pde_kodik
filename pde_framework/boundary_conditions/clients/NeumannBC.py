"""Neumann boundary-condition implementation."""

from __future__ import annotations

import numpy as np

from pde_framework.boundary_conditions.IBoundaryCondition import IBoundaryCondition
from pde_framework.calculation_core import apply_neumann_1d


class NeumannBC(IBoundaryCondition):
    """Neumann (prescribed normal derivative) BC for 1D fields.

    Parameters
    ----------
    left_gradient : float
        Prescribed gradient at the left boundary (du/dn).
    right_gradient : float
        Prescribed gradient at the right boundary (du/dn).
    """

    def __init__(self, left_gradient: float, right_gradient: float) -> None:
        self.left_gradient = float(left_gradient)
        self.right_gradient = float(right_gradient)

    def apply(self, field_data: np.ndarray, grid) -> None:
        """Apply the Neumann BC to `field_data` in place."""

        dx = float(grid.dx)
        apply_neumann_1d(field_data, dx, self.left_gradient, self.right_gradient)

    def __repr__(self) -> str:
        return (
            f"NeumannBC(left_gradient={self.left_gradient}, right_gradient={self.right_gradient})"
        )
