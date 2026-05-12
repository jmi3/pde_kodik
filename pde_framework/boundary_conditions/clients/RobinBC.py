"""Robin (mixed) boundary-condition implementation."""

from __future__ import annotations

import numpy as np

from pde_framework.boundary_conditions.IBoundaryCondition import IBoundaryCondition
from pde_framework.calculation_core import apply_robin_1d


class RobinBC(IBoundaryCondition):
    """Robin boundary condition: a*u + b*du/dn = c at each boundary.

    Parameters
    ----------
    left_a, left_b, left_c : floats
        Coefficients for the left boundary: a*u + b*du/dn = c
    right_a, right_b, right_c : floats
        Coefficients for the right boundary.
    """

    def __init__(
        self,
        left_a: float,
        left_b: float,
        left_c: float,
        right_a: float,
        right_b: float,
        right_c: float,
    ) -> None:
        self.left_a = float(left_a)
        self.left_b = float(left_b)
        self.left_c = float(left_c)
        self.right_a = float(right_a)
        self.right_b = float(right_b)
        self.right_c = float(right_c)

    def apply(self, field_data: np.ndarray, grid) -> None:
        dx = float(grid.dx)
        apply_robin_1d(
            field_data,
            dx,
            self.left_a,
            self.left_b,
            self.left_c,
            self.right_a,
            self.right_b,
            self.right_c,
        )

    def __repr__(self) -> str:
        return (
            f"RobinBC(left=({self.left_a},{self.left_b},{self.left_c}),"
            f" right=({self.right_a},{self.right_b},{self.right_c}))"
        )
