"""Periodic boundary-condition implementation."""

from __future__ import annotations

import numpy as np

from pde_framework.boundary_conditions.IBoundaryCondition import IBoundaryCondition
from pde_framework.calculation_core import apply_periodic_1d


class PeriodicBC(IBoundaryCondition):
    """Periodic BC for 1D fields.

    This implementation enforces periodicity by copying interior neighbor
    values to the explicit boundary nodes using the project's kernel.
    """

    def apply(self, field_data: np.ndarray, grid) -> None:
        apply_periodic_1d(field_data)

    def __repr__(self) -> str:
        return "PeriodicBC()"
