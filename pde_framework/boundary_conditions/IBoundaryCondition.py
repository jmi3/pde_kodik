"""Abstract interface for boundary conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IBoundaryCondition(ABC):
    """Abstract boundary-condition interface.

    Implementations apply a boundary condition in place to raw field data.
    """

    @abstractmethod
    def apply(self, field_data: np.ndarray, grid) -> None:
        """Apply the boundary condition in place.

        Parameters
        ----------
        field_data : np.ndarray
            Raw field values to modify.
        grid : object
            Grid-like object used by the boundary condition.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
