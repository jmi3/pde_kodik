from __future__ import annotations

from abc import ABC, abstractmethod

import numpy

from pde_framework.field import ScalarField


class IEquation(ABC):
    """Interface for PDE equation objects.

    Implementations must provide a method to evaluate the right-hand side
    (time derivative) for a given `ScalarField` state.
    """

    @abstractmethod
    def rhs(self, state: ScalarField) -> numpy.ndarray:
        """Return the right-hand side (du/dt) as a NumPy array matching state.data."""
