"""Integrator interface for time stepping algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField


class ISpatialIntegrator(ABC):
    
    @abstractmethod
    def integrate(self, field: ScalarField, a: float | int, b: float | int) -> complex | float:
        """Integrate a field from a to b."""
