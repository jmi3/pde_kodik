"""Integrator interface for time stepping algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField


class IIntegrator(ABC):
    @abstractmethod
    def step(self, state: ScalarField, equation: IEquation, dt: float) -> ScalarField:
        """Advance `state` by one time step and return the new `ScalarField`."""
