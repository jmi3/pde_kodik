from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pde_framework.equations.IEquation import IEquation
from pde_framework.grid.Grid1D import Grid1D


class IDiffusionEquation(IEquation, ABC):
    """
    Interface for PDE diffusion equation objects of shape

    (d/dt) psi = alpha T (d/dx d/dx) psi + beta U (d/dx) psi + gamma V psi

    for T function of x, U function of x, V function of x, alpha, beta and gamma complex constants

    Implementations must provide a method to evaluate the right-hand side
    (time derivative) for a given `ScalarField` state.
    """

    @abstractmethod
    def get_tridiagonal_form(self, grid: Grid1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
