"""Uniform one-dimensional computational grid.

This module provides the Grid1D class, a passive data structure for a uniform
1D domain with evenly spaced nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Grid1D:
    """Uniform one-dimensional computational grid.

    Parameters
    ----------
    x_min : float
        Left boundary of the domain.
    x_max : float
        Right boundary of the domain.
    n_points : int
        Number of grid points, including boundaries.

    Attributes
    ----------
    x : np.ndarray
        Grid coordinates.
    dx : float
        Grid spacing.
    shape : tuple[int]
        Shape of the grid coordinates.
    n_points : int
        Number of points in the grid.
    interior_slice : slice
        Slice selecting interior points.
    """

    x_min: float
    x_max: float
    n_points: int
    x: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)
    shape: tuple[int, ...] = field(init=False)
    interior_slice: slice = field(init=False)

    def __post_init__(self) -> None:
        """Validate inputs and build coordinates."""
        if self.n_points < 2:
            raise ValueError("n_points must be at least 2")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

        dx = (self.x_max - self.x_min) / (self.n_points - 1)
        x = np.linspace(self.x_min, self.x_max, self.n_points, dtype=float)

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "dx", float(dx))
        object.__setattr__(self, "shape", (self.n_points,))
        object.__setattr__(self, "interior_slice", slice(1, -1))

    def find_nearest_index(self, x_val: float) -> int:
        """Najde index nejbližšího grid bodu k zadané hodnotě x_val.

        Parameters
        ----------
        x_val : float
            Souřadnice, k níž hledáme nejbližší gridpoint.

        Returns
        -------
        int
            Index nejbližšího gridpointu.
        """
        return int(np.argmin(np.abs(self.x - x_val)))
