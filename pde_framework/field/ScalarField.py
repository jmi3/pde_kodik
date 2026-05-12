"""Scalar field data structure bound to a computational grid."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from pde_framework.utils.validators import (
    validate_field_matches_grid,
    validate_grid1d,
    validate_scalar_field_binary_compatibility,
)

if TYPE_CHECKING:
    from pde_framework.grid import Grid1D


class ScalarField:
    """Scalar field defined on a grid.

    Parameters
    ----------
    grid : Grid1D
        Computational grid.
    data : array-like, optional
        Field values. If omitted, the field is initialized with zeros.
    """

    def __init__(self, grid: Grid1D, data: np.ndarray | None = None) -> None:
        validate_grid1d(grid)

        self._grid = grid
        if data is None:
            self._data = np.zeros(grid.shape, dtype=complex)
        else:
            array = np.asarray(data, dtype=complex)
            self._data = array.copy()

        validate_field_matches_grid(self)

    @property
    def grid(self) -> Grid1D:
        """Computational grid associated with the field."""

        return self._grid

    @property
    def data(self) -> np.ndarray:
        """Raw field values as a NumPy array."""

        return self._data

    def copy(self) -> ScalarField:
        """Return a deep copy of the field."""

        return ScalarField(self.grid, data=self.data.copy())

    def fill(self, value: complex) -> None:
        """Fill the field with a constant value."""

        self._data.fill(value)

    def apply_function(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a function to the grid coordinates and store the result."""

        values = np.asarray(func(self.grid.x), dtype=complex)
        if values.shape != self.grid.shape:
            raise ValueError(
                f"function output shape {values.shape} does not match grid shape {self.grid.shape}"
            )
        self._data = values.copy()

    def norm(self, order: int | float = 2) -> float:
        """Compute the vector norm of the field values."""

        return float(np.linalg.norm(self.data, ord=order))

    def _ensure_compatible(self, other: ScalarField) -> None:
        if not isinstance(other, ScalarField):
            raise TypeError("operation requires another ScalarField")
        validate_scalar_field_binary_compatibility(self, other)

    def __add__(self, other: ScalarField) -> ScalarField:
        """Add two scalar fields."""

        self._ensure_compatible(other)
        return ScalarField(self.grid, data=self.data + other.data)

    def __sub__(self, other: ScalarField) -> ScalarField:
        """Subtract one scalar field from another."""

        self._ensure_compatible(other)
        return ScalarField(self.grid, data=self.data - other.data)

    def __mul__(self, multiple: float | complex | ScalarField) -> ScalarField:
        """Multiply the field by a scalar."""

        if np.isscalar(multiple):
            return ScalarField(self.grid, data=self.data * multiple)

        if isinstance(multiple, ScalarField):
            return ScalarField(self.grid, data=self.data * multiple.data)

        raise NotImplementedError()

    def __rmul__(self, multiple: float | complex | ScalarField) -> ScalarField:
        """Right scalar multiplication."""

        return self.__mul__(multiple)
