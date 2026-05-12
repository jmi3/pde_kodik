"""Vector field data structure bound to a computational grid."""

from __future__ import annotations

import numpy as np

from pde_framework.utils.validators import validate_grid1d


class VectorField:
    """Vector field defined on a grid.

    For the current 1D framework, the vector field stores one component.

    Parameters
    ----------
    grid : Grid1D
        Computational grid.
    components : sequence[np.ndarray]
        Vector components. For 1D, exactly one component is expected.
    """

    def __init__(self, grid, components) -> None:
        validate_grid1d(grid)

        component_list = [np.asarray(component, dtype=float).copy() for component in components]
        if len(component_list) != 1:
            raise ValueError(
                f"1D VectorField expects exactly 1 component, got {len(component_list)}"
            )

        for index, component in enumerate(component_list):
            if component.shape != grid.shape:
                raise ValueError(
                    f"component[{index}] shape {component.shape} does not match grid shape {grid.shape}"
                )

        self._grid = grid
        self._components = component_list

    @property
    def grid(self):
        """Computational grid associated with the vector field."""

        return self._grid

    @property
    def components(self) -> list[np.ndarray]:
        """Vector components as mutable NumPy arrays."""

        return self._components

    def copy(self) -> VectorField:
        """Return a deep copy of the vector field."""

        return VectorField(self.grid, [component.copy() for component in self.components])

    def norm(self, order: int | float = 2) -> float:
        """Compute norm across all components."""

        flattened = np.concatenate([component.ravel() for component in self.components])
        return float(np.linalg.norm(flattened, ord=order))

    def __repr__(self) -> str:
        return f"VectorField(n_components={len(self.components)}, grid_shape={self.grid.shape})"
