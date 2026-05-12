"""Pure numerical kernels for discrete divergence operators."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def divergence_1d(v: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute 1D divergence (dv/dx) using central interior differences.

    Boundaries are set with one-sided differences to keep full output shape.
    """

    output = np.zeros_like(v)
    if v.size == 0:
        return output

    if v.size == 1:
        output[0] = 0.0
        return output

    output[0] = (v[1] - v[0]) / dx
    output[v.size - 1] = (v[v.size - 1] - v[v.size - 2]) / dx
    if v.size > 2:
        inv_2dx = 1.0 / (2.0 * dx)
        for index in range(1, v.size - 1):
            output[index] = (v[index + 1] - v[index - 1]) * inv_2dx
    return output
