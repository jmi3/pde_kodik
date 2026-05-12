"""Pure numerical kernels for discrete gradient operators."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def gradient_1d_central(u: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D gradient using central differences in the interior.

    Interior points use centered differences. Boundaries use one-sided
    differences for a fully defined output array.
    """

    output = np.zeros_like(u)
    if u.size == 0:
        return output

    if u.size == 1:
        output[0] = 0.0
        return output

    output[0] = (u[1] - u[0]) / dx
    output[u.size - 1] = (u[u.size - 1] - u[u.size - 2]) / dx
    if u.size > 2:
        inv_2dx = 1.0 / (2.0 * dx)
        for index in range(1, u.size - 1):
            output[index] = (u[index + 1] - u[index - 1]) * inv_2dx
    return output


@njit(cache=True)
def gradient_1d_forward(u: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D gradient using forward differences.

    The last point uses backward difference to keep shape unchanged.
    """

    output = np.zeros_like(u)
    if u.size == 0:
        return output

    if u.size == 1:
        output[0] = 0.0
        return output

    inv_dx = 1.0 / dx
    for index in range(0, u.size - 1):
        output[index] = (u[index + 1] - u[index]) * inv_dx
    output[u.size - 1] = (u[u.size - 1] - u[u.size - 2]) * inv_dx
    return output


@njit(cache=True)
def gradient_1d_backward(u: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D gradient using backward differences.

    The first point uses forward difference to keep shape unchanged.
    """

    output = np.zeros_like(u)
    if u.size == 0:
        return output

    if u.size == 1:
        output[0] = 0.0
        return output

    inv_dx = 1.0 / dx
    output[0] = (u[1] - u[0]) * inv_dx
    for index in range(1, u.size):
        output[index] = (u[index] - u[index - 1]) * inv_dx
    return output
