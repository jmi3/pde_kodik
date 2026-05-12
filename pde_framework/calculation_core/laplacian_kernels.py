"""Pure numerical kernels for discrete Laplacian operators."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def laplacian_1d(u: np.ndarray, dx: float) -> np.ndarray:
    """Compute the 1D discrete Laplacian with centered differences.

    Parameters
    ----------
    u : np.ndarray
        Input 1D state.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    np.ndarray
        Discrete Laplacian of ``u`` with zero-valued boundary entries.
    """

    output = np.zeros_like(u)
    if u.size < 3:
        return output

    inv_dx2 = 1.0 / (dx * dx)
    for index in range(1, u.size - 1):
        output[index] = (u[index - 1] - 2.0 * u[index] + u[index + 1]) * inv_dx2
    return output
