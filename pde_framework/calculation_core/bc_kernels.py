"""Pure numerical kernels for applying boundary conditions."""

from __future__ import annotations

import numpy as np
from numba import njit


def apply_dirichlet_1d(u: np.ndarray, left_value: compelx, right_value: float) -> None:
    """Apply Dirichlet boundary conditions in place.

    Parameters
    ----------
    u : np.ndarray
        State vector modified in place.
    left_value : float
        Value to assign to the left boundary.
    right_value : float
        Value to assign to the right boundary.
    """

    if u.size == 0:
        return

    u[0] = left_value
    if u.size > 1:
        u[u.size - 1] = right_value


def apply_neumann_1d(u: np.ndarray, dx: float, left_gradient: float, right_gradient: float) -> None:
    """Apply Neumann (prescribed gradient) BCs in place.

    Uses a simple one-sided finite-difference approximation for the
    boundary derivative:

        (u[1] - u[0]) / dx = left_gradient
        (u[-1] - u[-2]) / dx = right_gradient

    which yields boundary values updated in-place without modifying
    interior points.
    """

    n = u.size
    if n == 0:
        return

    if n > 1:
        u[0] = u[1] - left_gradient * dx
        u[n - 1] = u[n - 2] + right_gradient * dx
    else:
        # Single-point domain: set to average implied by gradients (degenerate)
        u[0] = 0.0


def apply_periodic_1d(u: np.ndarray) -> None:
    """Apply periodic boundary conditions in place.

    Conventions:
    - Grid stores boundary nodes explicitly. We enforce periodicity by
      copying interior neighbor values to the boundary nodes using a
      ghost-like convention:

        u[0] <- u[-2]
        u[-1] <- u[1]

    This keeps the interior (1..-2) unchanged and ensures values at the
    two boundary nodes are consistent with a periodic tiling.
    """

    n = u.size
    if n < 2:
        return

    if n > 2:
        u[0] = u[n - 2]
        u[n - 1] = u[1]
    else:
        # Two-point domain: make them equal
        u[0] = u[1]


def apply_robin_1d(
    u: np.ndarray,
    dx: float,
    left_a: float,
    left_b: float,
    left_c: float,
    right_a: float,
    right_b: float,
    right_c: float,
) -> None:
    """Apply Robin BCs (a*u + b*du/dn = c) in place.

    We approximate the normal derivative at boundaries by a one-sided
    finite difference: du/dn ~ (u[1] - u[0]) / dx for the left boundary
    and du/dn ~ (u[-1] - u[-2]) / dx for the right boundary. The
    resulting linear equations are solved for the boundary node values.
    """

    n = u.size
    if n == 0:
        return

    if n > 1:
        # Left boundary: a*u0 + b*(u1 - u0)/dx = c -> u0*(a - b/dx) + (b/dx)*u1 = c
        denom_left = left_a - left_b / dx
        if denom_left != 0.0:
            u[0] = (left_c - (left_b / dx) * u[1]) / denom_left
        else:
            # Degenerate case: fall back to a simple assignment
            u[0] = left_c / (left_a if left_a != 0.0 else 1.0)

        # Right boundary: a*uN + b*(uN - uN-1)/dx = c -> uN*(a + b/dx) - (b/dx)*uN-1 = c
        denom_right = right_a + right_b / dx
        if denom_right != 0.0:
            u[n - 1] = (right_c + (right_b / dx) * u[n - 2]) / denom_right
        else:
            u[n - 1] = right_c / (right_a if right_a != 0.0 else 1.0)
    else:
        # Single-point domain: set to average of requested constants
        u[0] = (left_c + right_c) / (left_a + right_a if (left_a + right_a) != 0.0 else 1.0)
