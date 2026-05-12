"""Pure numerical kernels for explicit time stepping."""

from __future__ import annotations

import numpy as np

from pde_framework.calculation_core import thomas, tridiag_multiply_vec


def euler_step_1d(u: np.ndarray, rhs: np.ndarray, dt: float) -> np.ndarray:
    """Advance a 1D state by one explicit Euler step.

    Parameters
    ----------
    u : np.ndarray
        Current state.
    rhs : np.ndarray
        Right-hand side evaluated at the current state.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Updated state ``u + dt * rhs``.
    """

    return u + dt * rhs


def rk4_step_1d(
    u: np.ndarray,
    rhs_fn,
    dt: float,
) -> np.ndarray:
    """Advance a 1D state by one explicit 4-stage Runge-Kutta step.

    Parameters
    ----------
    u : np.ndarray
        Current state.
    rhs_fn : callable
        Function that computes right-hand side: rhs_fn(state) → np.ndarray.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Updated state using RK4 formula.

    Notes
    -----
    Classical RK4 stages:

        k1 = rhs_fn(u)
        k2 = rhs_fn(u + 0.5*dt*k1)
        k3 = rhs_fn(u + 0.5*dt*k2)
        k4 = rhs_fn(u + dt*k3)

        u_new = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """

    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)

    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def cn_step_1d(
    u: np.ndarray, subdiag: np.ndarray, maindiag: np.ndarray, superdiag: np.ndarray, dt: float
) -> np.ndarray:
    """
    Perform one Crank-Nicolson time step for a 1D linear system.

    This function advances the solution vector `u` by one time step using
    the Crank-Nicolson scheme

        (I - dt/2 * A) u_next = (I + dt/2 * A) u,

    where A is a tridiagonal operator represented by its subdiagonal,
    main diagonal, and superdiagonal.

    Parameters
    ----------
    u : ndarray, shape (X,)
        Current solution vector.
    subdiag : ndarray, shape (X,)
        Subdiagonal of the tridiagonal operator A.
        Values subdiag[1] ... subdiag[X-1] are used.
    maindiag : ndarray, shape (X,)
        Main diagonal of the tridiagonal operator A.
    superdiag : ndarray, shape (X,)
        Superdiagonal of the tridiagonal operator A.
        Values superdiag[0] ... superdiag[X-2] are used.
    dt : float
        Time step size.

    Returns
    -------
    u_next : ndarray, shape (X,)
        Solution vector after one Crank-Nicolson time step.
    """

    main_ones = np.ones_like(maindiag)
    dt_half = dt / 2
    dt_half_subdiag = subdiag * dt_half
    dt_half_maindiag = maindiag * dt_half
    dt_half_superdiag = superdiag * dt_half

    # Calculate the multiplication
    rhs_u = tridiag_multiply_vec(
        u, dt_half_subdiag, main_ones + dt_half_maindiag, dt_half_superdiag
    )

    # Invert the tridiagonal system and return the result
    return thomas(rhs_u, -dt_half_subdiag, main_ones - dt_half_maindiag, -dt_half_superdiag)
