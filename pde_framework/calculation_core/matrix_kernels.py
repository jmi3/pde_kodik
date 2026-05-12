import numpy as np
from numba import njit


@njit(cache=True)
def thomas(d: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    Solve Ax = d for a tridiagonal matrix A using the Thomas algorithm.

    Parameters
    ----------
    d : ndarray, shape (X,)
        Right-hand side vector.
    a : ndarray, shape (X,)
        Subdiagonal. Values a[1] ... a[X-1] are used.
    b : ndarray, shape (X,)
        Main diagonal.
    c : ndarray, shape (X,)
        Superdiagonal. Values c[0] ... c[X-2] are used.

    Returns
    -------
    x : ndarray, shape (X,)
        Solution vector.
    """

    points = d.shape[0]

    # Allocate output and temporary modified superdiagonal.
    x = np.empty(points, dtype=np.complex128)
    scratch = np.empty(points, dtype=np.complex128)

    if points == 0:
        return x

    if points == 1:
        x[0] = d[0] / b[0]
        return x

    # Forward sweep.
    inv_denom = 1.0 / b[0]
    scratch[0] = c[0] * inv_denom
    x[0] = d[0] * inv_denom

    for ix in range(1, points):
        denom = b[ix] - a[ix] * scratch[ix - 1]
        inv_denom = 1.0 / denom

        if ix < points - 1:
            scratch[ix] = c[ix] * inv_denom

        x[ix] = (d[ix] - a[ix] * x[ix - 1]) * inv_denom

    # Back substitution.
    for ix in range(points - 2, -1, -1):
        x[ix] -= scratch[ix] * x[ix + 1]

    return x


@njit(cache=True)
def tridiag_multiply_vec(x, a, b, c):
    """
    Compute y = A x, where A is a tridiagonal matrix.

    Parameters
    ----------
    x : ndarray, shape (X,)
        Input vector.
    a : ndarray, shape (X,)
        Subdiagonal. Values a[1] ... a[X-1] are used.
    b : ndarray, shape (X,)
        Main diagonal.
    c : ndarray, shape (X,)
        Superdiagonal. Values c[0] ... c[X-2] are used.

    Returns
    -------
    y : ndarray, shape (X,)
        Result of the matrix-vector product A x.
    """

    points = x.shape[0]

    # Allocate output vector.
    y = np.empty(points, dtype=np.complex128)

    if points == 0:
        return y

    if points == 1:
        y[0] = b[0] * x[0]
        return y

    # First row.
    y[0] = b[0] * x[0] + c[0] * x[1]

    # Inner rows.
    for ix in range(1, points - 1):
        y[ix] = a[ix] * x[ix - 1] + b[ix] * x[ix] + c[ix] * x[ix + 1]

    # Last row.
    y[points - 1] = a[points - 1] * x[points - 2] + b[points - 1] * x[points - 1]

    return y
