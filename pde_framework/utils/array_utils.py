"""Array utilities used throughout the framework."""

from __future__ import annotations

import numpy as np


def gaussian_1d(
    x: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return a one-dimensional Gaussian profile.

    Parameters
    ----------
    x : np.ndarray
        Sample locations.
    center : float
        Gaussian center.
    sigma : float
        Standard deviation of the Gaussian; must be positive.
    amplitude : float, default=1.0
        Peak amplitude.

    Returns
    -------
    np.ndarray
        Gaussian values evaluated on ``x``.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    coordinates = np.asarray(x, dtype=float)
    exponent = -0.5 * ((coordinates - center) / sigma) ** 2
    return amplitude * np.exp(exponent)
