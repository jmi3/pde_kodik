"""Validation helpers for core data structures.

This module currently provides validators for one-dimensional grids.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def validate_grid1d(grid: Any) -> None:
    """Validate a one-dimensional grid-like object.

    Parameters
    ----------
    grid : Any
        Grid-like object with ``x`` and ``dx`` attributes.

    Raises
    ------
    ValueError
        If the grid coordinates are not strictly increasing or if ``dx`` is
        not positive.
    AttributeError
        If the required grid attributes are missing.
    """

    if not hasattr(grid, "x"):
        raise AttributeError("grid must define an 'x' attribute")
    if not hasattr(grid, "dx"):
        raise AttributeError("grid must define a 'dx' attribute")

    coordinates = np.asarray(grid.x, dtype=float)
    if coordinates.ndim != 1:
        raise ValueError("grid.x must be one-dimensional")
    if coordinates.size < 2:
        raise ValueError("grid.x must contain at least two points")

    differences = np.diff(coordinates)
    if np.any(differences <= 0.0):
        raise ValueError("grid.x must be strictly increasing")

    dx = float(grid.dx)
    if dx <= 0.0:
        raise ValueError("grid.dx must be positive")


def validate_field_matches_grid(field: Any) -> None:
    """Validate that field data shape matches its grid shape.

    Parameters
    ----------
    field : Any
        Field-like object with ``grid`` and ``data`` attributes where
        ``grid`` has a ``shape`` attribute.

    Raises
    ------
    AttributeError
        If required field attributes are missing.
    ValueError
        If field data shape differs from the expected grid shape.
    """

    if not hasattr(field, "grid"):
        raise AttributeError("field must define a 'grid' attribute")
    if not hasattr(field, "data"):
        raise AttributeError("field must define a 'data' attribute")
    if not hasattr(field.grid, "shape"):
        raise AttributeError("field.grid must define a 'shape' attribute")

    expected_shape = tuple(field.grid.shape)
    actual_shape = np.asarray(field.data).shape
    if actual_shape != expected_shape:
        raise ValueError(
            "field data shape does not match grid shape: "
            f"expected {expected_shape}, got {actual_shape}"
        )


def validate_scalar_field_binary_compatibility(left: Any, right: Any) -> None:
    """Validate compatibility of two scalar fields for binary operations.

    Two fields are considered compatible when both shape and coordinates are
    numerically equal.

    Parameters
    ----------
    left : Any
        Left field-like operand.
    right : Any
        Right field-like operand.

    Raises
    ------
    AttributeError
        If either field does not expose required ``grid``/``data``/``x``
        attributes.
    ValueError
        If fields use incompatible grids.
    """

    validate_field_matches_grid(left)
    validate_field_matches_grid(right)

    left_shape = tuple(left.grid.shape)
    right_shape = tuple(right.grid.shape)
    if left_shape != right_shape:
        raise ValueError(
            "ScalarField grids are not compatible: "
            f"left shape {left_shape}, right shape {right_shape}"
        )

    if not hasattr(left.grid, "x") or not hasattr(right.grid, "x"):
        raise AttributeError("both fields must use grids with an 'x' attribute")

    left_x = np.asarray(left.grid.x, dtype=float)
    right_x = np.asarray(right.grid.x, dtype=float)
    if not np.allclose(left_x, right_x):
        raise ValueError("ScalarField grids are not compatible: grid coordinates differ")


def validate_heat_cfl_1d(
    alpha: float,
    dt: float,
    dx: float,
    safety_factor: float = 0.5,
    *,
    mode: str = "raise",
) -> bool:
    """Validate explicit CFL-like stability limit for 1D heat equation.

    The explicit Euler update for 1D diffusion is stable when
    ``alpha * dt / dx**2 <= safety_factor``. Typical conservative value is 0.5.

    Parameters
    ----------
    alpha : float
        Diffusion coefficient, must be positive.
    dt : float
        Time step, must be positive.
    dx : float
        Spatial spacing, must be positive.
    safety_factor : float, default=0.5
        Stability threshold.
    mode : {"raise", "warn"}, default="raise"
        Behavior for unstable configuration.

    Returns
    -------
    bool
        ``True`` if stable, ``False`` if unstable and ``mode='warn'``.

    Raises
    ------
    ValueError
        If inputs are invalid or if unstable in ``mode='raise'``.
    """

    alpha_value = float(alpha)
    dt_value = float(dt)
    dx_value = float(dx)
    sf_value = float(safety_factor)

    if alpha_value <= 0.0:
        raise ValueError("alpha must be positive")
    if dt_value <= 0.0:
        raise ValueError("dt must be positive")
    if dx_value <= 0.0:
        raise ValueError("dx must be positive")
    if sf_value <= 0.0:
        raise ValueError("safety_factor must be positive")
    if mode not in {"raise", "warn"}:
        raise ValueError("mode must be either 'raise' or 'warn'")

    cfl_number = alpha_value * dt_value / (dx_value * dx_value)
    stable = cfl_number <= sf_value
    if stable:
        return True

    message = (
        "Explicit heat scheme violates CFL condition: "
        f"alpha*dt/dx^2={cfl_number:.6g} exceeds safety_factor={sf_value:.6g}. "
        f"Use dt <= {sf_value * dx_value * dx_value / alpha_value:.6g}."
    )
    if mode == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False

    raise ValueError(message)
