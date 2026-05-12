"""Tests for utility validators."""

from dataclasses import dataclass

import numpy as np
import pytest

from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.utils import validate_field_matches_grid, validate_grid1d, validate_heat_cfl_1d


@dataclass
class GridStub:
    """Minimal grid-like stub for validator tests."""

    x: np.ndarray
    dx: float


def test_validate_grid1d_accepts_monotonic_positive_dx() -> None:
    """Validator accepts a strictly increasing 1D grid with positive spacing."""
    grid = GridStub(x=np.array([0.0, 0.5, 1.0]), dx=0.5)

    validate_grid1d(grid)


def test_validate_grid1d_rejects_non_monotonic_coordinates() -> None:
    """Validator rejects non-monotonic coordinate arrays."""
    grid = GridStub(x=np.array([0.0, 0.5, 0.4]), dx=0.5)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_grid1d(grid)


def test_validate_grid1d_rejects_non_positive_dx() -> None:
    """Validator rejects non-positive spacing."""
    grid = GridStub(x=np.array([0.0, 0.5, 1.0]), dx=0.0)

    with pytest.raises(ValueError, match="must be positive"):
        validate_grid1d(grid)


def test_validate_field_matches_grid_accepts_consistent_field() -> None:
    """Field-grid validator accepts matching shapes."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=5)
    field = ScalarField(grid, data=np.ones(grid.shape))

    validate_field_matches_grid(field)


def test_validate_field_matches_grid_rejects_mismatch() -> None:
    """Field-grid validator reports expected and actual shape clearly."""

    class BrokenField:
        def __init__(self) -> None:
            self.grid = type("GridStub", (), {"shape": (5,)})()
            self.data = np.zeros(3)

    with pytest.raises(ValueError, match=r"expected \(5,\), got \(3,\)"):
        validate_field_matches_grid(BrokenField())


def test_validate_heat_cfl_1d_stable_case() -> None:
    """Heat CFL validator accepts stable explicit step size."""

    assert validate_heat_cfl_1d(alpha=0.1, dt=0.001, dx=0.1)


def test_validate_heat_cfl_1d_raises_for_unstable_case() -> None:
    """Heat CFL validator rejects unstable explicit step size."""

    with pytest.raises(ValueError, match="violates CFL condition"):
        validate_heat_cfl_1d(alpha=0.1, dt=0.1, dx=0.1)


def test_validate_heat_cfl_1d_can_warn_instead_of_raise() -> None:
    """Heat CFL validator can emit warning for unstable configuration."""

    with pytest.warns(RuntimeWarning, match="violates CFL condition"):
        stable = validate_heat_cfl_1d(alpha=0.1, dt=0.1, dx=0.1, mode="warn")

    assert stable is False
