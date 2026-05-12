"""Tests for field-to-grid validation behavior."""

import numpy as np
import pytest

from pde_framework.field import ScalarField
from pde_framework.grid import Grid1D
from pde_framework.utils import validate_field_matches_grid


def test_scalar_field_constructor_uses_field_grid_validation() -> None:
    """ScalarField constructor should fail when data shape mismatches the grid."""
    grid = Grid1D(x_min=0.0, x_max=1.0, n_points=6)

    with pytest.raises(ValueError, match=r"expected \(6,\), got \(5,\)"):
        ScalarField(grid, data=np.zeros(5))


def test_validate_field_matches_grid_reports_expected_and_actual_shape() -> None:
    """Validation error message should include expected and actual shape."""

    class FieldStub:
        def __init__(self) -> None:
            self.grid = type("GridStub", (), {"shape": (4,)})()
            self.data = np.zeros(2)

    with pytest.raises(ValueError, match=r"expected \(4,\), got \(2,\)"):
        validate_field_matches_grid(FieldStub())
