"""
Utils - Utility functions for common operations.
"""

from .array_utils import gaussian_1d
from .validators import (
    validate_field_matches_grid,
    validate_grid1d,
    validate_heat_cfl_1d,
    validate_scalar_field_binary_compatibility,
)

__all__ = [
    "gaussian_1d",
    "validate_field_matches_grid",
    "validate_grid1d",
    "validate_heat_cfl_1d",
    "validate_scalar_field_binary_compatibility",
]
