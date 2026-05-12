"""Tests for the boundary-condition interface."""

import pytest

from pde_framework.boundary_conditions import IBoundaryCondition


def test_interface_is_abstract() -> None:
    """The interface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IBoundaryCondition()


def test_interface_requires_apply_and_repr() -> None:
    """Concrete implementations must define apply and __repr__."""

    class MissingMethods(IBoundaryCondition):
        pass

    with pytest.raises(TypeError):
        MissingMethods()
