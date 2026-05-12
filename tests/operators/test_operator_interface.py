"""Tests for the operator interface."""

import pytest

from pde_framework.operators import IOperator


def test_operator_interface_is_abstract() -> None:
    """The operator interface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IOperator()
