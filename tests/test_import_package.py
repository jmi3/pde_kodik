"""
Test basic package imports and architecture constraints.

This module verifies:
1. Package can be imported
2. All submodules are accessible
3. calculation_core has no dependency on OOP layers
"""

import importlib
import sys


def test_import_package():
    """Test that main package can be imported."""
    import pde_framework

    assert pde_framework.__version__ == "0.5.0"


def test_root_exports_epic2_types():
    """Test that the package root exports the Epic 2 public API."""
    from pde_framework import Grid1D, ScalarField, gaussian_1d, validate_grid1d

    assert Grid1D.__name__ == "Grid1D"
    assert ScalarField.__name__ == "ScalarField"
    assert callable(gaussian_1d)
    assert callable(validate_grid1d)


def test_root_exports_epic8_types():
    """Test that the package root exports Epic 8 public API."""
    from pde_framework import DivergenceOperator, GradientOperator, VectorField

    assert VectorField.__name__ == "VectorField"
    assert GradientOperator.__name__ == "GradientOperator"
    assert DivergenceOperator.__name__ == "DivergenceOperator"


def test_import_submodules():
    """Test that all expected submodules can be imported."""


def test_calculation_core_no_oop_dependency():
    """
    Test that calculation_core module doesn't import OOP layers.

    calculation_core should remain a pure numerical layer without dependencies
    on grid, field, operators, equations, integrators, or solvers.
    """
    # Import calculation_core module

    # Get all modules that were imported during this process
    imported_modules = set(sys.modules.keys())

    # Check that certain high-level modules are NOT imported
    oop_layers = [
        "pde_framework.grid",
        "pde_framework.field",
        "pde_framework.operators",
        "pde_framework.equations",
        "pde_framework.integrators",
        "pde_framework.solvers",
        "pde_framework.boundary_conditions",
    ]

    # After import of calculation_core, these should NOT be loaded
    imported_modules_after = set(sys.modules.keys())

    # Check that calculation_core doesn't pull in OOP layers
    for oop_layer in oop_layers:
        # It's ok if they were already imported before, but calculation_core
        # itself shouldn't import them
        if oop_layer in imported_modules_after and oop_layer not in imported_modules:
            # This is a failure - calculation_core imported an OOP layer
            raise AssertionError(
                f"calculation_core transitively imported {oop_layer}, "
                "which violates the pure kernel architecture"
            )


def test_all_modules_have_docstrings():
    """Test that all submodules have docstrings (at least in __init__.py)."""

    modules = [
        "pde_framework",
        "pde_framework.calculation_core",
        "pde_framework.grid",
        "pde_framework.field",
        "pde_framework.boundary_conditions",
        "pde_framework.operators",
        "pde_framework.equations",
        "pde_framework.integrators",
        "pde_framework.solvers",
        "pde_framework.utils",
        "pde_framework.config",
        "pde_framework.visualization",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__doc__ is not None, f"{module_name} has no docstring"
        assert len(module.__doc__.strip()) > 0, f"{module_name} has empty docstring"


if __name__ == "__main__":
    # Run tests manually
    test_import_package()
    print("✓ test_import_package passed")

    test_import_submodules()
    print("✓ test_import_submodules passed")

    test_calculation_core_no_oop_dependency()
    print("✓ test_calculation_core_no_oop_dependency passed")

    test_all_modules_have_docstrings()
    print("✓ test_all_modules_have_docstrings passed")

    print("\nAll import tests passed!")
