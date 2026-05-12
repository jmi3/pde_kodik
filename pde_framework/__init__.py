"""
PDE Numerical Framework

A modular framework for solving partial differential equations.
"""

from pde_framework.boundary_conditions import DirichletBC, IBoundaryCondition
from pde_framework.equations.clients.HeatEquation import HeatEquation
from pde_framework.equations.clients.SchrodingerEquation import SchrodingerEquation
from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField, VectorField
from pde_framework.grid import Grid1D
from pde_framework.integrators.clients.CrankNicolson1DIntegrator import CrankNicolson1DIntegrator
from pde_framework.integrators.clients.ExplicitEulerIntegrator import ExplicitEulerIntegrator
from pde_framework.integrators.clients.RK4Integrator import RK4Integrator
from pde_framework.integrators.clients.SpatialIntegrator import SpatialIntegrator 
from pde_framework.integrators.IIntegrator import IIntegrator
from pde_framework.operators import (
    DivergenceOperator,
    GradientOperator,
    IOperator,
    LaplacianOperator,
)
from pde_framework.solvers.Solver import Solver
from pde_framework.utils import (
    gaussian_1d,
    validate_field_matches_grid,
    validate_grid1d,
    validate_heat_cfl_1d,
    validate_scalar_field_binary_compatibility,
)

__version__ = "0.5.0"

__all__ = [
    "CrankNicolson1DIntegrator",
    "DirichletBC",
    "DivergenceOperator",
    "ExplicitEulerIntegrator",
    "GradientOperator",
    "Grid1D",
    "HeatEquation",
    "IBoundaryCondition",
    "IEquation",
    "IIntegrator",
    "IOperator",
    "LaplacianOperator",
    "RK4Integrator",
    "ScalarField",
    "SchrodingerEquation",
    "Solver",
    "SpatialIntegrator",
    "VectorField",
    "__version__",
    "gaussian_1d",
    "validate_field_matches_grid",
    "validate_grid1d",
    "validate_heat_cfl_1d",
    "validate_scalar_field_binary_compatibility",
]
