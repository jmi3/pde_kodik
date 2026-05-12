"""
Calculation Core - Pure numerical kernels for PDE operations.

This module contains optimized numerical kernels without OOP dependencies.
"""

from .bc_kernels import (
    apply_dirichlet_1d,
    apply_neumann_1d,
    apply_periodic_1d,
    apply_robin_1d,
)
from .divergence_kernels import divergence_1d
from .gradient_kernels import gradient_1d_backward, gradient_1d_central, gradient_1d_forward
from .laplacian_kernels import laplacian_1d
from .matrix_kernels import thomas, tridiag_multiply_vec
from .time_kernels import cn_step_1d, euler_step_1d, rk4_step_1d

__all__ = [
    "apply_dirichlet_1d",
    "apply_neumann_1d",
    "apply_periodic_1d",
    "apply_robin_1d",
    "cn_step_1d",
    "divergence_1d",
    "euler_step_1d",
    "gradient_1d_backward",
    "gradient_1d_central",
    "gradient_1d_forward",
    "laplacian_1d",
    "rk4_step_1d",
    "thomas",
    "tridiag_multiply_vec",
]
