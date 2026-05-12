from __future__ import annotations

import numpy as np

from pde_framework.field import ScalarField
from pde_framework.integrators.ISpatialIntegrator import ISpatialIntegrator


class SpatialIntegrator(ISpatialIntegrator):
    """
    Integrator helper for scalar fields.

    Methods
    -------
    integrate_trapezoidal(field, a, b)
        Composite trapezoidal rule on interval [a, b].
    integrate_simpson(field, a, b)
        Composite Simpson's rule on interval [a, b]. Requires an
        even number of subintervals (odd number of nodes).
    """

    def __init__(self, expression):
        if expression not in ["trapezoidal", "simpson"]:
            raise NotImplementedError("This type of integration is unknown.")
        self.expression = expression

    def integrate(self, field: ScalarField, a: float | int, b: float | int) -> complex | float:
        match self.expression:
            case "trapezoidal":
                return self.integrate_trapezoidal(field, a, b)

            case "simpson":
                return self.integrate_simpson(field, a, b)

            case _:
                raise NotImplementedError()

    def _resolve_interval(self, grid, a: float | int, b: float | int) -> tuple[int, int]:
        """Return pair of integer indices (i0, i1) covering [a, b].

        If `a`/`b` are integers they are interpreted as indices.
        Otherwise they are treated as coordinate values and the
        nearest grid nodes inside [a,b] are selected.
        """

        x = grid.x
        if isinstance(a, int) and isinstance(b, int):
            i0, i1 = int(a), int(b)
        else:
            ai = float(a)
            bi = float(b)
            if bi < ai:
                raise ValueError("right endpoint must be >= left endpoint")
            # first index with x >= a
            i0 = int(np.searchsorted(x, ai, side="left"))
            # last index with x <= b
            i1 = int(np.searchsorted(x, bi, side="right") - 1)

        if i0 < 0 or i1 >= grid.n_points:
            raise IndexError("interval endpoints fall outside the grid")
        if i1 <= i0:
            raise ValueError("interval must contain at least one subinterval")

        return i0, i1

    def integrate_trapezoidal(self, field: ScalarField, a: float | int, b: float | int) -> complex:
        """Compute integral of `field` over [a, b] with composite trapezoidal rule.

        `a` and `b` may be grid coordinates or integer indices. The implementation
        selects the grid nodes that lie inside the requested interval and applies
        the standard composite trapezoidal formula using `grid.dx`.
        """

        i0, i1 = self._resolve_interval(field.grid, a, b)
        y = field.data[i0 : i1 + 1]
        dx = field.grid.dx

        if y.size < 2:
            raise ValueError("Need at least two nodes for trapezoidal rule")

        integral = dx * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])
        return complex(integral)

    def integrate_simpson(self, field: ScalarField, a: float | int, b: float | int) -> complex:
        """Compute integral of `field` over [a, b] with composite Simpson's rule.

        Requires an even number of subintervals (i.e. an odd number of nodes).
        `a` and `b` may be coordinates or integer indices; the implementation
        selects grid nodes inside [a,b].
        """

        i0, i1 = self._resolve_interval(field.grid, a, b)
        y = field.data[i0 : i1 + 1]
        dx = field.grid.dx

        n_intervals = y.size - 1
        if n_intervals < 1:
            raise ValueError("Need at least two nodes for Simpson's rule")
        if n_intervals % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of subintervals")

        # Simpson composite: dx/3 * (y0 + yn + 4*sum(odd indices) + 2*sum(even indices))
        odd_sum = y[1:-1:2].sum() if y.size > 2 else 0.0
        even_sum = y[2:-1:2].sum() if y.size > 3 else 0.0

        integral = dx / 3.0 * (y[0] + y[-1] + 4.0 * odd_sum + 2.0 * even_sum)
        return complex(integral)
