"""High-level solver orchestration for time-dependent PDEs."""

from __future__ import annotations

import numpy as np

from pde_framework.equations.IEquation import IEquation
from pde_framework.field import ScalarField
from pde_framework.integrators.IIntegrator import IIntegrator
from pde_framework.grid import Grid1D

from pde_framework.operators.clients.GradientOperator import GradientOperator


class Solver:
    """Simple solver that advances a state using an integrator and equation.

    Parameters
    ----------
    integrator : IIntegrator
        Time stepping algorithm.
    equation : IEquation
        PDE equation providing the RHS and optional BCs.
    """

    def __init__(self, integrator: IIntegrator, equation: IEquation) -> None:
        self.integrator = integrator
        self.equation = equation

        self.time_slices: list[ScalarField] = []
        self.time_slices_dx: list[ScalarField] = []

        self.derivative = GradientOperator(scheme="central")

    def run(
        self,
        initial: ScalarField,
        dt: float,
        n_steps: int | None = None,
        snapshot_stride: int | None = None,
        *,
        t_end: float | None = None,
        save_every: int = 1,
        time_slices: list[float] | None = None,
    ) -> list[ScalarField]:
        """Advance the initial state and return snapshots.

        Parameters
        ----------
        initial : ScalarField
            Initial state.
        dt : float
            Time step. Must be positive.
        n_steps : int | None, optional
            Number of steps to run. Backward-compatible option.
        snapshot_stride : int | None, optional
            Backward-compatible alias for ``save_every``.
        t_end : float | None, optional
            End time. If provided, steps are derived from ``ceil(t_end / dt)``.
            Must be positive.
        save_every : int, default=1
            Save each N-th step. Must be at least 1.
        time_slices : list[float] | None, optional
            Spatial positions at which to store the time evolution of the
            solution and its spatial derivative.

        Notes
        -----
        If the equation defines a callable ``validate_time_step(state, dt)`` hook,
        this method invokes it before stepping.

        After running, the solver stores:

        ``self.time_slices``
            Time evolution of ``u(t, x_i)`` for each requested spatial slice.

        ``self.time_slices_dx``
            Time evolution of ``du/dx(t, x_i)`` for each requested spatial slice.
        """
        dt_value = float(dt)
        if dt_value <= 0.0:
            raise ValueError("dt must be positive")

        if snapshot_stride is not None:
            save_every = snapshot_stride

        if save_every < 1:
            raise ValueError("save_every must be at least 1")

        if t_end is not None:
            t_end_value = float(t_end)
            if t_end_value <= 0.0:
                raise ValueError("t_end must be positive")

            derived_steps = int(np.ceil(t_end_value / dt_value))
            if n_steps is None:
                n_steps = derived_steps

        if n_steps is None:
            raise ValueError("either n_steps or t_end must be provided")

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        validation_hook = getattr(self.equation, "validate_time_step", None)
        if callable(validation_hook):
            validation_hook(initial, dt_value)

        state = initial
        snapshots: list[ScalarField] = [state.copy()]

        if time_slices is None:
            time_slices = []

        slices_data = [
            np.zeros(shape=(n_steps + 1,), dtype=np.complex128)
            for _ in time_slices
        ]

        slices_dx_data = [
            np.zeros(shape=(n_steps + 1,), dtype=np.complex128)
            for _ in time_slices
        ]

        slices_indices = [
            initial.grid.find_nearest_index(slice_pos)
            for slice_pos in time_slices
        ]

        state_dx = self.derivative.apply(state)

        for slice_number, slice_pos_index in enumerate(slices_indices):
            slices_data[slice_number][0] = state.data[slice_pos_index]
            slices_dx_data[slice_number][0] = state_dx.data[slice_pos_index]

        for step in range(1, n_steps + 1):
            state = self.integrator.step(state, self.equation, dt_value)

            if step % save_every == 0:
                snapshots.append(state.copy())

            state_dx = self.derivative.apply(state)

            for slice_number, slice_pos_index in enumerate(slices_indices):
                slices_data[slice_number][step] = state.data[slice_pos_index]
                slices_dx_data[slice_number][step] = state_dx.data[slice_pos_index]

        time_grid = Grid1D(
            x_min=0,
            x_max=t_end or n_steps*dt_value,
            n_points=n_steps + 1,
        )

        self.time_slices = [
            ScalarField(grid=time_grid, data=slice_data)
            for slice_data in slices_data
        ]

        self.time_slices_dx = [
            ScalarField(grid=time_grid, data=slice_dx_data)
            for slice_dx_data in slices_dx_data
        ]

        return snapshots