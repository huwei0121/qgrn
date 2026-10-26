"""Validated solver options."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverOptions:
    """Iteration, model and constraint settings for Q-GRN.

    ``time_limit`` is a wall-clock budget in seconds, including restoration and
    final evaluation. An oracle or factorization already running is not interrupted.
    ``max_iterations`` counts accepted steps. By default, no objective, gradient
    or small-step precision test stops the solve. Numerical inability to produce
    a representable improving step is reported separately from convergence.

    ``scale_update="step_aware"`` uses accepted backtracking lengths to update
    the regularization scale; ``"ratio"`` uses only actual/predicted reduction.
    ``curvature_estimator`` is ``"ratio"`` (default), ``"residual"`` or ``"split"``.
    The latter two require a complete residual contract from constrained problems.
    ``use_hardware_constraints`` requests the problem's hardware_geometry method.
    Its ``hardware_model`` is a slack-constrained QP or a local tangent face.

    Targets are used only when ``stop_on_target=True``. With no explicit
    ``objective_target``, the threshold is 1e-3 times problem.objective_weight,
    whose default is 1. A target also checks gate error and primal
    feasibility, but does not certify KKT stationarity or global optimality.
    """

    time_limit: float = 60.0
    max_iterations: int = 3000
    barrier_initial: float = 1e-3
    barrier_minimum: float = 1e-9
    initial_regularization_ratio: float = 0.01
    gate_error_target: float = 1e-3
    objective_target: float | None = None
    stop_on_target: bool = False
    feasibility_tolerance: float = 1e-6
    direction_tolerance: float = 1e-6
    curvature_estimator: str = "ratio"
    scale_update: str = "step_aware"
    remainder_feedback: str = "underprediction"
    estimator_decay: float = 0.5
    objective_scale: float | None = None
    step_scale: float = 1.0
    use_hardware_constraints: bool = False
    hardware_model: str = "slack"
    slack_fraction: float = 0.9
    hardware_qp_updates: int = 64
    active_margin: float = 0.01
    max_active_constraints: int = 32
    barrier_update: str = "decrement"

    def __post_init__(self):
        if not math.isfinite(self.time_limit) or self.time_limit < 0:
            raise ValueError("time_limit must be finite and nonnegative")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 0
        ):
            raise ValueError("max_iterations must be a nonnegative integer")
        if (
            not all(map(math.isfinite, (self.barrier_initial, self.barrier_minimum)))
            or not 0 <= self.barrier_minimum <= self.barrier_initial
        ):
            raise ValueError("Require 0 <= barrier_minimum <= barrier_initial")
        if (
            not math.isfinite(self.initial_regularization_ratio)
            or self.initial_regularization_ratio <= 0
        ):
            raise ValueError("initial_regularization_ratio must be positive")
        if not math.isfinite(self.gate_error_target) or self.gate_error_target < 0:
            raise ValueError("gate_error_target must be finite and nonnegative")
        if not isinstance(self.stop_on_target, bool):
            raise ValueError("stop_on_target must be a boolean")
        if self.objective_target is not None and (
            not math.isfinite(self.objective_target) or self.objective_target < 0
        ):
            raise ValueError("objective_target must be finite and nonnegative")
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.feasibility_tolerance, self.direction_tolerance)
        ):
            raise ValueError("Residual tolerances must be finite and positive")
        if self.curvature_estimator not in ("ratio", "residual", "split"):
            raise ValueError("curvature_estimator must be ratio, residual, or split")
        if self.scale_update not in ("ratio", "step_aware"):
            raise ValueError("scale_update must be ratio or step_aware")
        if self.remainder_feedback not in ("underprediction", "all", "relative"):
            raise ValueError("remainder_feedback must be underprediction, all, or relative")
        if not math.isfinite(self.estimator_decay) or not 0 < self.estimator_decay <= 1:
            raise ValueError("estimator_decay must be in (0, 1]")
        if self.objective_scale is not None and (
            not math.isfinite(self.objective_scale) or self.objective_scale <= 0
        ):
            raise ValueError("objective_scale must be positive when specified")
        if not math.isfinite(self.step_scale) or self.step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if not isinstance(self.use_hardware_constraints, bool):
            raise ValueError("use_hardware_constraints must be a boolean")
        if self.hardware_model not in ("slack", "face"):
            raise ValueError("hardware_model must be slack or face")
        if not math.isfinite(self.slack_fraction) or not 0 < self.slack_fraction < 1:
            raise ValueError("slack_fraction must lie in (0, 1)")
        if (
            isinstance(self.hardware_qp_updates, bool)
            or not isinstance(self.hardware_qp_updates, (int, np.integer))
            or self.hardware_qp_updates < 1
        ):
            raise ValueError("hardware_qp_updates must be a positive integer")
        if not math.isfinite(self.active_margin) or not 0 < self.active_margin < 1:
            raise ValueError("active_margin must be in (0, 1)")
        if (
            isinstance(self.max_active_constraints, bool)
            or not isinstance(self.max_active_constraints, (int, np.integer))
            or self.max_active_constraints < 1
        ):
            raise ValueError("max_active_constraints must be a positive integer")
        if self.barrier_update not in ("decrement", "adaptive"):
            raise ValueError("barrier_update must be decrement or adaptive")
