"""Option and record adapters for the GRNOptions/solve_grn API.

Applications can migrate to qgrn.minimize and qgrn.SolverOptions incrementally.
This module translates option names and diagnostic records while preserving
the GRNOptions defaults, including target stopping and reduction-ratio updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .options import SolverOptions
from .solver import minimize

_OPTION_NAMES = {
    "seconds": "time_limit",
    "maxiter": "max_iterations",
    "beta_initial": "barrier_initial",
    "beta_min": "barrier_minimum",
    "initial_mu_ratio": "initial_regularization_ratio",
    "target_gate": "gate_error_target",
    "target_objective": "objective_target",
    "estimator": "curvature_estimator",
    "ratio_update": "scale_update",
    "feedback": "remainder_feedback",
    "hardware_tangent": "use_hardware_constraints",
    "max_active": "max_active_constraints",
}


@dataclass(frozen=True)
class GRNOptions:
    """Compatibility options, translated and validated by SolverOptions."""

    seconds: float = 60.0
    maxiter: int = 3000
    beta_initial: float = 1e-3
    beta_min: float = 1e-9
    initial_mu_ratio: float = 0.01
    target_gate: float = 1e-3
    target_objective: float | None = None
    stop_on_target: bool = True
    feasibility_tolerance: float = 1e-6
    direction_tolerance: float = 1e-6
    estimator: str = "ratio"
    ratio_update: str = "legacy"
    feedback: str = "underprediction"
    estimator_decay: float = 0.5
    objective_scale: float | None = None
    step_scale: float = 1.0
    hardware_tangent: bool = False
    hardware_model: str = "slack"
    slack_fraction: float = 0.9
    hardware_qp_updates: int = 64
    active_margin: float = 0.01
    max_active: int = 32
    barrier_update: str = "legacy"

    def __post_init__(self):
        self.to_options()

    def to_options(self) -> SolverOptions:
        """Convert field names and update-rule aliases to SolverOptions."""
        values = asdict(self)
        translated = {_OPTION_NAMES.get(key, key): value for key, value in values.items()}
        if translated["scale_update"] == "legacy":
            translated["scale_update"] = "ratio"
        if translated["barrier_update"] == "legacy":
            translated["barrier_update"] = "decrement"
        return SolverOptions(**translated)


def solve_grn(task, start, options: GRNOptions | None = None, *, started=None):
    """Run Q-GRN and return records using the GRNOptions field names."""
    options = GRNOptions() if options is None else options
    if not isinstance(options, GRNOptions):
        raise TypeError("options must be GRNOptions")
    result = dict(minimize(task, start, options.to_options(), started=started))
    for compatibility_name, public in _OPTION_NAMES.items():
        if public in result["parameters"]:
            result["parameters"][compatibility_name] = result["parameters"].pop(public)
    result["parameters"]["ratio_update"] = options.ratio_update
    result["parameters"]["barrier_update"] = options.barrier_update
    result["parameters"]["method"] = "grn"
    for trial in result["trials"]:
        if trial.get("ratio_feedback_rule") == "ratio":
            trial["ratio_feedback_rule"] = "legacy"
    return result
