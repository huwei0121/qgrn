"""Q-GRN iterations, acceptance tests, budgets and diagnostic records."""

from __future__ import annotations

import math
from dataclasses import asdict
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from numpy.typing import ArrayLike

from ._utils import _field, _finite_or_none, _norm_inf, _trace
from .active_constraints import select_tangent_geometry, solve_tangent_direction
from .adaptation import (
    _accepted_ratio_scale,
    _remainder_sample,
    _remainder_update,
    barrier_feedback,
)
from .direction import solve_direction
from .geometry import AffineGeometry
from .options import SolverOptions
from .oracle import (
    _barrier_reduction,
    _normalize,
    _objective_reduction,
    _remainder_packet,
)
from .problem import ConstrainedOracle, ResidualOracle
from .result import OptimizationResult
from .slack_constraints import solve_slack_direction


def minimize(
    problem: ResidualOracle | ConstrainedOracle,
    x0: ArrayLike,
    options: SolverOptions | None = None,
    *,
    started: float | None = None,
) -> OptimizationResult:
    """Minimize a real residual objective using Q-GRN.

    ``problem`` supplies ``residual_oracle(x, order)`` for f=||r||^2, or
    ``grn_oracle(x, derivatives=..., beta=...)`` for a complete physical model
    with constraints and regularizers. See :class:`ResidualProblem` for a
    callable residual/Jacobian adapter. Complex quantities must be realified
    by the problem adapter. Optional affine constraints are homogeneous A x=0;
    general affine offsets require explicit restoration and feasibility methods.

    Initialization, restoration, trial evaluations, rejected attempts and the
    final physical evaluation count towards ``time_limit``. ``started`` is an
    optional perf_counter timestamp for including caller setup. Running numerical
    operations are not interrupted. The result contains the last accepted point
    and the full accepted-iterate history.
    """
    options = SolverOptions() if options is None else options
    if not isinstance(options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    if not callable(getattr(problem, "grn_oracle", None)) and not callable(
        getattr(problem, "residual_oracle", None)
    ):
        raise TypeError("problem must provide residual_oracle or grn_oracle")
    started = perf_counter() if started is None else float(started)
    if not math.isfinite(started):
        raise ValueError("started must be a finite perf_counter timestamp")

    def elapsed() -> float:
        return perf_counter() - started

    def expired() -> bool:
        return elapsed() >= options.time_limit

    y = np.asarray(x0, dtype=float).copy()
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("Initial coordinates must be a finite vector")
    geometry = AffineGeometry(problem, y.size)
    if options.use_hardware_constraints and not hasattr(problem, "hardware_geometry"):
        raise ValueError("use_hardware_constraints requires hardware_geometry")
    y, restoration = geometry.restore(y)
    calls = dict(value=0, jacobian=0, failed=0)
    oracle_seconds = 0.0

    def query(x, derivatives, beta):
        nonlocal oracle_seconds
        calls["jacobian" if derivatives else "value"] += 1
        tick = perf_counter()
        try:
            if not np.all(np.isfinite(x)):
                return SimpleNamespace(valid=False, reason="nonfinite_coordinates")
            if hasattr(problem, "grn_oracle"):
                raw = problem.grn_oracle(x, derivatives=derivatives, beta=beta)
            else:
                raw = problem.residual_oracle(x, order=int(derivatives))
            answer = _normalize(raw, y.size, derivatives)
            calls["failed"] += int(not answer.valid)
            return answer
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            calls["failed"] += 1
            return SimpleNamespace(
                valid=False, reason="oracle_numerical_exception", error=str(error)
            )
        except ValueError as error:
            if str(error) != "SINGULAR_OVERLAP_CHART":
                raise
            calls["failed"] += 1
            return SimpleNamespace(
                valid=False,
                reason="nondifferentiable_unsquared_zero_overlap",
                error=str(error),
            )
        finally:
            oracle_seconds += perf_counter() - tick

    target = options.objective_target if options.stop_on_target else None
    if options.stop_on_target and target is None:
        target = float(getattr(problem, "objective_weight", 1.0)) * 1e-3
    beta, kappa, initial_kappa, initial_curvature = (
        options.barrier_initial,
        None,
        None,
        None,
    )
    delta, objective_scale = 0.0, options.objective_scale
    use_remainders = options.curvature_estimator != "ratio"
    explicit_adapter = hasattr(problem, "grn_oracle")
    history, trials, barrier_updates = [], [], []
    stage_initial_chi = None
    direction_seconds = 0.0
    accepted, stage = 0, 0
    status, reason = "ITERATION_LIMIT", "maximum_accepted_steps"
    for k in range(options.max_iterations + 1):
        a = query(y, True, beta)
        if not a.valid:
            status = (
                "NONDIFFERENTIABLE_POINT"
                if a.reason == "nondifferentiable_unsquared_zero_overlap"
                else "NUMERICAL_FAILURE"
            )
            reason = a.reason
            break
        if options.barrier_update == "adaptive":
            chain_chi = float(np.linalg.norm(geometry.project(a.g + a.gb)))
            if stage_initial_chi is None:
                stage_initial_chi = chain_chi
            if not expired() and k < options.max_iterations:
                decision = barrier_feedback(
                    beta=beta,
                    beta_min=options.barrier_minimum,
                    chi=chain_chi,
                    physical_norm=float(np.linalg.norm(geometry.project(a.g))),
                    barrier_norm=float(np.linalg.norm(geometry.project(a.gb))),
                    stage_initial_chi=stage_initial_chi,
                    stage_accepted=stage,
                )
                decision.update(iteration=k, elapsed=elapsed())
                barrier_updates.append(decision)
                if decision["reduce"]:
                    beta, stage = decision["beta_next"], 0
                    # Refresh derivatives at the reduced barrier parameter.
                    a = query(y, True, beta)
                    if not a.valid:
                        status, reason = "NUMERICAL_FAILURE", a.reason
                        break
                    stage_initial_chi = float(np.linalg.norm(geometry.project(a.g + a.gb)))
                    decision.update(new_stage_chi=stage_initial_chi, requery_elapsed=elapsed())
        if use_remainders:
            remainder_F, remainder_J = _remainder_packet(a, explicit_adapter=explicit_adapter)
            if objective_scale is None:
                # Fix this scale at initialization, preserving covariance
                # under a uniform objective rescaling.
                objective_scale = (
                    a.f
                    if a.f > 0
                    else max(
                        2.0 * float(np.sum(a.Jr * a.Jr)) * options.step_scale**2,
                        np.finfo(float).tiny,
                    )
                )
        step_geometry, active_info, hardware = geometry, None, None
        if options.use_hardware_constraints and not expired():
            hardware = problem.hardware_geometry(y)
            if options.hardware_model == "face":
                step_geometry, active_info = select_tangent_geometry(
                    geometry,
                    hardware,
                    a.g + a.gb,
                    margin=options.active_margin,
                    max_active=options.max_active_constraints,
                )
            else:
                active_info = dict(
                    model="linearized_slack",
                    inequalities=len(hardware.slack),
                    fraction=options.slack_fraction,
                    min_slack=float(np.min(hardware.slack)),
                )
        chi = float(np.linalg.norm(step_geometry.project(a.g + a.gb)))
        if not math.isfinite(chi):
            status, reason = "NUMERICAL_FAILURE", "nonfinite_gradient_norm"
            break
        feasibility = geometry.feasibility(y)
        target_met = (
            options.stop_on_target
            and a.gate_error <= options.gate_error_target
            and a.f <= target
            and feasibility <= options.feasibility_tolerance
        )
        if kappa is None and not target_met and not expired() and k < options.max_iterations:
            initial_curvature = max(
                (2.0 * np.sum(a.Jr * a.Jr) + _trace(a.H0)) / max(1, min(a.Jr.shape[0], y.size)),
                1e-9,
            )
            mu0 = options.initial_regularization_ratio * initial_curvature
            initial_kappa = mu0 * mu0 / max(chi, 1e-30)
            kappa = initial_kappa
            if not math.isfinite(kappa) or kappa <= 0:
                status, reason = "NUMERICAL_FAILURE", "invalid_initial_curvature_scale"
                break
        mu_now = None if kappa is None else max(delta + math.sqrt(kappa * max(chi, 1e-30)), 1e-12)
        history.append(
            dict(
                k=k,
                iteration=k,
                elapsed=elapsed(),
                objective=a.f,
                f=a.f,
                gate_error=a.gate_error,
                ell=a.ell,
                h=a.h,
                beta=beta,
                chi=chi,
                y=y.copy(),
                x=y.copy(),
                feasibility=_norm_inf(geometry.A @ y),
                physical_control_feasibility=feasibility,
                kappa=kappa,
                initial_kappa=initial_kappa,
                mu=mu_now,
                kind="accepted",
            )
        )
        if use_remainders:
            history[-1].update(
                delta=delta,
                estimator=options.curvature_estimator,
                objective_scale=objective_scale,
            )
        if options.use_hardware_constraints:
            history[-1].update(
                active=active_info,
                chain_chi=float(np.linalg.norm(geometry.project(a.g + a.gb))),
            )
        if target_met:
            status, reason = (
                "TARGET_REACHED",
                "modeled_gate_objective_and_control_feasibility",
            )
            break
        if expired():
            status, reason = "TIME_LIMIT", "solve_budget_checkpoint"
            break
        if k == options.max_iterations:
            break
        committed, budget_stop, numerical_stop = False, False, False
        unresolved_attempts = 0
        for attempt in range(1, 31):
            if expired():
                budget_stop = True
                break
            if not math.isfinite(kappa) or kappa <= 0 or not math.isfinite(delta):
                status, reason = "NUMERICAL_FAILURE", "nonfinite_regularization_scale"
                break
            kappa_before, delta_before = kappa, delta
            mu = max(delta + math.sqrt(kappa * max(chi, 1e-30)), 1e-12)
            direction, direction_error = None, None
            direction_tick = perf_counter()
            try:
                if options.use_hardware_constraints and options.hardware_model == "slack":
                    direction = solve_slack_direction(
                        a,
                        mu,
                        geometry,
                        hardware,
                        solve_direction,
                        fraction=options.slack_fraction,
                        max_active=options.max_active_constraints,
                        max_updates=options.hardware_qp_updates,
                    )
                elif options.use_hardware_constraints:
                    direction = solve_tangent_direction(a, mu, step_geometry, solve_direction)
                else:
                    direction = solve_direction(a, mu, geometry)
            except Exception as error:
                direction_error = f"{type(error).__name__}: {error}"
            direction_cost = perf_counter() - direction_tick
            direction_seconds += direction_cost
            direction_accurate = (
                direction is not None
                and math.isfinite(direction.tangent_relative_residual)
                and direction.tangent_relative_residual <= options.direction_tolerance
            )
            if direction is not None and hasattr(direction, "qp_exact"):
                # Accept an inexact QP direction only when it satisfies the
                # linearized inequalities and decreases the quadratic model.
                direction_accurate = (
                    math.isfinite(direction.model_change)
                    and direction.model_change < 0
                    and direction.linearized_constraint_violation <= options.direction_tolerance
                    and (not direction.qp_exact or direction_accurate)
                )
            valid = (
                direction_error is None
                and math.isfinite(direction.slope)
                and direction.slope < 0
                and np.all(np.isfinite(direction.p))
                and math.isfinite(direction.tangent_relative_residual)
                and direction_accurate
                and direction.constraint_residual
                <= options.direction_tolerance * max(1.0, _norm_inf(direction.p))
            )
            unrepresentable = bool(
                direction_error is None
                and direction is not None
                and np.all(np.isfinite(direction.p))
                and np.array_equal(y + direction.p, y)
            )
            if not valid or unrepresentable:
                numerical_stop = unrepresentable
                if not numerical_stop:
                    kappa *= 100.0
                if use_remainders:
                    delta *= 1.0 if numerical_stop else 10.0
                trials.append(
                    dict(
                        k=k,
                        attempt=attempt,
                        elapsed=elapsed(),
                        mu=mu,
                        chi=chi,
                        kappa_before=kappa_before,
                        kappa_after=kappa,
                        initial_kappa=initial_kappa,
                        alpha=0.0,
                        rho=None,
                        actual=None,
                        pred=None,
                        accepted=False,
                        slope=None if direction is None else direction.slope,
                        direction_residual=None
                        if direction is None
                        else direction.relative_residual,
                        tangent_relative_residual=None
                        if direction is None
                        else direction.tangent_relative_residual,
                        constraint_residual=None
                        if direction is None
                        else direction.constraint_residual,
                        reason="unrepresentable_direction"
                        if numerical_stop
                        else "direction_exception"
                        if direction_error
                        else "invalid_or_inaccurate_direction",
                        error=direction_error,
                        direction_seconds=direction_cost,
                    )
                )
                if direction is not None and hasattr(direction, "qp_exact"):
                    trials[-1].update(
                        {
                            key: getattr(direction, key)
                            for key in (
                                "qp_exact",
                                "qp_status",
                                "qp_updates",
                                "qp_inverse_calls",
                                "model_change",
                                "linearized_constraint_violation",
                                "qp_stationarity_residual",
                                "qp_dual_residual",
                                "qp_complementarity_residual",
                            )
                        }
                    )
                if use_remainders:
                    trials[-1].update(
                        delta_before=delta_before,
                        delta_after=delta,
                        estimator=options.curvature_estimator,
                        feedback=options.remainder_feedback,
                        remainder_samples=[],
                        remainder_update_decision="unrepresentable_direction"
                        if numerical_stop
                        else "invalid_direction_escalation",
                    )
                if numerical_stop:
                    break
                continue
            alpha, rho, actual, pred = 1.0, -math.inf, -math.inf, -math.inf
            remainder_samples = []
            invalid_trial_reasons = {}
            accepted_elapsed = None
            ratio_feedback_rule = None
            informative_rejection, numerical_witness = False, False
            reduction_noise, reduction_method, barrier_method = None, None, None
            for _ in range(25):
                if expired():
                    budget_stop = True
                    break
                p = alpha * direction.p
                trial = query(y + p, False, beta)
                if trial.valid:
                    jp, jregp = a.Jr @ p, a.Jreg @ p
                    complete_J = _field(a.raw, "remainder_jacobian")
                    residual_step = (
                        (complete_J(p) if callable(complete_J) else complete_J @ p)
                        if complete_J is not None
                        else math.sqrt(2.0) * jp
                        if not explicit_adapter
                        else None
                    )
                    (
                        physical_change,
                        physical_noise,
                        residual_unresolved,
                        reduction_method,
                    ) = _objective_reduction(
                        a,
                        trial,
                        explicit_adapter=explicit_adapter,
                        residual_step=residual_step,
                    )
                    (
                        barrier_change,
                        barrier_noise,
                        barrier_unresolved,
                        barrier_method,
                    ) = _barrier_reduction(a, trial)
                    linear, quadratic, regularizer = (
                        float(a.g @ p),
                        float(jp @ jp),
                        float(jregp @ jregp) / 2,
                    )
                    pred = math.fsum((-linear, -quadratic, -regularizer, barrier_change))
                    actual = math.fsum((physical_change, barrier_change))
                    reduction_noise = (
                        physical_noise
                        + barrier_noise
                        + 8
                        * np.finfo(float).eps
                        * (abs(linear) + abs(quadratic) + abs(regularizer) + abs(barrier_change))
                    )
                    rho = actual / pred if math.isfinite(pred) and pred > 0 else -math.inf
                    if use_remainders:
                        trial_F, _ = _remainder_packet(trial, explicit_adapter=explicit_adapter)
                        Js = remainder_J(p) if callable(remainder_J) else remainder_J @ p
                        model_noise = (
                            32
                            * np.finfo(float).eps
                            * max(
                                abs(a.f),
                                abs(trial.f),
                                abs(a.barrier),
                                abs(trial.barrier),
                                abs(float(a.g @ p)),
                                abs(pred),
                                np.finfo(float).tiny,
                            )
                        )
                        sample = _remainder_sample(
                            remainder_F,
                            trial_F,
                            Js,
                            p,
                            objective_scale=objective_scale,
                            step_scale=options.step_scale,
                            model_error=pred - actual,
                            model_noise=model_noise,
                        )
                        sample.update(
                            alpha=alpha,
                            elapsed=elapsed(),
                            prediction=pred,
                            accepted=False,
                            update_eligible=False,
                            update_decision="rejected_line_search_sample",
                        )
                        remainder_samples.append(sample)
                    if math.isfinite(actual) and math.isfinite(rho) and rho >= 0.1 and actual > 0:
                        y += p
                        accepted += 1
                        stage += 1
                        committed = True
                        accepted_elapsed = elapsed()
                        if use_remainders:
                            sample["accepted"] = True
                            kappa, delta = _remainder_update(sample, kappa, delta, options)
                        else:
                            ratio_feedback_rule = (
                                "ratio" if invalid_trial_reasons else options.scale_update
                            )
                            kappa = _accepted_ratio_scale(kappa, rho, ratio_feedback_rule, alpha)
                        break
                    # Only rejected steps enter this diagnostic. A resolved
                    # positive decrease uses the reduction-ratio test,
                    # regardless of its size relative to the full objective.
                    informative_rejection |= bool(
                        abs(actual) > reduction_noise or abs(pred) > reduction_noise
                    )
                    # A tiny backtracked step can be unrepresentable even
                    # when the full direction is useful. Only the full trial
                    # supplies a numerical witness; no small-step tolerance.
                    numerical_witness |= bool(
                        alpha == 1.0
                        and (
                            np.array_equal(y + p, y) or (residual_unresolved and barrier_unresolved)
                        )
                    )
                else:
                    invalid_trial_reasons[trial.reason] = (
                        invalid_trial_reasons.get(trial.reason, 0) + 1
                    )
                    if use_remainders:
                        remainder_samples.append(
                            dict(
                                alpha=alpha,
                                elapsed=elapsed(),
                                accepted=False,
                                informative=False,
                                reason=trial.reason,
                                update_eligible=False,
                                update_decision="invalid_oracle_no_sample",
                            )
                        )
                alpha *= 0.5
            unresolved_attempt = bool(
                not committed
                and not budget_stop
                and not informative_rejection
                and not invalid_trial_reasons
            )
            unresolved_attempts = unresolved_attempts + 1 if unresolved_attempt else 0
            # Retry an unresolved search once with stronger damping. Stop if
            # the full step is unrepresentable or two consecutive searches
            # cannot resolve any objective change.
            numerical_stop = bool(
                unresolved_attempt and (numerical_witness or unresolved_attempts >= 2)
            )
            if not committed and not budget_stop and not numerical_stop:
                kappa *= 16
                if use_remainders:
                    delta *= 4
            trials.append(
                dict(
                    k=k,
                    attempt=attempt,
                    elapsed=elapsed(),
                    mu=mu,
                    chi=chi,
                    kappa_before=kappa_before,
                    kappa_after=kappa,
                    initial_kappa=initial_kappa,
                    alpha=alpha,
                    rho=_finite_or_none(rho),
                    actual=_finite_or_none(actual),
                    pred=_finite_or_none(pred),
                    accepted=committed,
                    slope=direction.slope,
                    direction_residual=direction.relative_residual,
                    tangent_relative_residual=direction.tangent_relative_residual,
                    constraint_residual=direction.constraint_residual,
                    reason="accepted"
                    if committed
                    else "budget_checkpoint"
                    if budget_stop
                    else (
                        "numerically_unresolved_line_search"
                        if numerical_witness
                        else "repeated_unresolved_line_search"
                    )
                    if numerical_stop
                    else "line_search_unresolved_retry"
                    if unresolved_attempt
                    else "line_search_rejected",
                    error=None,
                    direction_seconds=direction_cost,
                    invalid_trial_reasons=invalid_trial_reasons,
                    accepted_elapsed=accepted_elapsed,
                    beta=beta,
                    reduction_method=reduction_method,
                    barrier_difference_method=barrier_method,
                    reduction_noise=reduction_noise,
                    informative_rejection=informative_rejection,
                    numerical_witness=numerical_witness,
                    unresolved_attempts=unresolved_attempts,
                    ratio_feedback_rule=ratio_feedback_rule,
                    ratio_feedback_domain_fallback=bool(
                        committed
                        and invalid_trial_reasons
                        and options.curvature_estimator == "ratio"
                        and options.scale_update == "step_aware"
                    ),
                    step_norm=direction.step_norm,
                    accepted_step_norm=alpha * direction.step_norm if committed else None,
                    damping_energy=0.5 * mu * float(direction.p @ direction.p),
                    accepted_damping_energy=(
                        0.5 * mu * alpha * alpha * float(direction.p @ direction.p)
                        if committed
                        else None
                    ),
                    actual_resolved=(
                        None if reduction_noise is None else bool(abs(actual) > reduction_noise)
                    ),
                )
            )
            if options.use_hardware_constraints:
                trials[-1].update(
                    active_selected=len(direction.active_constraint_ids),
                    active_constraint_ids=direction.active_constraint_ids,
                    active_released=direction.active_released,
                    effective_tangent_chi=float(
                        np.linalg.norm(direction.active_geometry.project(a.g + a.gb))
                    ),
                )
                if hasattr(direction, "qp_exact"):
                    trials[-1].update(
                        {
                            key: getattr(direction, key)
                            for key in (
                                "qp_exact",
                                "qp_status",
                                "qp_updates",
                                "qp_inverse_calls",
                                "model_change",
                                "linearized_constraint_violation",
                                "qp_stationarity_residual",
                                "qp_dual_residual",
                                "qp_complementarity_residual",
                            )
                        }
                    )
            if use_remainders:
                trials[-1].update(
                    delta_before=delta_before,
                    delta_after=delta,
                    estimator=options.curvature_estimator,
                    feedback=options.remainder_feedback,
                    remainder_samples=remainder_samples,
                    remainder_update_decision=(
                        remainder_samples[-1]["update_decision"]
                        if committed
                        else "budget_checkpoint"
                        if budget_stop
                        else "numerically_unresolved_line_search"
                        if numerical_stop
                        else "rejected_direction_escalation"
                    ),
                )
            if committed:
                if (
                    options.barrier_update == "decrement"
                    and beta > options.barrier_minimum
                    and (-alpha * direction.slope < 0.1 * beta or stage >= 30)
                ):
                    beta, stage = max(options.barrier_minimum, 0.1 * beta), 0
                break
            if budget_stop or numerical_stop:
                break
        if not committed:
            if status != "NUMERICAL_FAILURE":
                status = (
                    "TIME_LIMIT"
                    if budget_stop or expired()
                    else "NUMERICAL_STAGNATION"
                    if numerical_stop
                    else "MODEL_FAILURE"
                )
                reason = (
                    "solve_budget_checkpoint"
                    if status == "TIME_LIMIT"
                    else trials[-1]["reason"]
                    if numerical_stop
                    else "direction_or_model_attempts_exhausted"
                )
            break
    final = query(y, False, 0.0)
    if not final.valid:
        raise RuntimeError(f"Cannot return a finite modeled trajectory: {final.reason}")
    feasibility = geometry.feasibility(y)
    quality = (
        (
            final.gate_error <= options.gate_error_target
            and final.f <= target
            and feasibility <= options.feasibility_tolerance
        )
        if options.stop_on_target
        else None
    )
    parameters = dict(
        asdict(options),
        objective_target=target,
        method="qgrn",
        adaptive_rule=(
            "curvature_initialized_kappa_step_feedback"
            if options.scale_update == "step_aware"
            else "curvature_initialized_kappa_rho_feedback"
        ),
        reduction_rule="stable complete-residual and log-slack differences when available",
        numerical_exit_rule="rejected unresolved line search or unrepresentable direction; no target, gradient, or step tolerance",
    )
    if use_remainders:
        parameters.update(
            adaptive_rule=f"curvature_initialized_{options.curvature_estimator}_remainder_{options.remainder_feedback}",
            objective_scale_used=objective_scale,
            residual_contract="f=0.5*norm(F)^2; barrier excluded",
            rejected_update="invalid direction: kappa *= 100, delta *= 10; rejected search: kappa *= 16, delta *= 4",
            accepted_update="max(observed, decay*history); "
            + {
                "all": "unsigned informative samples",
                "underprediction": "signed underprediction eligibility",
                "relative": "E > max(noise, 0.25*pred) eligibility",
            }[options.remainder_feedback],
        )
    parameters.update(
        barrier_rule="reduce beta by 0.1 when accepted linear decrease < 0.1*beta or after 30 accepted steps"
        if options.barrier_update == "decrement"
        else "fixed-beta affine-tangent stationarity progress or gradient balance; minimum 3 accepted steps",
        hardware_rule=(
            "linearized hardware inequalities Cp <= fraction*slack; reused inverse and bounded feasible QP"
            if options.hardware_model == "slack"
            else "near-face tangent with non-descent fallback"
        )
        if options.use_hardware_constraints
        else "oracle-supplied barrier and nonlinear feasibility checks",
    )
    extras = problem.return_fields(y, final.raw) if hasattr(problem, "return_fields") else {}
    checks = dict(
        extras.pop("checks", {}),
        primal=feasibility,
        modeled_targets_met=quality,
        independent_physics_verified=False,
        kkt_certified=False,
    )
    complete = elapsed()
    result = dict(
        status=status,
        termination_reason=reason,
        failure_detail=None,
        x=y.copy(),
        y=y.copy(),
        objective=final.f,
        f=final.f,
        gate_error=final.gate_error,
        ell=final.ell,
        history=history,
        trials=trials,
        accepted=accepted,
        iterations=accepted,
        restoration=restoration,
        parameters=parameters,
        initial_kappa=initial_kappa,
        initial_curvature=initial_curvature,
        final_kappa=kappa,
        calls=calls,
        oracle_seconds=oracle_seconds,
        direction_seconds=direction_seconds,
        barrier_updates=barrier_updates,
        elapsed=complete,
        within_budget=complete <= options.time_limit,
        final=final.raw,
        clock_started=started,
        checks=checks,
        implementation=(
            "Q-GRN with accepted-step regularization feedback"
            if options.scale_update == "step_aware"
            else "Q-GRN with reduction-ratio regularization feedback"
        ),
    )
    if use_remainders:
        result.update(
            final_delta=delta,
            objective_scale=objective_scale,
            implementation=f"Q-GRN with {options.curvature_estimator} remainder and {options.remainder_feedback} feedback",
        )
    if options.use_hardware_constraints or options.barrier_update != "decrement":
        result["implementation"] += "; hardware and barrier settings recorded in parameters"
    if set(result).intersection(extras):
        raise ValueError("Problem return_fields cannot replace solver records")
    result.update(extras)
    return OptimizationResult(result)
