"""Curvature observations, accepted-step scale updates and barrier continuation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ._utils import _finite_or_none


def remainder_feedback(
    sample: Mapping,
    *,
    feedback: str,
    prediction: float | None = None,
    relative_threshold: float = 0.25,
) -> dict:
    """Decide whether an accepted sample may refresh the regularization scale.

    ``all`` retains the unsigned residual AdaN+-style observation. For
    ``relative``, an informative sample is used only when
    ``model_error > max(model_noise, relative_threshold * prediction)``. Thus
    ``relative_threshold=.25`` asks whether the model loses more than a quarter
    of its predicted decrease, using the *actual accepted step*. It is the
    ``rho < .75`` threshold when noise does not dominate. An ineligible
    sample still permits explicit history decay in the calling solver.

    The units are objective units for error, noise, and prediction. This helper
    does not change the conversion from residual curvature to kappa.
    """
    if feedback not in ("all", "underprediction", "relative"):
        raise ValueError("Unknown remainder feedback mode")
    if not math.isfinite(relative_threshold) or not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must be in [0, 1)")
    if not sample["informative"]:
        return dict(eligible=False, reason="history_decay_unresolved_sample", cutoff=None)
    if feedback == "all":
        return dict(eligible=True, reason="observed_remainder", cutoff=None)
    error, noise = float(sample["model_error"]), float(sample["model_noise"])
    if not math.isfinite(error) or not math.isfinite(noise) or noise < 0:
        raise ValueError("Informative samples require finite error and nonnegative noise")
    cutoff = noise
    if feedback == "relative":
        if prediction is None or not math.isfinite(prediction) or prediction <= 0:
            raise ValueError(
                "Relative feedback requires a positive finite accepted-step prediction"
            )
        cutoff = max(cutoff, relative_threshold * prediction)
    eligible = error > cutoff
    reason = (
        "observed_remainder"
        if eligible
        else "history_decay_small_relative_defect"
        if feedback == "relative"
        else "history_decay_favorable_model"
    )
    return dict(eligible=bool(eligible), reason=reason, cutoff=float(cutoff))


def barrier_feedback(
    *,
    beta: float,
    beta_min: float,
    chi: float,
    physical_norm: float,
    barrier_norm: float,
    stage_initial_chi: float,
    stage_accepted: int,
    min_stage_accepted: int = 3,
    stage_reduction: float = 0.25,
    balance_tolerance: float = 0.1,
    beta_reduction: float = 0.1,
) -> dict:
    """Propose a barrier decrease from derivatives evaluated at fixed beta.

    Let P be the affine-chain tangent projector. Inputs must be
    ``chi=||P(g+gb)||``, ``physical_norm=||Pg||``, and
    ``barrier_norm=||Pgb||`` at the same current point and beta.
    ``stage_initial_chi`` is recorded after the preceding beta change; the
    accepted-step counter counts only steps at that beta. The minimum count is
    a minimum stage length; a gradient condition is also required.

    Decrease beta if the stage has accumulated at least three accepted steps
    and either its gradient norm has decreased by a factor four or its current
    stationarity residual is at most 10 percent of the two gradient norms' sum.
    The first condition measures stage progress; the second measures balance
    of the physical and barrier gradients. With no barrier-gradient cancellation,
    the latter is normally one and does not fire merely on small raw units.

    Both tests compare quantities with gradient units and measure relative
    progress. They are continuation criteria, not fixed-barrier optimality
    tolerances. Use the affine-chain projector throughout: projecting onto an
    active hardware face would remove the corresponding normal residuals.
    """
    values = dict(
        beta=beta,
        beta_min=beta_min,
        chi=chi,
        physical_norm=physical_norm,
        barrier_norm=barrier_norm,
        stage_initial_chi=stage_initial_chi,
    )
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if beta_min > beta:
        raise ValueError("beta_min must not exceed beta")
    for name, value in (
        ("stage_accepted", stage_accepted),
        ("min_stage_accepted", min_stage_accepted),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if name == "min_stage_accepted" else 0)
        ):
            raise ValueError(f"{name} must be an integer in its allowed range")
    for name, value in (
        ("stage_reduction", stage_reduction),
        ("balance_tolerance", balance_tolerance),
        ("beta_reduction", beta_reduction),
    ):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")

    # Ratios are diagnostic. Direct inequalities below avoid arbitrary unit
    # floors, so uniform objective/coordinate rescaling leaves decisions intact.
    scale = physical_norm + barrier_norm
    relative_chi = chi / scale if scale > 0 else (0.0 if chi == 0 else None)
    progress_ratio = (
        chi / stage_initial_chi if stage_initial_chi > 0 else (0.0 if chi == 0 else None)
    )
    progressed = chi <= stage_reduction * stage_initial_chi
    balanced = chi <= balance_tolerance * scale
    if beta <= beta_min:
        reduce, reason = False, "beta_at_floor"
    elif stage_accepted < min_stage_accepted:
        reduce, reason = False, "insufficient_stage_steps"
    elif balanced:
        reduce, reason = True, "balanced_barrier_gradient"
    elif progressed:
        reduce, reason = True, "stage_stationarity_progress"
    else:
        reduce, reason = False, "continue_fixed_barrier_stage"
    beta_next = max(beta_min, beta_reduction * beta) if reduce else beta
    return dict(
        beta_before=float(beta),
        beta_next=float(beta_next),
        reduce=bool(reduce),
        reason=reason,
        chi=float(chi),
        stage_initial_chi=float(stage_initial_chi),
        physical_norm=float(physical_norm),
        barrier_norm=float(barrier_norm),
        gradient_scale=float(scale),
        relative_chi=relative_chi,
        stage_progress_ratio=progress_ratio,
        stage_accepted=stage_accepted,
        min_stage_accepted=min_stage_accepted,
        stage_reduction=float(stage_reduction),
        balance_tolerance=float(balance_tolerance),
        beta_reduction=float(beta_reduction),
        original_kkt_certificate=False,
    )


def _accepted_ratio_scale(kappa, rho, rule, alpha=1.0):
    """Update the next GRN scale, retaining mu=sqrt(kappa*chi).

    In the damping-dominated regime p approximately equals -g/mu. A model
    backtrack alpha then suggests mu/alpha at the same chi; the factor 1/sqrt(2)
    permits a subsequent decrease in regularization. At fixed chi, the mu ratio
    lies in [1/2, 4]. A domain-limited line search uses reduction-ratio feedback
    because its step length also reflects physical feasibility.
    """
    if rule == "step_aware":
        return max(kappa * min(16.0, max(0.25, 0.5 / (alpha * alpha))), 1e-24)
    return max(kappa / 4, 1e-24) if rho > 0.75 else 4 * kappa if rho < 0.25 else kappa


def _remainder_sample(
    F, trial_F, Js, step, *, objective_scale, step_scale, model_error, model_noise
):
    """Measure the Gauss-Newton model error on an evaluated step.

    For epsilon=F_new-F-DF*s, the exact identity is E=a+b with
    a=<F,epsilon>, b=<DF*s,epsilon>+.5*||epsilon||^2. The sample is
    uninformative when subtraction is below its floating point noise estimate.
    """
    F, trial_F, Js = (np.asarray(v, dtype=float) for v in (F, trial_F, Js))
    if F.shape != trial_F.shape or F.shape != Js.shape:
        raise ValueError("Remainder sample residual dimensions mismatch")
    t = float(np.linalg.norm(step))
    epsilon = trial_F - F - Js
    ne = float(np.linalg.norm(epsilon))
    noise = 32 * np.finfo(float).eps * sum(float(np.linalg.norm(v)) for v in (F, trial_F, Js))
    a, b = float(F @ epsilon), float(Js @ epsilon) + 0.5 * ne * ne
    finite = all(math.isfinite(v) for v in (t, ne, noise, a, b, model_error, model_noise))
    resolved = finite and t > 0 and math.isfinite(t * t * t) and t * t * t > 0
    informative = resolved and ne > noise
    mf = ne / (t * t) if informative else 0.0
    dr = 2 * max(a, 0.0) / (t * t) if informative else 0.0
    kr = 6 * max(b, 0.0) / (t * t * t) if informative else 0.0
    kc = math.sqrt(objective_scale) / step_scale * mf
    if not all(math.isfinite(v) for v in (mf, dr, kr, kc)):
        informative = False
        mf = dr = kr = kc = 0.0
    identity_error = (a + b) - model_error
    # Check that the complete residual model includes the same physical costs
    # as the objective and quadratic prediction.
    identity_tolerance = max(
        8 * model_noise, 1e-9 * max(abs(a + b), abs(model_error), np.finfo(float).tiny)
    )
    if finite and abs(identity_error) > identity_tolerance:
        raise ValueError(
            "Remainder residual/Jacobian does not match the solver's objective quadratic model"
        )
    reason = (
        "nonfinite_sample"
        if not finite
        else "unresolved_step"
        if not resolved
        else "unresolved_remainder"
        if ne <= noise
        else "informative"
        if informative
        else "nonfinite_estimate"
    )
    return dict(
        informative=informative,
        reason=reason,
        step_norm=_finite_or_none(t),
        remainder_norm=_finite_or_none(ne),
        remainder_noise=_finite_or_none(noise),
        a=_finite_or_none(a),
        b=_finite_or_none(b),
        residual_scale=mf,
        residual_kappa=kc,
        split_delta=dr,
        split_kappa=kr,
        model_error=_finite_or_none(model_error),
        model_noise=_finite_or_none(model_noise),
        identity_error=_finite_or_none(identity_error),
        identity_tolerance=identity_tolerance,
        split_bound=(0.5 * dr * t * t + kr * t * t * t / 6) if informative else None,
    )


def _remainder_update(sample, kappa, delta, options):
    """Combine an accepted remainder observation with a decayed scale estimate.

    Unresolved samples contribute no observation; the estimate then decays by
    estimator_decay. Rejected searches increase the scale in the outer solver.
    """
    eligible = sample["informative"] and (
        options.remainder_feedback == "all" or sample["model_error"] > sample["model_noise"]
    )
    if options.remainder_feedback == "relative":
        decision = remainder_feedback(
            sample, feedback="relative", prediction=sample.get("prediction")
        )
        eligible = decision["eligible"]
        sample.update(relative_cutoff=decision["cutoff"], relative_decision=decision["reason"])
    observed_kappa = (
        sample["residual_kappa"]
        if options.curvature_estimator == "residual"
        else sample["split_kappa"]
    )
    observed_delta = sample["split_delta"] if options.curvature_estimator == "split" else 0.0
    kappa_next = max(observed_kappa if eligible else 0.0, options.estimator_decay * kappa, 1e-24)
    delta_next = (
        max(observed_delta if eligible else 0.0, options.estimator_decay * delta)
        if options.curvature_estimator == "split"
        else 0.0
    )
    sample.update(
        update_eligible=bool(eligible),
        update_decision=(
            "observed_remainder"
            if eligible
            else "history_decay_unresolved_sample"
            if not sample["informative"]
            else "history_decay_favorable_model"
        ),
    )
    if options.remainder_feedback == "relative" and not eligible and sample["informative"]:
        sample["update_decision"] = "history_decay_small_relative_defect"
    return kappa_next, delta_next
