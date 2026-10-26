"""Strict-interior hardware steps with a bounded primal feasible active set.

The model is min g'p + p'Bp/2 subject to Ap=0, Cp <= fraction*slack.
The base GRN inverse action is reused; only the working-set Schur matrix is
dense. Iteration and working-set limits are reported with the returned step's
feasibility, model decrease and KKT residuals. The nonlinear oracle checks
hardware feasibility during backtracking.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import linalg, sparse


def _norm_inf(value):
    return float(np.max(np.abs(value), initial=0.0))


def solve_slack_direction(
    a, mu, geometry, hardware, solve, *, fraction=0.9, max_active=32, max_updates=64
):
    """Solve a slack-aware GRN model, with explicit approximation diagnostics.

    ``solve(..., expose_inverse=True)`` must return the chain-constrained GRN
    direction, ``inverse(rhs)=K_B rhs`` and ``model_product(p)=Bp``. The algorithm
    starts at feasible p=0 and moves toward each equality-constrained minimizer
    until a new row blocks the segment. Negative model multipliers release rows.
    The descent interpretation assumes B is positive definite on ker(A).

    All hardware rows are checked, including when the working-set budget binds.
    ``qp_exact`` records whether the linearized-QP KKT residual checks passed.
    These checks apply to this quadratic subproblem only.
    Returned multipliers use the original, unnormalized hardware rows.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    if max_active < 0 or max_updates < 1:
        raise ValueError("Invalid working-set budgets")
    slack = np.asarray(hardware.slack, dtype=float)
    raw_C = sparse.csr_matrix(hardware.jacobian, dtype=float)
    if (
        slack.ndim != 1
        or raw_C.shape != (slack.size, geometry.n)
        or not np.all(np.isfinite(slack))
        or np.any(slack <= 0)
        or not np.all(np.isfinite(raw_C.data))
    ):
        raise ValueError("Slack direction requires finite, strictly interior hardware geometry")
    ids = np.asarray(hardware.ids)
    if ids.shape != slack.shape:
        raise ValueError("Hardware IDs must correspond to every inequality")

    free = solve(a, mu, geometry, expose_inverse=True)
    p0 = np.asarray(free.p, dtype=float)
    gradient = np.asarray(a.g + a.gb, dtype=float)
    if p0.shape != (geometry.n,) or not np.all(np.isfinite(p0)):
        raise ValueError("Invalid free GRN direction")
    # Row normalization prevents hardware units from changing Schur conditioning.
    row_norm = np.sqrt(np.asarray(raw_C.multiply(raw_C).sum(axis=1)).ravel())
    row_norm[row_norm == 0] = 1.0
    C = sparse.diags(1.0 / row_norm, format="csr") @ raw_C
    b = fraction * slack / row_norm
    cp0 = np.asarray(C @ p0).ravel()
    p, cp = np.zeros_like(p0), np.zeros_like(b)
    working, released = [], []
    columns = {}
    inverse_calls, updates = 0, 0
    status, exact = "update_limit", False
    lam = np.empty(0)
    gradient_norm = max(float(np.linalg.norm(gradient)), 1e-30)
    tangent_norm = max(float(np.linalg.norm(geometry.project(gradient))), 1e-30)
    multiplier_tolerance = 1e-10 * gradient_norm

    for updates in range(1, max_updates + 1):
        if working:
            W = np.column_stack([columns[i] for i in working])
            CW = C[working]
            schur = np.asarray(CW @ W)
            schur = 0.5 * (schur + schur.T)
            try:
                factor = linalg.cho_factor(schur, lower=True, check_finite=False)
                lam = linalg.cho_solve(factor, cp0[working] - b[working], check_finite=False)
            except linalg.LinAlgError:
                status = "dependent_or_indefinite_working_set"
                break
            candidate = p0 - W @ lam
            # Correct subtraction error when the unrestricted step is much
            # larger than the remaining hardware slack.
            for _ in range(2):
                equality_error = np.asarray(CW @ candidate).ravel() - b[working]
                if np.all(np.abs(equality_error) <= 32 * np.finfo(float).eps * b[working]):
                    break
                correction = linalg.cho_solve(factor, equality_error, check_finite=False)
                lam += correction
                candidate -= W @ correction
        else:
            candidate, lam = p0.copy(), np.empty(0)
        candidate_cp = np.asarray(C @ candidate).ravel()
        # Componentwise tolerance is relative to each actual linearized bound.
        violation = candidate_cp - b
        # Sparse products and subtraction p0-W*lambda have an absolute
        # roundoff scale; using only b as that scale can re-add duplicate rows.
        roundoff_scale = np.asarray(abs(C) @ np.maximum(np.abs(p0), np.abs(candidate))).ravel() + b
        threshold = 128 * np.finfo(float).eps * roundoff_scale
        violating = violation > threshold
        if not np.any(violating):
            p, cp = candidate, candidate_cp
            negative = np.flatnonzero(lam < -multiplier_tolerance)
            if not len(negative):
                status, exact = "solved", True
                break
            remove = int(negative[np.argmin(lam[negative])])
            released_id = ids[working[remove]]
            released.append(released_id.item() if hasattr(released_id, "item") else released_id)
            del working[remove]
            lam = np.empty(0)  # Not associated with the updated working set.
            continue

        delta = candidate - p
        cd = candidate_cp - cp
        # Only endpoint-violated rows can block before alpha=1. In particular,
        # duplicate rows with roundoff-positive directional change must not
        # become artificial blockers at alpha approximately zero.
        eligible = (cd > 0) & violating
        if working:
            eligible[working] = False
        blockers = np.flatnonzero(eligible)
        if not len(blockers):
            status = "linearized_feasibility_roundoff"
            break
        ratios = (b[blockers] - cp[blockers]) / cd[blockers]
        index = int(np.argmin(ratios))
        blocker = int(blockers[index])
        alpha = float(np.clip(ratios[index], 0.0, 1.0))
        p = p + alpha * delta
        cp = np.asarray(C @ p).ravel()
        # A working-set limit returns the current feasible segment point.
        if len(working) >= max_active:
            status = "working_set_limit"
            break
        if blocker not in columns:
            columns[blocker] = np.asarray(free.inverse(C.getrow(blocker).toarray().ravel()))
            inverse_calls += 1
        working.append(blocker)
        lam = np.empty(0)

    # A common radial contraction removes roundoff violations while preserving
    # Ap=0 and every linearized inequality.
    raw_cp = np.asarray(raw_C @ p).ravel()
    target = fraction * slack
    overshoot = raw_cp > target
    if np.any(overshoot):
        contract = float(np.min(target[overshoot] / raw_cp[overshoot]))
        p *= max(0.0, contract * (1.0 - 32 * np.finfo(float).eps))
        raw_cp = np.asarray(raw_C @ p).ravel()
    product = np.asarray(free.model_product(p))
    model_change = float(gradient @ p + 0.5 * p @ product)
    # On indefiniteness or numerical trouble preserve a feasible ray of the
    # affine-chain direction when that ray gives strict model descent.
    if not np.isfinite(model_change) or (model_change >= 0 and np.linalg.norm(p0) > 0):
        outward = np.asarray(raw_C @ p0).ravel()
        positive = outward > 0
        alpha = (
            min(1.0, float(np.min(target[positive] / outward[positive])))
            if np.any(positive)
            else 1.0
        )
        alpha *= 1.0 - 32 * np.finfo(float).eps
        free_slope, free_quad = float(gradient @ p0), float(p0 @ free.model_product(p0))
        if free_slope < 0 and free_quad > 0:
            alpha = min(alpha, -free_slope / free_quad)
        p = alpha * p0
        product = np.asarray(free.model_product(p))
        model_change = float(gradient @ p + 0.5 * p @ product)
        raw_cp = np.asarray(raw_C @ p).ravel()
        working, lam, exact = [], np.empty(0), False
        status = "feasible_ray_fallback" if model_change < 0 else "no_model_descent"

    # For an intermediate segment, these are estimates from the most recent
    # equality minimizer, or zero after adding a new row. Recompute the KKT
    # residuals at the returned step, including steps stopped by a work limit.
    if len(lam) != len(working):
        lam = np.zeros(len(working))
    if working:
        CW = C[working]
        residual = geometry.project(gradient + product + np.asarray(CW.T @ lam).ravel())
    else:
        lam, residual = np.empty(0), geometry.project(gradient + product)
    residual_norm = float(np.linalg.norm(residual))
    primal = max(float(np.max((raw_cp - target) / slack, initial=0.0)), 0.0)
    dual = max(float(-np.min(lam, initial=0.0)) / gradient_norm, 0.0)
    comp = _norm_inf(lam * (np.asarray(C[working] @ p).ravel() - b[working])) if working else 0.0
    comp /= max(abs(float(gradient @ p)), abs(float(p @ product)), 1e-30)
    exact = bool(
        exact
        and primal <= 1e-8
        and dual <= 1e-8
        and residual_norm / tangent_norm <= 1e-6
        and comp <= 1e-6
    )
    if status == "solved" and not exact:
        status = "inexact_kkt"
    return SimpleNamespace(
        p=p,
        slope=float(gradient @ p),
        step_norm=float(np.linalg.norm(p)),
        endpoint_system=free.endpoint_system,
        relative_residual=residual_norm / gradient_norm,
        tangent_relative_residual=residual_norm / tangent_norm,
        constraint_residual=_norm_inf(geometry.A @ p),
        active_geometry=geometry,
        active_constraint_ids=ids[working].tolist(),
        active_released=released,
        active_multipliers=lam / row_norm[working],
        qp_exact=exact,
        qp_status=status,
        qp_updates=updates,
        qp_inverse_calls=inverse_calls,
        model_change=model_change,
        linearized_constraint_violation=primal,
        qp_stationarity_residual=residual_norm / tangent_norm,
        qp_dual_residual=dual,
        qp_complementarity_residual=comp,
    )
