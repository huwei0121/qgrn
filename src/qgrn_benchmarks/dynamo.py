# SPDX-License-Identifier: LGPL-3.0-only
# Python translation of DYNAMO v1.2 by Shai Machnes (copyright 2010).
# Source archive, attribution and license: DYNAMO_TRANSLATION.txt.
# Python translation maintained by the Q-GRN project.
"""Function-level translation of the author MATLAB DYNAMO numerical core.

The MATLAB global ``OC`` becomes an explicit object. Function names, cache
flags, column-major mask indexing, termination precedence, and Krotov update
order are retained. Array indices become zero-based. The original PSU gradient
formula is retained as ``gradient_PSU_norm``; its mathematically consistent
alternative is selected explicitly through ``gradient_mode``.

``BFGS_search_function`` defaults to SciPy BFGS. An explicit independent
fminunc-compatible backend implements BFGS scaling and stopping mechanisms with
an independent strong-Wolfe search. Other search updates are translated directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize


@dataclass
class TerminationConditions:
    loop_count: int = 3000
    goal: float | None = 1 - 1e-4
    wall_time_to_stop: float = 120.0
    cputime_to_stop: float = 120.0
    gradient_norm_min: float | None = 1e-20


def dynamo_structure_init(num_spins):
    d = 2**num_spins
    oc = NS(
        config=NS(
            uInitial=np.full((d, d), np.nan),
            uFinal=np.full((d, d), np.nan),
            numControls=None,
            totalTime=np.nan,
            controlsInitialScaling=np.nan,
            normType="PSU",
            hamDrift=np.full((d, d), np.nan),
            hamControl=[],
            gradientFunc=gradient_exact,
            normFunc=PSU_norm,
            gradientNormFunc=gradient_PSU_norm,
        ),
        timeSlots=NS(nTimeSlots=None, expmFunc=expm, calcPfromHfunc=calcPfromH_exact_gradient),
        const=NS(
            termination_reason=NS(
                goal_achieved=1, loop_count=2, wall_time=3, cpu_time=4, gradient_norm=5
            ),
            termination_reason_str=[
                "Goal achieved",
                "Loop count limit reached",
                "Wall time limit reached",
                "CPU time limit reached",
                "Minimal gradient norm reached",
            ],
        ),
    )
    return oc


def configure(
    H0,
    Hc,
    target,
    total_time,
    controls,
    *,
    initial=None,
    gradient_mode="author_original",
):
    d = np.asarray(H0).shape[0]
    oc = dynamo_structure_init(int(round(np.log2(d))))
    oc.config.hamDrift = np.array(H0, dtype=complex, copy=True)
    oc.config.hamControl = [np.array(H, dtype=complex, copy=True) for H in Hc]
    oc.config.uInitial = (
        np.eye(d, dtype=complex) if initial is None else np.array(initial, copy=True)
    )
    oc.config.uFinal = np.array(target, dtype=complex, copy=True)
    oc.config.totalTime = float(total_time)
    oc.config.controlsInitialScaling = 1.0
    oc.timeSlots.nTimeSlots = len(controls)
    if gradient_mode == "mathematical_abs_trace":
        oc.config.gradientNormFunc = gradient_PSU_norm_mathematical
    elif gradient_mode != "author_original":
        raise ValueError("Unknown gradient mode")
    oc.gradient_mode = gradient_mode
    intialize_timeslot_controls(oc, controls)
    initSetNorm(oc)
    return oc


def intialize_timeslot_controls(oc, control_values):
    # Original misspelling retained for direct source correspondence.
    c, ts = oc.config, oc.timeSlots
    c.numControls = len(c.hamControl)
    M = ts.nTimeSlots
    if not hasattr(ts, "tau") or np.any(np.isnan(ts.tau)):
        ts.tau = np.full(M, c.totalTime / M)
    p = NS(
        controls=np.full(np.shape(control_values), np.nan),
        H=[None] * M,
        P=[None] * M,
        U=[None] * (M + 1),
        L=[None] * (M + 1),
        H_is_stale=np.ones(M, bool),
        P_is_stale=np.ones(M, bool),
        U_is_stale=np.r_[False, np.ones(M, bool)],
        L_is_stale=np.r_[np.ones(M, bool), False],
        H_needed_now=np.zeros(M, bool),
        P_needed_now=np.zeros(M, bool),
        U_needed_now=np.zeros(M + 1, bool),
        L_needed_now=np.zeros(M + 1, bool),
        curr_Phi0_val_stale=True,
        curr_Phi0_val=np.nan,
        H_eigVal=[None] * M,
        H_eigVec=[None] * M,
        H_factorMat=[None] * M,
    )
    p.U[0], p.L[-1] = c.uInitial.copy(), c.uFinal.conj().T.copy()
    ts.currPoint = p
    update_timeslot_controls(oc, control_values)


def to_col(value):
    return np.asarray(value).reshape(-1, order="F")


def get_current_controls(oc, subspace_mask=None):
    values = oc.timeSlots.currPoint.controls
    return values.copy() if subspace_mask is None else to_col(values)[to_col(subspace_mask)].copy()


def update_timeslot_controls(oc, new_control_values, subspace_mask=None):
    p = oc.timeSlots.currPoint
    if subspace_mask is None:
        mask = np.ones(p.controls.shape, bool)
        new = np.asarray(new_control_values, dtype=float).reshape(p.controls.shape, order="F")
    else:
        mask = np.asarray(subspace_mask, dtype=bool)
        flat = to_col(p.controls).copy()
        flat[to_col(mask)] = to_col(new_control_values)
        new = flat.reshape(p.controls.shape, order="F")
    old = p.controls.copy()
    old[~mask] = 0
    selected = old.copy()
    selected[mask] = new[mask]
    changed = np.any(selected != old, axis=1)
    # MATLAB's ~isempty(changed_t_mask) is true even for an all-false mask.
    if changed.size:
        p.H_is_stale[changed] = True
        p.P_is_stale[changed] = True
        p.controls[mask] = new[mask]
        stale = np.flatnonzero(p.H_is_stale)
        if stale.size:
            p.U_is_stale[stale[0] + 1 :] = True
            p.L_is_stale[: stale[-1] + 1] = True
        p.curr_Phi0_val_stale = True
        p.curr_Phi0_val = np.nan


def invalidate_timeslots_calc_cache(oc):
    p = oc.timeSlots.currPoint
    p.H_is_stale[:] = True
    p.P_is_stale[:] = True
    p.U_is_stale[1:] = True
    p.L_is_stale[:-1] = True
    p.curr_Phi0_val_stale = True


def fill_timeslots_cache(oc):
    invalidate_timeslots_calc_cache(oc)
    p = oc.timeSlots.currPoint
    for name in ("H_needed_now", "P_needed_now", "U_needed_now", "L_needed_now"):
        getattr(p, name)[:] = True
    recompute_timeslots_now(oc)
    get_current_value(oc)


def recompute_timeslots_now(oc):
    ts, c = oc.timeSlots, oc.config
    p, M = ts.currPoint, ts.nTimeSlots
    ur = p.U_needed_now & p.U_is_stale
    lr = p.L_needed_now & p.L_is_stale
    for t in range(M, 0, -1):
        if ur[t] and p.U_is_stale[t - 1]:
            ur[t - 1] = True
    for t in range(M - 1):
        if lr[t] and p.L_is_stale[t + 1]:
            lr[t + 1] = True
    pr = (ur[1:] | lr[:-1] | p.P_needed_now) & p.P_is_stale
    hr = (pr | p.H_needed_now) & p.H_is_stale
    for t in np.flatnonzero(hr):
        H = c.hamDrift.copy()
        for j, Hc in enumerate(c.hamControl):
            H = H + p.controls[t, j] * Hc
        p.H[t] = H
    for t in np.flatnonzero(pr):
        ts.calcPfromHfunc(oc, t)
    for t in np.flatnonzero(ur):
        p.U[t] = p.P[t - 1] @ p.U[t - 1]
    for t in np.flatnonzero(lr)[::-1]:
        p.L[t] = p.L[t + 1] @ p.P[t]
    for stale, mask in (
        (p.H_is_stale, hr),
        (p.P_is_stale, pr),
        (p.U_is_stale, ur),
        (p.L_is_stale, lr),
    ):
        stale[mask] = False
    for name in ("H_needed_now", "P_needed_now", "U_needed_now", "L_needed_now"):
        getattr(p, name)[:] = False


def calcPfromH_expm(oc, t):
    ts = oc.timeSlots
    ts.currPoint.P[t] = ts.expmFunc(-1j * ts.tau[t] * ts.currPoint.H[t])


def calcPfromH_exact_gradient(oc, t):
    ts, p = oc.timeSlots, oc.timeSlots.currPoint
    # Hermitian eigensystem of H is the same spectral calculation as eig(-i*tau*H).
    # eigh preserves an orthonormal basis at degeneracies; LAPACK order differs.
    real_eigenvalues, V = eigh(p.H[t])
    z = -1j * ts.tau[t] * real_eigenvalues
    ez = np.exp(z)
    difference = z[:, None] - z[None, :]
    degenerate = np.abs(difference) < 1e-10
    safe = difference.copy()
    safe[degenerate] = 1
    factor = (ez[:, None] - ez[None, :]) / safe
    factor[degenerate] = np.broadcast_to(ez[:, None], factor.shape)[degenerate]
    p.H_eigVal[t], p.H_eigVec[t], p.H_factorMat[t] = z, V, factor
    p.P[t] = (V * ez) @ V.conj().T


def get_current_value_setup_recalc(oc):
    p = oc.timeSlots.currPoint
    flags = (~p.U_is_stale).astype(int) + 2 * (~p.L_is_stale).astype(int)
    both = np.flatnonzero(flags == 3)
    if both.size:
        t = both[0]
    else:
        positions = np.flatnonzero(flags > 0)
        pairs = np.flatnonzero((flags[positions[:-1]] == 1) & (flags[positions[1:]] == 2))
        if not pairs.size:
            raise RuntimeError("There should be at least one valid propagation pair")
        best = pairs[np.argmin(positions[pairs + 1] - positions[pairs])]
        t = positions[best]
    p.U_needed_now[t] = True
    p.L_needed_now[t] = True
    return t


def trace_matmul(A, B):
    return np.sum(np.asarray(A).T * np.asarray(B))


def Phi0_norm(A, B=None):
    return np.trace(A) if B is None else trace_matmul(A, B)


def PSU_norm(oc, A, B=None):
    v = A if np.ndim(A) == 0 and B is None else Phi0_norm(A, B)
    return float(abs(v) / oc.config.normNorm)


def SU_norm(oc, A, B=None):
    v = A if np.ndim(A) == 0 and B is None else Phi0_norm(A, B)
    return float(np.real(v) / oc.config.normNorm)


def initSetNorm(oc):
    oc.config.normNorm = 1.0
    oc.config.normNorm = oc.config.normFunc(oc, oc.config.uFinal.conj().T, oc.config.uFinal)


def get_current_value_Phi0_norm(oc):
    p = oc.timeSlots.currPoint
    if not p.curr_Phi0_val_stale:
        return p.curr_Phi0_val
    t = get_current_value_setup_recalc(oc)
    recompute_timeslots_now(oc)
    v = Phi0_norm(p.U[t], p.L[t])
    p.curr_Phi0_val_stale = False
    p.curr_Phi0_val = v
    return v


def get_current_value(oc):
    counts = getattr(oc, "_optimization_query_counts", None)
    if counts is not None:
        counts["value_requests"] += 1
        key = (
            "value_cache_misses"
            if oc.timeSlots.currPoint.curr_Phi0_val_stale
            else "value_cache_hits"
        )
        counts[key] += 1
    value = oc.config.normFunc(oc, get_current_value_Phi0_norm(oc))
    if counts is not None:
        counts["value_requests_completed"] += 1
    return value


def filter_by_subspace_mask(value, subset_mask):
    mask = np.asarray(subset_mask, dtype=bool)
    result = np.array(value, dtype=complex if np.iscomplexobj(value) else float, copy=True)
    result[~mask] = np.nan
    return result[np.any(mask, axis=1), :]


def gradient_exact(oc, subspace_mask):
    p, ts, c = oc.timeSlots.currPoint, oc.timeSlots, oc.config
    mask = np.asarray(subspace_mask, bool)
    slots = np.any(mask, axis=1)
    p.H_needed_now[slots] = True
    p.P_needed_now[slots] = True
    p.U_needed_now[:-1] |= slots
    p.L_needed_now[1:] |= slots
    value = get_current_value_Phi0_norm(oc)
    recompute_timeslots_now(oc)
    gradient = np.zeros(mask.shape, complex)
    for flat_index in np.flatnonzero(to_col(mask)):
        t, j = np.unravel_index(flat_index, mask.shape, order="F")
        V = p.H_eigVec[t]
        direction = V.conj().T @ (-1j * ts.tau[t] * c.hamControl[j]) @ V
        derivative = V @ (direction * p.H_factorMat[t]) @ V.conj().T
        gradient[t, j] = Phi0_norm(p.L[t + 1] @ derivative @ p.U[t])
    return value, filter_by_subspace_mask(gradient, mask)


def gradient_PSU_norm(oc, subspace_mask):
    value, gradient = oc.config.gradientFunc(oc, subspace_mask)
    return 2 * np.real(gradient * np.conj(value)) / oc.config.normNorm


def gradient_PSU_norm_mathematical(oc, subspace_mask):
    value, gradient = oc.config.gradientFunc(oc, subspace_mask)
    if abs(value) == 0:
        raise ValueError("Absolute trace objective is nondifferentiable at zero overlap")
    return np.real(gradient * np.conj(value)) / (oc.config.normNorm * abs(value))


def gradient_SU_norm(oc, subspace_mask):
    _, gradient = oc.config.gradientFunc(oc, subspace_mask)
    return np.real(gradient) / oc.config.normNorm


def gradient_first_order_aprox(oc, subspace_mask):
    p, ts, c = oc.timeSlots.currPoint, oc.timeSlots, oc.config
    mask = np.asarray(subspace_mask, bool)
    slots = np.any(mask, axis=1)
    p.H_needed_now[:] = True
    p.U_needed_now[1:] |= slots
    p.L_needed_now[1:] |= slots
    value = get_current_value_Phi0_norm(oc)
    recompute_timeslots_now(oc)
    gradient = np.full(mask.shape, np.nan + 0j)
    for t, j in zip(*np.nonzero(mask)):
        gradient[t, j] = Phi0_norm(p.L[t + 1] @ (-1j * ts.tau[t] * c.hamControl[j]) @ p.U[t + 1])
    return value, filter_by_subspace_mask(gradient, mask)


def gradient_finite_diff(oc, subspace_mask, epsilon):
    p, ts, c = oc.timeSlots.currPoint, oc.timeSlots, oc.config
    mask = np.asarray(subspace_mask, bool)
    slots = np.any(mask, axis=1)
    p.H_needed_now[:] = True
    p.U_needed_now[:-1] |= slots
    p.L_needed_now[1:] |= slots
    value = get_current_value_Phi0_norm(oc)
    recompute_timeslots_now(oc)
    gradient = np.full(mask.shape, np.nan + 0j)
    for t, j in zip(*np.nonzero(mask)):
        peps = ts.expmFunc(-1j * ts.tau[t] * (p.H[t] + epsilon * c.hamControl[j]))
        gradient[t, j] = (Phi0_norm(peps @ p.U[t], p.L[t + 1]) - value) / epsilon
    return value, filter_by_subspace_mask(gradient, mask)


def _termination(term, count, wall, cpu, grad, value):
    reason = None
    # Original independent if statements: later conditions take precedence.
    if term.loop_count <= count:
        reason = 2
    if term.wall_time_to_stop <= wall:
        reason = 3
    if term.cputime_to_stop <= cpu:
        reason = 4
    if term.gradient_norm_min is not None and term.gradient_norm_min >= grad:
        reason = 5
    if term.goal is not None and term.goal <= value:
        reason = 1
    return reason


def _observe(oc, kind, iteration):
    if hasattr(oc, "observer"):
        oc.observer(oc, kind, iteration)


def _validate_search_mask(oc, subspace_mask):
    mask = np.asarray(subspace_mask, dtype=bool)
    if mask.shape != oc.timeSlots.currPoint.controls.shape or not np.any(mask):
        raise ValueError("Search requires a nonempty mask matching the control matrix")
    if np.any(np.any(mask, axis=1) != np.all(mask, axis=1)):
        raise ValueError(
            "Partial-control masks are unsupported by the author gradient layout; select complete time-slot rows"
        )
    return mask


def First_order_search_function(oc, subspace_mask, termination_conditions):
    subspace_mask = _validate_search_mask(oc, subspace_mask)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    if not hasattr(oc.config, "FirstOrder"):
        oc.config.FirstOrder = NS(step_size=0.1)
    if not hasattr(oc.config.FirstOrder, "step_size"):
        oc.config.FirstOrder.step_size = 0.1
    counter = 0
    while True:
        curr_x = get_current_controls(oc, subspace_mask)
        curr_val = get_current_value(oc)
        curr_grad = oc.config.gradientNormFunc(oc, subspace_mask)
        norm_grad = np.sqrt(np.sum(curr_grad * curr_grad))
        step = oc.config.FirstOrder.step_size
        next_x = curr_x + step * to_col(curr_grad)
        update_timeslot_controls(oc, next_x, subspace_mask)
        next_val = get_current_value(oc)
        actual = next_val - curr_val
        expected = norm_grad**2 * step / oc.config.normNorm
        if actual < (4 / 12) * expected:
            oc.config.FirstOrder.step_size *= 0.99
        elif actual > (8 / 12) * expected:
            oc.config.FirstOrder.step_size *= 1.01
        counter += 1
        _observe(oc, "first_order_update", counter)
        reason = _termination(
            termination_conditions,
            counter,
            time.perf_counter() - wall0,
            time.process_time() - cpu0,
            norm_grad,
            next_val,
        )
        if reason is not None:
            return reason


def Block_cycle_search_function(
    oc,
    subspace_mask,
    meta_term_cond,
    per_block_term_cond,
    block_search_function,
    block_size,
):
    subspace_mask = _validate_search_mask(oc, subspace_mask)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    mask = np.asarray(subspace_mask, bool)
    rows = np.flatnonzero(np.any(mask, axis=1))
    if len(rows) == 0 or block_size < 1:
        raise ValueError("Nonempty mask and positive block size required")
    nblocks = int(np.ceil(len(rows) / block_size))
    blocks = []
    if nblocks < 2:
        blocks = [mask]
    else:
        for k in range(nblocks - 1):
            b = np.zeros(mask.shape, bool)
            selected = rows[k * block_size : (k + 1) * block_size]
            b[selected] = mask[selected]
            blocks.append(b)
        b = np.zeros(mask.shape, bool)
        selected = rows[-block_size:]
        b[selected] = mask[selected]
        blocks.append(b)
    counter = 0
    while True:
        reason = block_search_function(oc, blocks[counter % nblocks], per_block_term_cond)
        counter += 1
        value = get_current_value(oc)
        if reason == 1:
            return 1
        reason = _termination(
            meta_term_cond,
            counter,
            time.perf_counter() - wall0,
            time.process_time() - cpu0,
            np.inf,
            value,
        )
        if reason is not None:
            return reason


def Krotov_search_function(oc, subspace_mask, term_cond):
    per_block = TerminationConditions(**vars(term_cond))
    per_block.loop_count = 1
    return Block_cycle_search_function(
        oc, subspace_mask, term_cond, per_block, First_order_search_function, 1
    )


def BFGS_search_function(
    oc,
    subspace_mask,
    termination_conditions,
    *,
    maxiter=3000,
    backend="scipy",
    precision_stopping=True,
):
    """Original wrapper with an explicitly selected independent inner solver."""
    if backend not in ("scipy", "fminunc_compatible"):
        raise ValueError("Unknown BFGS backend")
    subspace_mask = _validate_search_mask(oc, subspace_mask)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    counter = 0
    last_grad_norm = np.nan
    termination_reason = 5

    def goal_and_gradient_function_wrapper(x):
        nonlocal counter, last_grad_norm
        counter += 1
        update_timeslot_controls(oc, x, subspace_mask)
        v = -get_current_value(oc)
        gradient = -oc.config.gradientNormFunc(oc, subspace_mask)
        last_grad_norm = float(np.sum(gradient * gradient))
        return v, to_col(gradient)

    def monitor_function(x):
        nonlocal termination_reason
        # The upstream monitor reads the last oracle cache; do not alter it.
        _observe(oc, "bfgs_accepted", counter)
        reason = _termination(
            termination_conditions,
            counter,
            time.perf_counter() - wall0,
            time.process_time() - cpu0,
            last_grad_norm,
            get_current_value(oc),
        )
        if reason is not None:
            termination_reason = reason
            raise StopIteration

    author_options = {
        "TolX": 1e-8,
        "TolFun": 1e-8,
        "MaxIter": maxiter,
        "Display": "off",
    }
    override = getattr(getattr(oc.config, "BFGS", NS()), "fminopt", {})
    override = override if isinstance(override, dict) else vars(override)
    unknown = set(override) - set(author_options)
    if unknown:
        raise ValueError(f"No declared SciPy substitution for fminunc options: {sorted(unknown)}")
    author_options.update(override)
    for name in ("TolX", "TolFun"):
        value = author_options[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
            or not np.isreal(value)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    limit = author_options["MaxIter"]
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("MaxIter must be a nonnegative integer")
    if author_options["Display"] not in ("off", "final"):
        raise ValueError("Only Display='off' or 'final' has a declared SciPy substitute")
    if backend == "scipy":
        effective_gtol = float(author_options["TolFun"]) if precision_stopping else 0.0
        effective_xrtol = float(author_options["TolX"]) if precision_stopping else 0.0
        result = minimize(
            goal_and_gradient_function_wrapper,
            get_current_controls(oc, subspace_mask),
            method="BFGS",
            jac=True,
            callback=monitor_function,
            options={
                "maxiter": int(author_options["MaxIter"]),
                "gtol": effective_gtol,
                "xrtol": effective_xrtol,
                "disp": author_options["Display"] != "off",
            },
        )
        backend_label = "SciPy BFGS with SciPy line search and stopping rules"
    else:
        from .bfgs import FminuncOptions
        from .bfgs import minimize as compatible_minimize

        options = FminuncOptions(
            tol_fun=float(author_options["TolFun"]),
            tol_x=float(author_options["TolX"]),
            max_iterations=int(author_options["MaxIter"]),
            max_seconds=float(termination_conditions.wall_time_to_stop),
            max_evaluations=None if precision_stopping else 10**9,
            time_budget_only=not precision_stopping,
        )
        effective_gtol = author_options["TolFun"] if precision_stopping else None
        effective_xrtol = author_options["TolX"] if precision_stopping else None
        result = compatible_minimize(
            goal_and_gradient_function_wrapper,
            get_current_controls(oc, subspace_mask),
            options=options,
            callback=monitor_function,
            started=wall0,
        )
        backend_label = "Independent fminunc-compatible BFGS with cubic strong-Wolfe search"
        if result.status == 4:
            termination_reason = oc.const.termination_reason.wall_time
    update_timeslot_controls(oc, result.x, subspace_mask)
    oc.last_bfgs_result = NS(
        result=result,
        oracle_calls=counter,
        backend=backend_label,
        backend_key=backend,
        author_options=author_options,
        python_gtol=effective_gtol,
        python_xrtol=effective_xrtol,
    )
    return termination_reason


def Two_method_crossover_function(
    oc,
    subspace_mask,
    first_method_term_cond,
    first_search_method,
    second_method_term_cond,
    second_search_method,
):
    first_search_method(oc, subspace_mask, first_method_term_cond)
    return second_search_method(oc, subspace_mask, second_method_term_cond)


def spaghetti_stats_collector_init(oc):
    if not hasattr(oc, "stats"):
        oc.stats = NS(goal_value=[], goal_cpu_time=[], goal_wall_time=[])


def spaghetti_stats_collector_goal_func_wrapper(oc, goal_func, x):
    v = goal_func(x)
    oc.stats.goal_value.append(v)
    oc.stats.goal_cpu_time.append(time.process_time())
    oc.stats.goal_wall_time.append(time.time())
    return v


def computeHamiltonians(O):  # noqa: E741 - upstream mathematical API
    O.hamTotal = []
    for t in range(O.numTimeSlices):
        H = np.array(O.hamDrift, copy=True)
        for j in range(O.numControls):
            H = H + O.controls[t, j] * O.hamControl[j]
        O.hamTotal.append(H)
    return O


def qft(n):
    N = 2**n
    w = np.exp(2j * np.pi / N)
    row = np.arange(N)
    return w ** (row[:, None] * row[None, :]) / np.sqrt(N)


def randseed(seedvalue=None):
    """Return a NumPy MT19937 RNG; import explicit controls for MATLAB parity.

    MATLAB/NumPy seeding and normal transforms are not asserted identical.
    No global NumPy random state is changed.
    """
    if seedvalue is None:
        seedvalue = time.time_ns() % (2**32)
    return np.random.Generator(np.random.MT19937(int(seedvalue)))


def optimize(
    oc,
    *,
    method="krotov",
    maxiter=2,
    seconds=120.0,
    goal=1 - 1e-4,
    history_mode="full",
    bfgs_backend="scipy",
    precision_stopping=True,
):
    """Finite observable interface. Krotov iterations count time-slice updates.

    Query counts include observations and the final returned scalar. A value
    request may reuse the cached complex overlap; it is not a propagator count.
    Propagator counts are calls to calcPfromHfunc for stale requested cache slots.
    All temporary instrumentation and the previous observer are restored on exit.
        Delta history retains the initial physical coordinate vector and only the
        changed coordinates thereafter; every observed objective/time/count remains.
        precision_stopping=False disables gradient/step precision exits; pass
        goal=None separately to disable the application fidelity target as well.
    """
    if method not in ("krotov", "bfgs"):
        raise ValueError("method must be krotov or bfgs")
    if bfgs_backend not in ("scipy", "fminunc_compatible"):
        raise ValueError("Unknown BFGS backend")
    if not isinstance(precision_stopping, bool):
        raise ValueError("precision_stopping must be a boolean")
    if history_mode not in ("full", "delta"):
        raise ValueError("history_mode must be full or delta")
    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 0
    ):
        raise ValueError("maxiter must be a nonnegative integer")
    if (
        isinstance(seconds, (bool, np.bool_))
        or not np.isscalar(seconds)
        or not np.isreal(seconds)
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("seconds must be finite and nonnegative")
    if goal is not None and (not np.isscalar(goal) or not np.isreal(goal) or not np.isfinite(goal)):
        raise ValueError(
            "goal must be finite and real, or None to disable objective-target stopping"
        )
    start = time.perf_counter()
    history = []
    previous_coordinates = None
    counts = dict(
        value_requests=0,
        value_requests_completed=0,
        value_cache_hits=0,
        value_cache_misses=0,
        gradient_requests=0,
        gradient_requests_completed=0,
        propagator_recomputations=0,
        propagator_recomputations_completed=0,
    )
    had_counts = hasattr(oc, "_optimization_query_counts")
    previous_counts = getattr(oc, "_optimization_query_counts", None)
    previous_gradient = oc.config.gradientNormFunc
    previous_propagator = oc.timeSlots.calcPfromHfunc

    def counted_gradient(state, *args, **kwargs):
        counts["gradient_requests"] += 1
        value = previous_gradient(state, *args, **kwargs)
        counts["gradient_requests_completed"] += 1
        return value

    def counted_propagator(state, *args, **kwargs):
        counts["propagator_recomputations"] += 1
        value = previous_propagator(state, *args, **kwargs)
        counts["propagator_recomputations_completed"] += 1
        return value

    def observer(state, kind, iteration):
        nonlocal previous_coordinates
        objective = 1 - get_current_value(state)
        if history_mode == "full":
            coordinates = dict(controls=get_current_controls(state))
        else:
            # The author controls are (time, channel); C order is the physical
            # (channel, time) F-order vector used by the common paper runner.
            current = state.timeSlots.currPoint.controls.reshape(-1, order="C")
            if previous_coordinates is None:
                coordinates = dict(x=current.copy())
                previous_coordinates = current.copy()
            else:
                changed = np.flatnonzero(current != previous_coordinates)
                values = current[changed]
                coordinates = dict(coordinate_delta_indices=changed, coordinate_delta_values=values)
                previous_coordinates[changed] = values
        observed_counts = counts.copy()
        history.append(
            dict(
                iteration=len(history),
                internal_iteration=iteration,
                kind=kind,
                elapsed=time.perf_counter() - start,
                objective=objective,
                **coordinates,
                step_size=getattr(getattr(state.config, "FirstOrder", NS()), "step_size", None),
                evaluations=observed_counts,
            )
        )

    had_observer = hasattr(oc, "observer")
    previous = getattr(oc, "observer", None)
    term = TerminationConditions(
        loop_count=maxiter if method == "krotov" else 10**10,
        goal=goal,
        wall_time_to_stop=seconds,
        cputime_to_stop=seconds,
        gradient_norm_min=1e-20 if precision_stopping else None,
    )
    mask = np.ones(oc.timeSlots.currPoint.controls.shape, bool)
    try:
        oc._optimization_query_counts = counts
        oc.config.gradientNormFunc = counted_gradient
        oc.timeSlots.calcPfromHfunc = counted_propagator
        oc.observer = observer
        observer(oc, "initial", 0)
        remaining = seconds - (time.perf_counter() - start)
        term.wall_time_to_stop = max(0.0, remaining)
        term.cputime_to_stop = max(0.0, remaining)
        if remaining <= 0:
            reason = 3
        elif maxiter == 0:
            reason = 2
        elif method == "krotov":
            reason = Krotov_search_function(oc, mask, term)
        elif method == "bfgs":
            reason = BFGS_search_function(
                oc,
                mask,
                term,
                maxiter=maxiter,
                backend=bfgs_backend,
                precision_stopping=precision_stopping,
            )
        else:
            raise ValueError("method must be krotov or bfgs")
        observer(oc, "returned", len(history))
        returned_controls = get_current_controls(oc)
        returned_objective = 1 - get_current_value(oc)
    finally:
        if had_observer:
            oc.observer = previous
        elif hasattr(oc, "observer"):
            del oc.observer
        oc.config.gradientNormFunc = previous_gradient
        oc.timeSlots.calcPfromHfunc = previous_propagator
        if had_counts:
            oc._optimization_query_counts = previous_counts
        elif hasattr(oc, "_optimization_query_counts"):
            del oc._optimization_query_counts
    backend_result = None
    if method == "bfgs" and hasattr(oc, "last_bfgs_result") and remaining > 0 and maxiter > 0:
        record = oc.last_bfgs_result
        r = record.result
        backend_result = dict(
            status=int(r.status),
            success=bool(r.success),
            message=str(r.message),
            iterations=int(r.nit),
            function_evaluations=int(r.nfev),
            gradient_evaluations=int(r.njev),
            fun=float(r.fun),
            options=record.author_options,
            python_gtol=record.python_gtol,
            python_xrtol=record.python_xrtol,
        )
        if bfgs_backend == "fminunc_compatible":
            backend_result.update(
                exitflag=int(r.exitflag),
                options=r.options,
                line_search=r.line_search,
                exact_native_clone=False,
                history=r.history,
                evaluation_history=r.evaluation_history,
                seconds=r.seconds,
            )
    author_status = oc.const.termination_reason_str[reason - 1]
    status = author_status
    if backend_result is not None:
        if bfgs_backend == "scipy":
            status = {
                0: "SCIPY_CONVERGED",
                1: "SCIPY_ITERATION_LIMIT",
                2: "SCIPY_PRECISION_LOSS",
                3: "SCIPY_NONFINITE",
                99: "SCIPY_CALLBACK_STOP",
            }.get(backend_result["status"], "SCIPY_OTHER_STOP")
        else:
            status = {
                0: "FMINUNC_COMPATIBLE_CONVERGED",
                1: "FMINUNC_COMPATIBLE_BUDGET_LIMIT",
                2: "FMINUNC_COMPATIBLE_LINE_SEARCH_STOP",
                3: "FMINUNC_COMPATIBLE_INVALID_OR_LIMIT",
                4: "FMINUNC_COMPATIBLE_TIME_LIMIT",
                99: "FMINUNC_COMPATIBLE_CALLBACK_STOP",
            }.get(backend_result["status"], "FMINUNC_COMPATIBLE_OTHER_STOP")
    return dict(
        status=status,
        author_status=author_status,
        author_termination_code=reason,
        status_semantics="Selected BFGS backend status; original wrapper reason preserved separately as author_status",
        backend_result=backend_result,
        history=history,
        controls=returned_controls,
        objective=returned_objective,
        history_mode=history_mode,
        history_coordinate_semantics="Initial x and zero-based deltas in C-order (time,channel), equivalent to F-order (channel,time); full mode stores author-shaped controls",
        evaluations=counts.copy(),
        evaluation_semantics={
            "scope": "This optimize call only, including initial/accepted/returned observations and final scalar return; configure and caller rollout excluded",
            "value_requests": "Calls to get_current_value, including cached observations; completed counts require normal return",
            "value_cache_hits": "Complex Phi0 overlap already fresh at value-request entry; normalized scalar callback still runs",
            "value_cache_misses": "Complex Phi0 overlap stale at value-request entry; may reuse all propagators and are not matrix-exponential counts",
            "gradient_requests": "Calls to configured gradientNormFunc, including successful cached-gradient work; no gradient-result memoization is added",
            "propagator_recomputations": "Calls to configured calcPfromHfunc for stale requested time slots, not all value/gradient requests or direct finite-difference expm calls",
            "failed_requests": "Request/recomputation totals increment before callbacks; completed fields increment only after normal return",
            "scipy_counters": "backend_result function_evaluations/gradient_evaluations retain selected backend nfev/njev and exclude observer/final-only value requests",
            "timing": "Instrumentation, snapshots and callback restoration are included in seconds",
        },
        seconds=time.perf_counter() - start,
        method=method,
        gradient_mode=oc.gradient_mode,
        provenance="translated_author_dynamo_v1_2",
        backend="direct Krotov translation"
        if method == "krotov"
        else (
            "SciPy BFGS replacing fminunc"
            if bfgs_backend == "scipy"
            else "Independent fminunc-compatible BFGS"
        ),
        bfgs_backend=bfgs_backend if method == "bfgs" else None,
        objective_target=goal,
        precision_stopping=precision_stopping,
        budget_unit="time_slice_updates" if method == "krotov" else "BFGS iterations",
    )
