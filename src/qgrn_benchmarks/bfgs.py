"""Independent BFGS with fminunc-compatible scaling and stopping conventions.

The implementation uses inverse-BFGS updates, initial secant scaling, a curvature
safeguard, and safeguarded cubic interpolation for strong-Wolfe line search.
These mathematical mechanisms follow documented fminunc conventions; this
independent Python implementation does not include MathWorks proprietary code.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class FminuncOptions:
    tol_fun: float = 1e-8
    tol_x: float = 1e-8
    max_iterations: int = 400
    max_evaluations: int | None = None
    max_seconds: float = math.inf
    c1: float = 0.01
    c2: float = 0.9
    max_line_search: int = 40
    objective_limit: float = -1e20
    store_iterates: bool = True
    time_budget_only: bool = False


class _BudgetStop(Exception):
    def __init__(self, kind):
        self.kind = kind


def _inverse_bfgs_update(B, s, y, iteration):
    """In-place O(n^2) inverse update; return transparent secant diagnostics."""
    eps = np.finfo(float).eps
    curvature = float(s @ y)
    threshold = math.sqrt(eps) * max(eps, float(np.linalg.norm(s) * np.linalg.norm(y)))
    valid = np.isfinite(curvature) and curvature >= threshold
    info = dict(
        secant_curvature=curvature,
        curvature_threshold=threshold,
        update_skipped=not valid,
        inverse_scale=None,
    )
    if not valid:
        return info
    if iteration == 1:
        scale = curvature / float(y @ y)
        B.fill(0.0)
        np.fill_diagonal(B, scale)
        info["inverse_scale"] = scale
    By = B @ y
    B += np.outer(((1.0 + float(y @ By) / curvature) / curvature) * s, s)
    B -= np.outer(s / curvature, By)
    B -= np.outer(By / curvature, s)
    return info


def _cubic_trial(a, fa, da, b, fb, db):
    """Cubic-Hermite minimizer safeguarded to the middle 80% of a bracket."""
    lo, hi = sorted((a, b))
    candidate = 0.5 * (lo + hi)
    if all(np.isfinite(z) for z in (fa, da, fb, db)) and a != b:
        d1 = da + db - 3.0 * (fa - fb) / (a - b)
        disc = d1 * d1 - da * db
        if disc >= 0 and np.isfinite(disc):
            d2 = math.copysign(math.sqrt(disc), b - a)
            den = db - da + 2.0 * d2
            if den != 0:
                value = b - (b - a) * (db + d2 - d1) / den
                if np.isfinite(value):
                    candidate = value
    return float(np.clip(candidate, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)))


def _line_search(query, x, f0, g0, p, alpha1, options):
    """Bracket/zoom strong Wolfe; all attempted evaluations are budgeted."""
    d0 = float(g0 @ p)
    last = (0.0, f0, d0, g0)
    count = 0
    bracket = None
    alpha = alpha1
    while count < options.max_line_search:
        f, g = query(x + alpha * p, alpha)
        count += 1
        d = float(g @ p)
        valid = np.isfinite(f) and np.all(np.isfinite(g))
        current = (alpha, f, d, g)
        if not valid or f > f0 + options.c1 * alpha * d0 or (count > 1 and f >= last[1]):
            bracket = (last, current)
            break
        if abs(d) <= -options.c2 * d0:
            return current, "strong_wolfe", count
        if d >= 0:
            bracket = (current, last)
            break
        last = current
        alpha *= 2.0
        if not np.isfinite(alpha):
            break
    if bracket is not None:
        low, high = bracket
        while count < options.max_line_search:
            alpha = _cubic_trial(*low[:3], *high[:3])
            if alpha == low[0] or alpha == high[0]:
                break
            f, g = query(x + alpha * p, alpha)
            count += 1
            d = float(g @ p)
            valid = np.isfinite(f) and np.all(np.isfinite(g))
            current = (alpha, f, d, g)
            if not valid or f > f0 + options.c1 * alpha * d0 or f >= low[1]:
                high = current
            else:
                if abs(d) <= -options.c2 * d0:
                    return current, "strong_wolfe", count
                if d * (high[0] - low[0]) >= 0:
                    high = low
                low = current
    # Preserve an improving final trial on line-search failure. The final query
    # keeps oracle caches aligned with accepted-point callbacks.
    final = current if valid and current[1] < f0 else None
    return final, "line_search_failed", count


def minimize(fun, x0, *, options=None, callback=None, started=None):
    """Minimize a real scalar with joint ``fun(x)->(f,g)`` evaluations.

    ``callback(x)`` runs after each accepted point; True or StopIteration
    requests termination. Caller-supplied perf_counter ``started`` includes
    setup in the wall budget. History contains initial and accepted points;
    evaluation_history contains every trial scalar and its real timestamp.
    A time/evaluation limit inside line search returns the last accepted point.
    time_budget_only disables gradient, step and ObjectiveLimit termination;
    natural line-search/non-descent failures remain explicit unsuccessful exits.
    """
    started = time.perf_counter() if started is None else float(started)
    opt = FminuncOptions() if options is None else options
    if not isinstance(opt, FminuncOptions):
        raise TypeError("options must be FminuncOptions")
    if not isinstance(opt.time_budget_only, bool):
        raise ValueError("time_budget_only must be a boolean")
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be nonempty and finite")
    for name in ("tol_fun", "tol_x"):
        value = getattr(opt, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(name + " must be finite and nonnegative")
    if math.isnan(opt.max_seconds) or opt.max_seconds < 0:
        raise ValueError("max_seconds must be nonnegative")
    for name, minimum in (("max_iterations", 0), ("max_line_search", 1)):
        value = getattr(opt, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + " has an invalid integer value")
    maxeval = 100 * x.size if opt.max_evaluations is None else opt.max_evaluations
    if isinstance(maxeval, bool) or not isinstance(maxeval, (int, np.integer)) or maxeval < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if not 0 < opt.c1 < opt.c2 < 1:
        raise ValueError("Require 0<c1<c2<1")
    if not np.isfinite(opt.objective_limit):
        raise ValueError("objective_limit must be finite")
    history, evaluations = [], []
    nfev = 0
    k = 0
    B = np.eye(x.size)

    def query(point, alpha=None):
        nonlocal nfev
        if nfev >= maxeval:
            raise _BudgetStop("evaluation_limit")
        if nfev and time.perf_counter() - started >= opt.max_seconds:
            raise _BudgetStop("time_limit")
        nfev += 1
        value = fun(point)
        if isinstance(value, dict):
            f, g = value["f"], value["g"]
        elif hasattr(value, "f") and hasattr(value, "g"):
            f, g = value.f, value.g
        else:
            f, g = value
        f = float(f)
        g = np.asarray(g, dtype=float).reshape(-1).copy()
        if g.shape != x.shape:
            raise ValueError("Gradient shape differs from x0")
        event = dict(
            evaluation=nfev,
            k=k,
            alpha=alpha,
            objective=f,
            gradient_norm_inf=float(np.linalg.norm(g, np.inf)),
        )
        event["elapsed"] = time.perf_counter() - started
        evaluations.append(event)
        return f, g

    def record(f, g, **extras):
        event = dict(
            k=k,
            objective=float(f),
            gradient_norm_inf=float(np.linalg.norm(g, np.inf)),
            nfev=nfev,
            njev=nfev,
            **extras,
        )
        if opt.store_iterates:
            event["x"] = x.copy()
        event["elapsed"] = time.perf_counter() - started
        history.append(event)
        return event

    def finish(flag, message, status):
        return SimpleNamespace(
            x=x.copy(),
            fun=float(f),
            jac=g.copy(),
            nit=k,
            nfev=nfev,
            njev=nfev,
            status=status,
            exitflag=flag,
            success=flag > 0 and (not opt.time_budget_only or status == 0),
            message=message,
            history=history,
            evaluation_history=evaluations,
            hess_inv=B,
            seconds=time.perf_counter() - started,
            clock_started=started,
            options={**asdict(opt), "max_evaluations": int(maxeval)},
            backend="independent_fminunc_compatible_bfgs",
            exact_native_clone=False,
            line_search="independent cubic strong-Wolfe bracket/zoom",
        )

    f, g = query(x)
    record(f, g, kind="initial", step_size=0.0)
    if not np.isfinite(f) or not np.all(np.isfinite(g)):
        return finish(-4, "Nonfinite initial objective or gradient", 3)
    g0norm = float(np.linalg.norm(g, np.inf))
    if not opt.time_budget_only and g0norm < opt.tol_fun:
        return finish(1, "Initial gradient tolerance", 0)
    if not opt.time_budget_only and f < opt.objective_limit:
        return finish(-3, "Objective below ObjectiveLimit", 3)
    if opt.max_iterations == 0:
        return finish(0, "Iteration limit", 1)
    while k < opt.max_iterations:
        if time.perf_counter() - started >= opt.max_seconds:
            return finish(0, "Wall time limit", 4)
        if nfev >= maxeval:
            return finish(0, "Evaluation limit", 1)
        p = -B @ g
        slope = float(g @ p)
        if not np.isfinite(slope) or slope >= 0:
            return finish(5, "Non-descent search direction", 2)
        alpha1 = min(1.0 / g0norm, 1.0) if k == 0 else 1.0
        try:
            trial, search_status, line_evaluations = _line_search(query, x, f, g, p, alpha1, opt)
        except _BudgetStop as exc:
            return finish(
                0,
                "Wall time limit" if exc.kind == "time_limit" else "Evaluation limit",
                4 if exc.kind == "time_limit" else 1,
            )
        if trial is None:
            return finish(5, "Line search could not reduce the objective", 2)
        alpha, fnew, _, gnew = trial
        s, y = alpha * p, gnew - g
        x = x + s
        f, g = fnew, gnew
        k += 1
        relative_step = float(np.linalg.norm(s / (1.0 + np.abs(x)), np.inf))
        event = record(
            f,
            g,
            kind="accepted",
            step_size=alpha,
            initial_step=alpha1,
            directional_derivative=slope,
            relative_step=relative_step,
            line_search_status=search_status,
            line_search_evaluations=line_evaluations,
        )
        if callback is not None:
            try:
                stop = callback(x.copy())
            except StopIteration:
                stop = True
            if stop:
                return finish(-1, "Stopped by callback", 99)
        if not opt.time_budget_only and np.linalg.norm(g, np.inf) < opt.tol_fun * (1.0 + g0norm):
            return finish(1, "Relative gradient tolerance", 0)
        if not opt.time_budget_only and f < opt.objective_limit:
            return finish(-3, "Objective below ObjectiveLimit", 3)
        if not opt.time_budget_only and relative_step < opt.tol_x:
            return finish(2, "Relative componentwise step tolerance", 0)
        if search_status != "strong_wolfe":
            return finish(5, "Line search could not establish strong Wolfe conditions", 2)
        if nfev >= maxeval:
            return finish(0, "Evaluation limit", 1)
        if k >= opt.max_iterations:
            return finish(0, "Iteration limit", 1)
        event.update(_inverse_bfgs_update(B, s, y, k))
    return finish(0, "Iteration limit", 1)
