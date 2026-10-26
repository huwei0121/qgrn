"""Oracle normalization and stable physical objective differences."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ._utils import _field


def _normalize(raw, n, derivatives):
    if not _field(raw, "valid", True):
        return SimpleNamespace(valid=False, reason=_field(raw, "reason", "invalid_oracle"))
    f = float(_field(raw, "f"))
    r = np.asarray(_field(raw, "r"), dtype=float)
    barrier = float(_field(raw, "barrier", 0.0))
    g = np.asarray(_field(raw, "g", np.zeros(n)), dtype=float)
    Jr = np.asarray(_field(raw, "Jr", np.empty((0, n))), dtype=float)
    if not derivatives:
        g, Jr = np.zeros(n), np.empty((0, n))
    if r.ndim != 1 or g.shape != (n,) or Jr.ndim != 2 or Jr.shape[1] != n:
        raise ValueError("Residual oracle dimensions do not match the optimization vector")
    if derivatives and Jr.shape[0] != r.size:
        raise ValueError("Endpoint Jacobian rows must match residual length")
    gb = np.asarray(_field(raw, "gb", np.zeros(n)), dtype=float)
    H0 = sparse.csc_matrix(_field(raw, "H0", sparse.csc_matrix((n, n))))
    Jreg = sparse.csc_matrix(_field(raw, "Jreg", sparse.csc_matrix((0, n))))
    if gb.shape != (n,) or H0.shape != (n, n) or Jreg.shape[1] != n:
        raise ValueError("Supplementary gradient/curvature dimensions mismatch")
    arrays = (r, g, Jr, gb, H0.data, Jreg.data)
    if not all(math.isfinite(v) for v in (f, barrier)) or not all(
        np.all(np.isfinite(v)) for v in arrays
    ):
        return SimpleNamespace(valid=False, reason="nonfinite_oracle")
    return SimpleNamespace(
        valid=True,
        reason="valid",
        derivatives=derivatives,
        f=f,
        r=r,
        g=g,
        Jr=Jr,
        barrier=barrier,
        gb=gb,
        H0=H0,
        Jreg=Jreg,
        gate_error=float(_field(raw, "gate_error", f)),
        ell=float(_field(raw, "ell", f)),
        h=_field(raw, "h"),
        raw=raw,
    )


def _remainder_packet(a, *, explicit_adapter):
    """Read and validate the complete physical residual for .5*||F||^2.

    An explicit adapter must provide its residual independently of any curvature
    factor stored as Jr. Its Jacobian may be a dense/sparse matrix, LinearOperator,
    or a callable applying DF to a step. The default endpoint contract is f=||r||^2.
    """
    F = _field(a.raw, "remainder_residual")
    J = _field(a.raw, "remainder_jacobian")
    if F is None:
        if explicit_adapter:
            raise ValueError(
                "Residual/split estimators require an explicit true remainder_residual contract from grn_oracle"
            )
        F = math.sqrt(2.0) * a.r
        J = math.sqrt(2.0) * a.Jr if a.derivatives else None
    F = np.asarray(F, dtype=float)
    if F.ndim != 1 or not np.all(np.isfinite(F)):
        raise ValueError("remainder_residual must be a finite real vector")
    residual_objective = 0.5 * float(F @ F)
    tolerance = 1e-9 * max(abs(a.f), abs(residual_objective), np.finfo(float).tiny)
    if abs(residual_objective - a.f) > tolerance:
        raise ValueError(
            "True remainder residual must represent the complete nonbarrier objective as .5*||F||^2"
        )
    if a.derivatives and J is None:
        raise ValueError(
            "Derivative oracle must provide remainder_jacobian with its remainder_residual"
        )
    if a.derivatives and not callable(J) and getattr(J, "shape", None) != (F.size, a.g.size):
        raise ValueError("remainder_jacobian dimensions mismatch")
    return F, J


def _objective_reduction(a, trial, *, explicit_adapter, residual_step=None):
    """Compute f(x)-f(x+p) without subtracting complete squared norms.

    Compute the difference using changed residual components so that a large
    constant residual does not hide a smaller improvement. The accompanying
    roundoff estimate diagnoses rejected searches; acceptance uses the computed
    reduction directly. The estimate covers this arithmetic, not errors in the
    underlying physical propagation.
    """
    F, trial_F = (_field(value.raw, "remainder_residual") for value in (a, trial))
    if F is None or trial_F is None:
        if explicit_adapter:
            change = math.fsum((a.f, -trial.f))
            noise = 8 * np.finfo(float).eps * max(abs(a.f), abs(trial.f))
            return change, noise, False, "scalar_difference"
        F, trial_F = math.sqrt(2.0) * a.r, math.sqrt(2.0) * trial.r
    F, trial_F = np.asarray(F, dtype=float), np.asarray(trial_F, dtype=float)
    if F.ndim != 1 or trial_F.shape != F.shape:
        raise ValueError("Complete objective residuals must have matching vector shapes")
    delta_F = trial_F - F
    change = math.fsum((-float(F @ delta_F), -0.5 * float(delta_F @ delta_F)))
    eps = np.finfo(float).eps
    component_noise = 8 * eps * (np.abs(F) + np.abs(trial_F))
    changed = delta_F != 0
    if residual_step is not None:
        expected = np.asarray(residual_step, dtype=float)
        if expected.shape != F.shape:
            raise ValueError("Residual model step must match the complete objective residual")
        changed = changed | (expected != 0)
    noise = float(np.abs(F[changed]) @ component_noise[changed])
    noise += 8 * eps * (abs(float(F @ delta_F)) + 0.5 * float(delta_F @ delta_F))
    unresolved = bool(np.all(np.abs(delta_F) <= component_noise))
    return change, noise, unresolved, "complete_residual_difference"


def _barrier_reduction(a, trial):
    """Compute logarithmic-barrier differences from normalized bound slacks.

    For box slack s=1-z**2, ds=-(z_new-z)*(z_new+z). Computing
    log1p(ds/s) retains small barrier changes. An optional
    barrier_difference_data packet supplies normalized box coordinates,
    time-bound slacks and a common barrier weight.
    """
    left, right = (_field(value.raw, "barrier_difference_data") for value in (a, trial))
    if left is None or right is None:
        value = math.fsum((a.barrier, -trial.barrier))
        noise = 8 * np.finfo(float).eps * (abs(a.barrier) + abs(trial.barrier))
        return value, noise, value == 0.0, "scalar_difference"
    weight = float(left["weight"])
    if weight != float(right["weight"]):
        raise ValueError("Barrier differences require the same fixed parameter")
    if weight == 0:
        return 0.0, 0.0, True, "log1p_slack_ratio"
    z, znew = np.asarray(left["box"]), np.asarray(right["box"])
    dz = znew - z
    ratio = -dz * (znew + z) / (1 - z * z)
    # Use the ratio only for nearby slacks. A large negative ratio can round
    # to -1 even though both endpoints are strictly interior.
    close = np.abs(ratio) < 0.5
    terms = np.empty_like(ratio)
    terms[close] = np.log1p(ratio[close])
    terms[~close] = np.log1p(-(znew[~close] ** 2)) - np.log1p(-(z[~close] ** 2))
    s, snew = np.asarray(left["time_slacks"]), np.asarray(right["time_slacks"])
    if s.size:
        relative = (snew - s) / s
        near = np.abs(relative) < 0.5
        time_terms = np.empty_like(relative)
        time_terms[near] = np.log1p(relative[near])
        time_terms[~near] = np.log(snew[~near]) - np.log(s[~near])
        terms = np.concatenate((terms, time_terms))
    value = weight * float(np.sum(terms))
    noise = 8 * np.finfo(float).eps * weight * float(np.sum(np.abs(terms)))
    # Estimate representability from componentwise slack changes.
    unresolved = bool(np.all(np.abs(dz) <= 8 * np.finfo(float).eps * (np.abs(z) + np.abs(znew))))
    if s.size:
        unresolved = unresolved and bool(
            np.all(np.abs(snew - s) <= 8 * np.finfo(float).eps * (np.abs(s) + np.abs(snew)))
        )
    return value, noise, unresolved, "log1p_slack_ratio"
