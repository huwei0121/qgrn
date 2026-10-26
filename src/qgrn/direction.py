"""Regularized Gauss--Newton directions via sparse KKT and endpoint Gram systems."""

import math
from types import SimpleNamespace

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from ._utils import _norm_inf


def solve_direction(a, mu, geometry, *, expose_inverse=False):
    """Solve the regularized Gauss-Newton equations on the affine tangent space.

    The system is P[(2 Jr.T Jr + H0 + mu I)p + g + gb] = 0, A p = 0,
    where P is the orthogonal projector onto ker(A). For a pure residual model
    with zero H0 and gb and no affine constraints, an endpoint Gram solve avoids
    subtractive cancellation in the Woodbury formula. Residual checks and up to
    two iterative refinements use these same equations.
    """
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError("Regularization mu must be finite and strictly positive")
    n, A = geometry.n, geometry.A
    gradient = a.g + a.gb
    V = math.sqrt(2.0) * a.Jr.T
    B = a.H0 + sparse.eye(n, format="csc") * mu
    endpoint_only = A.shape[0] == 0 and a.H0.nnz == 0 and np.count_nonzero(a.gb) == 0
    if endpoint_only:
        residual_gradient = 2.0 * a.Jr.T @ a.r
        scale = max(_norm_inf(gradient), _norm_inf(residual_gradient))
        endpoint_only = _norm_inf(gradient - residual_gradient) <= 32 * np.finfo(float).eps * scale
    if endpoint_only:
        gram = 2.0 * (a.Jr @ a.Jr.T)
        cf = linalg.cho_factor(gram + mu * np.eye(gram.shape[0]), lower=True, check_finite=False)
        p = -2.0 * a.Jr.T @ linalg.cho_solve(cf, a.r, check_finite=False)

        def inverse(rhs):
            # Apply the regularized inverse for iterative refinement.
            return (rhs - V @ linalg.cho_solve(cf, V.T @ rhs, check_finite=False)) / mu
    else:
        K = sparse.bmat([[B, A.T], [A, sparse.csc_matrix((A.shape[0], A.shape[0]))]], format="csc")
        factor = splu(K)
        rhs = np.vstack((np.column_stack((gradient, V)), np.zeros((A.shape[0], 1 + V.shape[1]))))
        solves = factor.solve(rhs)
        pg, PV = solves[:n, 0], solves[:n, 1:]
        small = np.eye(V.shape[1]) + V.T @ PV
        cf = linalg.cho_factor((small + small.T) / 2, lower=True, check_finite=False)
        p = -pg + PV @ linalg.cho_solve(cf, V.T @ pg, check_finite=False)

        def inverse(rhs):
            rhs = np.asarray(rhs, dtype=float)
            full_rhs = (
                np.concatenate((rhs, np.zeros(A.shape[0])))
                if rhs.ndim == 1
                else np.vstack((rhs, np.zeros((A.shape[0], rhs.shape[1]))))
            )
            q = factor.solve(full_rhs)[:n]
            return q - PV @ linalg.cho_solve(cf, V.T @ q, check_finite=False)

    gradient_norm = max(float(np.linalg.norm(gradient)), 1e-30)
    tangent_norm = max(float(np.linalg.norm(geometry.project(gradient))), 1e-30)
    residual = geometry.project(B @ p + V @ (V.T @ p) + gradient)
    tangent_rel = float(np.linalg.norm(residual)) / tangent_norm
    for _ in range(2):
        if tangent_rel < 1e-7:
            break
        p -= inverse(residual)
        residual = geometry.project(B @ p + V @ (V.T @ p) + gradient)
        tangent_rel = float(np.linalg.norm(residual)) / tangent_norm
    answer = SimpleNamespace(
        p=p,
        slope=float(gradient @ p),
        relative_residual=float(np.linalg.norm(residual)) / gradient_norm,
        tangent_relative_residual=tangent_rel,
        constraint_residual=_norm_inf(A @ p),
        step_norm=float(np.linalg.norm(p)),
        endpoint_system=endpoint_only,
    )
    if expose_inverse:
        answer.inverse = inverse
        answer.model_product = lambda value: B @ value + V @ (V.T @ value)
    return answer
