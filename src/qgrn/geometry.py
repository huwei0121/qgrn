"""Affine feasible geometry and explicit restoration adapters."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ._utils import _field, _norm_inf


class AffineGeometry:
    """Restore affine feasibility and project vectors onto ker(A).

    A must have full row rank. A supplied project_tangent method must implement
    the same orthogonal projection used by the affine direction equations.
    """

    def __init__(self, task, n):
        self.task, self.n = task, n
        self.A = sparse.csc_matrix(getattr(task, "A", sparse.csc_matrix((0, n))))
        if self.A.shape[1] != n:
            raise ValueError("Affine constraints have the wrong number of columns")
        self.problem_projector = getattr(task, "project_tangent", None)
        self.factor = (
            splu((self.A @ self.A.T).tocsc())
            if self.A.shape[0] and self.problem_projector is None
            else None
        )
        if getattr(task, "bounds", None) is not None and not hasattr(task, "grn_oracle"):
            raise ValueError("Bounds require an explicit grn_oracle and restoration adapter")

    def project(self, x):
        if self.problem_projector is not None:
            return self.problem_projector(x)
        return x - self.A.T @ self.factor.solve(self.A @ x) if self.factor is not None else x.copy()

    def restore(self, x):
        if hasattr(self.task, "restore_initial"):
            restored = self.task.restore_initial(x.copy())
            if isinstance(restored, tuple):
                return np.asarray(restored[0], dtype=float), restored[1]
            return np.asarray(restored, dtype=float), {"policy": "explicit_problem_adapter"}
        y = self.project(x)
        for _ in range(2):
            if _norm_inf(self.A @ y) < 1e-10:
                break
            y = self.project(y)
        return y, dict(
            initial_chain=_norm_inf(self.A @ x),
            final_chain=_norm_inf(self.A @ y),
            scaled_change=float(np.linalg.norm(y - x)),
            contraction=1.0,
            policy="identity or supplied affine projection",
        )

    def feasibility(self, x):
        if hasattr(self.task, "feasibility"):
            value = self.task.feasibility(x)
            return float(_field(value, "primal", value))
        return _norm_inf(self.A @ x)
