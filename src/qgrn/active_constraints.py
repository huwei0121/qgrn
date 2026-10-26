"""Near-boundary tangent models preserving affine control constraints.

The local model adds selected hardware normals to the tangent constraints.
The physical oracle checks nonlinear feasibility at each trial point.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse


class TangentGeometry:
    def __init__(self, base, rows, ids, slacks):
        self.base, self.n = base, base.n
        dense = np.asarray(rows, dtype=float)
        projected = np.vstack([base.project(row) for row in dense])
        norms = np.linalg.norm(projected, axis=1)
        valid = norms > 1e-12 * np.maximum(np.linalg.norm(dense, axis=1), 1e-30)
        dense, projected, norms = dense[valid], projected[valid], norms[valid]
        ids, slacks = np.asarray(ids)[valid], np.asarray(slacks)[valid]
        if len(dense):
            dense, projected = dense / norms[:, None], projected / norms[:, None]
            Q, R, piv = linalg.qr(projected.T, mode="economic", pivoting=True, check_finite=False)
            diagonal = np.abs(np.diag(R))
            rank = int(np.count_nonzero(diagonal > 1e-10 * max(diagonal[0], 1e-30)))
            keep = piv[:rank]
            self.C, self.Q = sparse.csc_matrix(dense[keep]), Q[:, :rank]
            self.ids, self.slacks = ids[keep], slacks[keep]
        else:
            self.C, self.Q = sparse.csc_matrix((0, self.n)), np.empty((self.n, 0))
            self.ids, self.slacks = ids, slacks
        self.A = sparse.vstack((base.A, self.C), format="csc")

    def project(self, value):
        p = self.base.project(value)
        return p - self.Q @ (self.Q.T @ p)


def select_tangent_geometry(base, hardware, gradient, *, margin=0.01, max_active=32):
    """Select nearby bounds whose negative chain-projected gradient points out.

    Rows use g<=0, slack=-g>0. Selection is rebuilt at every accepted point, so an
    inward gradient releases a face. Redundant rows are removed in the chain null
    space. At most max_active linearly independent rows enter the local model.
    """
    slack = np.asarray(hardware.slack, dtype=float)
    C = sparse.csr_matrix(hardware.jacobian)
    if C.shape != (len(slack), base.n) or not np.all(np.isfinite(slack)):
        raise ValueError("Invalid hardware geometry")
    descent = -base.project(gradient)
    outward = np.asarray(C @ descent).ravel()
    row_norm = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    threshold = 1e-12 * row_norm * max(np.linalg.norm(descent), 1e-30)
    candidates = np.flatnonzero((slack <= margin) & (outward > threshold))
    if not len(candidates):
        return base, dict(selected=0, candidates=0, released=0, constraint_ids=[])
    # Earliest local boundary intersections receive priority when the cap binds.
    order = np.argsort(slack[candidates] / outward[candidates], kind="stable")
    selected = candidates[order[:max_active]]
    ids = np.asarray(hardware.ids)[selected]
    geometry = TangentGeometry(base, C[selected].toarray(), ids, slack[selected])
    return geometry, dict(
        selected=int(geometry.C.shape[0]),
        candidates=int(len(candidates)),
        released=0,
        constraint_ids=geometry.ids.tolist(),
        min_selected_slack=float(np.min(geometry.slacks)) if len(geometry.slacks) else None,
    )


def solve_tangent_direction(a, mu, geometry, solve, *, max_releases=8):
    """Release a face with a negative quadratic-model multiplier and re-solve.

    All solves use the same positive regularization mu. Each released face enlarges
    the tangent space; the outer GRN acceptance and retry logic remain responsible
    for globalization. Every returned direction satisfies the affine control chain.
    """
    released = []
    while True:
        direction = solve(a, mu, geometry)
        if not isinstance(geometry, TangentGeometry) or not geometry.C.shape[0]:
            break
        p = direction.p
        if direction.slope >= 0 and np.linalg.norm(geometry.base.project(a.g + a.gb)) > 0:
            # Interior near-face equalities can delete all available descent.
            # Restore the chain direction and determine its feasible length
            # using the nonlinear oracle.
            released.extend(geometry.ids.tolist())
            geometry = geometry.base
            direction = solve(a, mu, geometry)
            direction.active_fallback = "non_descent_face_to_chain"
            break
        residual = geometry.base.project(a.g + a.gb + a.H0 @ p + mu * p + 2 * a.Jr.T @ (a.Jr @ p))
        PCt = np.column_stack([geometry.base.project(row) for row in geometry.C.toarray()])
        gram = np.asarray(geometry.C @ PCt)
        multipliers = linalg.solve(
            gram,
            -np.asarray(geometry.C @ residual).ravel(),
            assume_a="pos",
            check_finite=False,
        )
        tolerance = 1e-9 * max(np.linalg.norm(a.g + a.gb), 1e-30)
        bad = np.flatnonzero(multipliers < -tolerance)
        if not len(bad):
            direction.active_multipliers = multipliers
            break
        if len(released) >= max_releases:
            # A fresh unconstrained-chain direction avoids accepting an invalid
            # multiplier sign after exhausting the local active-set effort.
            released.extend(geometry.ids.tolist())
            geometry = geometry.base
            continue
        remove = int(bad[np.argmin(multipliers[bad])])
        released.append(
            geometry.ids[remove].item()
            if hasattr(geometry.ids[remove], "item")
            else geometry.ids[remove]
        )
        keep = np.arange(geometry.C.shape[0]) != remove
        geometry = (
            TangentGeometry(
                geometry.base,
                geometry.C.toarray()[keep],
                geometry.ids[keep],
                geometry.slacks[keep],
            )
            if np.any(keep)
            else geometry.base
        )
    direction.active_geometry = geometry
    direction.active_released = released
    direction.active_constraint_ids = (
        geometry.ids.tolist() if isinstance(geometry, TangentGeometry) else []
    )
    return direction
