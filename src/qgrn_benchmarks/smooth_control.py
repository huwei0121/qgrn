"""Full smooth-control PICO feasible-graph adapter for the Python GRN solver.

The optimization coordinates are scaled (u, h*du, h**2*ddu, log(h/h_ref)); states are
recovered causally, and the complete affine derivative chain remains enforced.
K denotes nodes, hence there are K-1 physical propagation intervals. A uniform
free step is supported; independent steps, open systems, subspace targets and
extra hardware constraints require separate explicit models.

Sparse linear solves replace inverse formation. By default H0 uses the original
positive Gauss-Newton/barrier model. The optional exact_rank2 barrier model adds
the shared-time composition curvature without changing the physical objective,
hardware bounds, barrier function or restoration rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu


def _inf_norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def _vec(x):
    return np.asarray(x).ravel(order="F")


@dataclass(frozen=True)
class ControlModel:
    H0: np.ndarray
    Hs: tuple[np.ndarray, ...]
    target: np.ndarray
    dimension: int
    channels: int
    K: int
    bound: np.ndarray
    acceleration_bound: np.ndarray
    Q: float
    R: float
    propagation: str


class GRNProblem:
    """Full-unitary smooth controls with fixed or uniform free time step.

    Use ``pack_initial(u, du, ddu)`` for the input to ``solve_grn``. Restoration is
    then performed once inside the timed solve. ``initial_state`` is also exposed
    for direct model/oracle work, matching the Julia API.
    """

    def __init__(
        self,
        H0,
        Hs,
        target,
        *,
        K,
        h,
        amplitude,
        acceleration,
        Q,
        R,
        free_time=True,
        h_bounds=(0.0, math.inf),
        propagation="exact",
        loss="process",
        regularizer="time_weighted",
        barrier_curvature="gn",
    ):
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 2:
            raise ValueError("K must be an integer >= 2")
        if not isinstance(free_time, (bool, np.bool_)):
            raise ValueError("free_time must be bool; independent steps are unsupported")
        if loss not in ("process", "average", "unsquared"):
            raise ValueError("Unsupported full-unitary fidelity loss")
        if propagation not in ("exact", "pade22"):
            raise ValueError("Propagation must be exact or pade22")
        if regularizer not in ("time_weighted", "node_sum"):
            raise ValueError("Unsupported regularizer")
        if barrier_curvature not in ("gn", "exact_rank2"):
            raise ValueError("barrier_curvature must be gn or exact_rank2")
        H0, target = np.asarray(H0, complex), np.asarray(target, complex)
        Hs = tuple(np.asarray(H, complex) for H in Hs)
        d = H0.shape[0] if H0.ndim == 2 else 0
        m = len(Hs)
        if d < 2 or H0.shape != (d, d) or target.shape != (d, d):
            raise ValueError("Matching full square target and Hamiltonian are required")
        if not m or not all(H.shape == (d, d) for H in Hs):
            raise ValueError("At least one matching full drive Hamiltonian is required")
        if not all(np.all(np.isfinite(a)) for a in (H0, target, *Hs)):
            raise ValueError("Hamiltonians and target must be finite")
        if not all(np.linalg.norm(H - H.conj().T) < 1e-10 for H in (H0, *Hs)):
            raise ValueError("Hamiltonians must be Hermitian")
        if np.linalg.norm(target.conj().T @ target - np.eye(d)) >= 1e-10:
            raise ValueError("Target must be full unitary; subspace models are unsupported")
        if not (
            math.isfinite(h)
            and h > 0
            and math.isfinite(Q)
            and Q > 0
            and math.isfinite(R)
            and R >= 0
        ):
            raise ValueError("Require finite h>0, Q>0 and R>=0")
        if len(h_bounds) != 2:
            raise ValueError("h_bounds must contain two scalar bounds")
        lo, hi = map(float, h_bounds)
        if not (math.isfinite(lo) and lo >= 0 and not math.isnan(hi) and lo < hi):
            raise ValueError("Require 0 <= lower < upper time bound")
        if not lo <= h <= hi:
            raise ValueError("Initial h is outside its bounds")
        b = np.full(m, amplitude, float) if np.isscalar(amplitude) else np.asarray(amplitude, float)
        ab = (
            np.full(m, acceleration, float)
            if np.isscalar(acceleration)
            else np.asarray(acceleration, float)
        )
        if (
            b.shape != (m,)
            or ab.shape != (m,)
            or not all(np.all(np.isfinite(a)) and np.all(a > 0) for a in (b, ab))
        ):
            raise ValueError("Finite positive hardware bounds are required per channel")

        self.p, self.nc = m * K, 3 * m * K
        self.n = self.nc + int(free_time)
        self.h_ref, self.h_bounds = float(h), (lo, hi)
        self.free_time, self.loss, self.regularizer = bool(free_time), loss, regularizer
        self.barrier_curvature = barrier_curvature
        self.base = ControlModel(
            H0.copy(),
            tuple(H.copy() for H in Hs),
            target.copy(),
            d,
            m,
            int(K),
            b.copy(),
            ab.copy(),
            float(Q),
            float(R),
            propagation,
        )
        rows, cols, vals, eqscale = [], [], [], []
        row = 0
        for offset in (0, self.p):
            for k in range(K - 1):
                for j in range(m):
                    i = offset + k * m + j
                    rows.extend((row, row, row))
                    cols.extend((i + m, i, i + self.p))
                    vals.extend((1.0, -1.0, -1.0))
                    eqscale.append(1.0 / b[j])
                    row += 1
        for k in (0, K - 1):
            for j in range(m):
                rows.append(row)
                cols.append(k * m + j)
                vals.append(1.0)
                eqscale.append(1.0 / b[j])
                row += 1
        chain = sparse.csc_matrix((vals, (rows, cols)), shape=(row, self.nc))
        control_scale = np.tile(np.tile(1.0 / b, K) / math.sqrt(K), 3)
        self.D = np.concatenate((control_scale, [1.0])) if free_time else control_scale
        A0 = sparse.diags(eqscale) @ chain @ sparse.diags(1.0 / control_scale)
        self.A = (
            sparse.hstack((A0, sparse.csc_matrix((row, 1))), format="csc")
            if free_time
            else A0.tocsc()
        )
        # A has full row rank; keep the projector as a sparse factorization.
        self.projector = splu((self.A @ self.A.T).tocsc())
        mask = np.ones(self.p, dtype=bool)
        mask[:m] = mask[-m:] = False
        self.bi = np.concatenate((np.flatnonzero(mask), np.arange(2 * self.p, 3 * self.p)))
        self.bb = np.concatenate((np.tile(b, K)[mask], np.tile(ab, K)))
        self.objective_weight = float(Q)

    def project_tangent(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"Expected {self.n} coordinates")
        return x - self.A.T @ self.projector.solve(self.A @ x)

    def _initial_arrays(self, u, du, ddu, h):
        arrays = tuple(np.asarray(a, float) for a in (u, du, ddu))
        shape = (self.base.channels, self.base.K)
        if not all(a.shape == shape and np.all(np.isfinite(a)) for a in arrays):
            raise ValueError(f"u, du and ddu must be finite channel-by-node arrays of size {shape}")
        if not math.isfinite(h) or h <= 0:
            raise ValueError("Initial h must be finite and positive")
        if not self.free_time and h != self.h_ref:
            raise ValueError("Fixed step cannot be changed during initialization")
        return arrays

    def pack_initial(self, u, du, ddu, *, h=None):
        """Pack raw physical input; no projection, contraction or time clipping."""
        h = self.h_ref if h is None else float(h)
        u, du, ddu = self._initial_arrays(u, du, ddu, h)
        z = np.concatenate(
            (
                _vec(u),
                h * _vec(du),
                h * h * _vec(ddu),
                [math.log(h / self.h_ref)] if self.free_time else [],
            )
        )
        y = z * self.D
        if not np.all(np.isfinite(y)):
            raise ValueError("Initial data overflows scaled coordinates")
        return y

    def initial_state(self, u, du, ddu, *, h=None):
        h = self.h_ref if h is None else float(h)
        u, du, ddu = self._initial_arrays(u, du, ddu, h)
        input_h = h
        if self.free_time:
            lo, hi = self.h_bounds
            h = max(h, lo + 1e-6 * max(1.0, abs(lo)))
            if math.isfinite(hi):
                h = min(h, hi - 1e-6 * (hi - lo))
            if not lo < h < hi:
                raise ValueError("Time interval is too narrow for interior initialization")
        y = self.pack_initial(u, du, ddu, h=h)
        before, initial = _inf_norm(self.A @ y), y.copy()
        y = self.project_tangent(y)
        for _ in range(2):
            if _inf_norm(self.A @ y) < 1e-10:
                break
            y = self.project_tangent(y)
        z = y / self.D
        physical = np.concatenate(
            (z[: self.p], z[self.p : 2 * self.p] / h, z[2 * self.p : 3 * self.p] / h**2)
        )
        largest_ratio = float(np.max(np.abs(physical[self.bi]) / self.bb))
        shrink = min(1.0, 0.95 / largest_ratio) if largest_ratio else 1.0
        y[: self.nc] *= shrink
        if not np.all(np.isfinite(y)):
            raise ValueError("Initial data overflows scaled coordinates")
        diagnostics = dict(
            initial_chain=before,
            final_chain=_inf_norm(self.A @ y),
            scaled_change=float(np.linalg.norm(y - initial)),
            contraction=shrink,
            input_h=input_h,
            restored_h=h,
            policy="Affine chain projection and common contraction; states recovered causally",
        )
        return SimpleNamespace(y=y, diagnostics=diagnostics)

    def restore_initial(self, y):
        s = self.unpack(y)
        result = self.initial_state(s.u, s.du, s.ddu, h=s.h)
        return result.y, result.diagnostics

    def unpack(self, y):
        y = np.asarray(y, float)
        if y.shape != (self.n,):
            raise ValueError(f"Expected {self.n} coordinates")
        z = y / self.D
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h = float(self.h_ref * np.exp(z[-1])) if self.free_time else self.h_ref
            shape = (self.base.channels, self.base.K)
            u = z[: self.p].reshape(shape, order="F")
            du = (z[self.p : 2 * self.p] / h).reshape(shape, order="F")
            # Multiplication preserves NumPy's nonfinite-result path for an
            # invalid finite log-step; Python h**2 can raise before bounds run.
            ddu = (z[2 * self.p : 3 * self.p] / (h * h)).reshape(shape, order="F")
        return SimpleNamespace(z=z, h=h, u=u, du=du, ddu=ddu)

    def endpoint_residual(self, U):
        b = self.base
        c = np.vdot(b.target, U) / b.dimension
        v = _vec((U - c * b.target) / math.sqrt(b.dimension))
        return np.concatenate((v.real, v.imag))

    def _hardware_box_jacobian(self, zbox, h):
        """Derivative of normalized signed hardware coordinates, in y units."""
        facs = np.where(self.bi >= 2 * self.p, 1 / h**2, 1.0)
        idx = np.arange(len(self.bi))
        Jb = sparse.csc_matrix(
            (facs / (self.bb * self.D[self.bi]), (idx, self.bi)),
            shape=(len(self.bi), self.n),
        )
        if self.free_time:
            dd = np.flatnonzero(self.bi >= 2 * self.p)
            Jb += sparse.csc_matrix(
                (-2 * zbox[dd] / self.D[-1], (dd, np.full(len(dd), self.n - 1))),
                shape=Jb.shape,
            )
        return Jb

    def _hardware_geometry_from(self, zbox, Jb, h):
        slack = [1 - zbox, 1 + zbox]
        jacobian = [Jb, -Jb]
        labels = [
            f"{('acceleration' if i >= 2 * self.p else 'amplitude')}:{i % self.p}" for i in self.bi
        ]
        ids = [f"{name}:upper" for name in labels] + [f"{name}:lower" for name in labels]
        if self.free_time:
            for name, bound, sign in (
                ("lower", self.h_bounds[0], 1.0),
                ("upper", self.h_bounds[1], -1.0),
            ):
                if math.isfinite(bound) and bound > 0:
                    scale = max(bound, 1.0)
                    slack.append(np.array([sign * (h - bound) / scale]))
                    jacobian.append(
                        sparse.csc_matrix(
                            ([-sign * h / scale / self.D[-1]], ([0], [self.n - 1])),
                            shape=(1, self.n),
                        )
                    )
                    ids.append(f"time:{name}")
        return SimpleNamespace(
            slack=np.concatenate(slack),
            jacobian=sparse.vstack(jacobian, format="csc"),
            ids=tuple(ids),
        )

    def hardware_geometry(self, y):
        """Normalized hardware constraints g <= 0, with slack=-g and C=dg/dy.

        Each signed box coordinate z contributes z-1 and -z-1. Positive finite
        time bounds use slacks (h-lo)/max(lo,1), (hi-h)/max(hi,1). A zero lower
        time bound is enforced by log-time coordinates, so has no explicit row.
        End-point amplitudes fixed by A are omitted as in the barrier itself.
        This method does not propagate quantum states or require strict slacks.
        """
        s = self.unpack(y)
        if not (np.all(np.isfinite(s.z)) and math.isfinite(s.h) and s.h > 0):
            raise ValueError("Finite coordinates and positive finite time are required")
        physical = np.concatenate((_vec(s.u), _vec(s.du), _vec(s.ddu)))
        zbox = physical[self.bi] / self.bb
        if not np.all(np.isfinite(zbox)):
            raise ValueError("Nonfinite normalized hardware coordinates")
        return self._hardware_geometry_from(zbox, self._hardware_box_jacobian(zbox, s.h), s.h)

    def local_map(self, u, h, derivatives=True):
        b = self.base
        H = b.H0.copy()
        for uj, Hj in zip(u, b.Hs):
            H += uj * Hj
        G = -1j * h * H
        if b.propagation == "pade22":
            E = np.eye(b.dimension, dtype=complex)
            B, C = E - G / 2 + G @ G / 12, E + G / 2 + G @ G / 12
            fac = linalg.lu_factor(B)
            P = linalg.lu_solve(fac, C)
            if not derivatives:
                return SimpleNamespace(P=P, dP=[], dTau=np.zeros_like(P))

            def deriv(D):
                dB, dC = -D / 2 + (G @ D + D @ G) / 12, D / 2 + (G @ D + D @ G) / 12
                return linalg.lu_solve(fac, dC - dB @ P)

            return SimpleNamespace(P=P, dP=[deriv(-1j * h * Hj) for Hj in b.Hs], dTau=deriv(G))
        ev, V = linalg.eigh(H)
        P = (V * np.exp(-1j * h * ev)) @ V.conj().T
        if not derivatives:
            return SimpleNamespace(P=P, dP=[], dTau=np.zeros_like(P))
        F = (
            -1j
            * h
            * np.exp(-1j * h * (ev[:, None] + ev[None, :]) / 2)
            * np.sinc(h * (ev[:, None] - ev[None, :]) / (2 * np.pi))
        )
        return SimpleNamespace(
            P=P,
            dP=[V @ (F * (V.conj().T @ Hj @ V)) @ V.conj().T for Hj in b.Hs],
            dTau=G @ P,
        )

    def grn_oracle(self, y, *, derivatives=True, beta=0.0):
        y = np.asarray(y, float)
        if y.shape != (self.n,):
            raise ValueError(f"Expected {self.n} coordinates")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")

        def invalid(reason):
            return SimpleNamespace(valid=False, reason=reason, f=math.inf, barrier=math.inf)

        if not np.all(np.isfinite(y)):
            return invalid("nonfinite_coordinates")
        b, s = self.base, self.unpack(y)
        h = s.h
        lo, hi = self.h_bounds
        time_ok = lo < h < hi if self.free_time else lo <= h <= hi
        if not (math.isfinite(h) and h > 0 and time_ok):
            return invalid("time_bounds_or_overflow")
        physical = np.concatenate((_vec(s.u), _vec(s.du), _vec(s.ddu)))
        zbox = physical[self.bi] / self.bb
        if not (np.all(np.isfinite(zbox)) and np.all(np.abs(zbox) < 1)):
            return invalid("hardware_bounds_or_overflow")
        states, maps = [np.eye(b.dimension, dtype=complex)], []
        for k in range(b.K - 1):
            a = self.local_map(s.u[:, k], h, derivatives)
            maps.append(a)
            states.append(a.P @ states[-1])
        U = states[-1]
        overlap = float(abs(np.vdot(b.target, U)) / b.dimension)
        r = self.endpoint_residual(U)
        Jr = np.zeros((len(r), self.n))
        if derivatives:
            suffix = np.eye(b.dimension, dtype=complex)
            dTau = np.zeros_like(suffix)
            for k in range(b.K - 2, -1, -1):
                for j in range(b.channels):
                    col = k * b.channels + j
                    Jr[:, col] = (
                        self.endpoint_residual(suffix @ maps[k].dP[j] @ states[k]) / self.D[col]
                    )
                if self.free_time:
                    dTau += suffix @ maps[k].dTau @ states[k]
                suffix = suffix @ maps[k].P
            if self.free_time:
                Jr[:, -1] = self.endpoint_residual(dTau) / self.D[-1]
        if self.loss == "unsquared":
            if overlap == 0 and derivatives:
                return invalid("nondifferentiable_unsquared_zero_overlap")
            q2 = 1.0 - float(r @ r)
            q = math.sqrt(q2) if derivatives and q2 > 0 else overlap
            c = 1 / math.sqrt(1 + q)
            if derivatives:
                Jr = c * Jr + (c**3 / (2 * q)) * np.outer(r, r @ Jr)
            r = c * r
        elif self.loss == "average":
            c = math.sqrt(b.dimension / (b.dimension + 1))
            r, Jr = c * r, c * Jr
        r, Jr = math.sqrt(b.Q) * r, math.sqrt(b.Q) * Jr
        weight = h if self.regularizer == "time_weighted" else 1.0
        reg = math.sqrt(b.R * weight) * physical
        alpha = np.repeat(
            [0.5, -0.5, -1.5] if self.regularizer == "time_weighted" else [0.0, -1.0, -2.0],
            self.p,
        )
        facs = np.concatenate((np.ones(self.p), np.full(self.p, 1 / h), np.full(self.p, 1 / h**2)))
        diagJ = math.sqrt(b.R * weight) * facs / self.D[: self.nc]
        Jreg = sparse.diags(diagJ, format="csc")
        if self.free_time:
            Jreg = sparse.hstack(
                (Jreg, sparse.csc_matrix((alpha * reg / self.D[-1])[:, None])),
                format="csc",
            )
        f = float(r @ r + reg @ reg / 2)
        tau = beta / max(len(zbox), 1)
        barrier = float(-tau * np.sum(np.log1p(-zbox * zbox)))
        barrier_time_slacks = []
        g = 2 * Jr.T @ r + Jreg.T @ reg
        H0 = (Jreg.T @ Jreg).tocsc()
        gb = np.zeros(self.n)
        correction = (
            sparse.csc_matrix((self.n, self.n)) if self.barrier_curvature == "exact_rank2" else None
        )
        if beta > 0 and derivatives:
            Jb = self._hardware_box_jacobian(zbox, h)
            v = 2 * tau * zbox / (1 - zbox * zbox)
            w = 2 * tau * (1 + zbox * zbox) / (1 - zbox * zbox) ** 2
            gb = Jb.T @ v
            H0 += Jb.T @ sparse.diags(w) @ Jb
            if self.free_time and self.barrier_curvature == "exact_rank2":
                dd = np.flatnonzero(self.bi >= 2 * self.p)
                cols = self.bi[dd]
                mixed = -2 * v[dd] / (h * h * self.bb[dd] * self.D[cols] * self.D[-1])
                diagonal = 4 * float(v[dd] @ zbox[dd]) / self.D[-1] ** 2
                timecols = np.full(len(dd), self.n - 1)
                correction += sparse.csc_matrix(
                    (
                        np.r_[mixed, mixed, diagonal],
                        (
                            np.r_[cols, timecols, self.n - 1],
                            np.r_[timecols, cols, self.n - 1],
                        ),
                    ),
                    shape=H0.shape,
                )
        for bound, sign in ((lo, 1.0), (hi, -1.0)):
            if self.free_time and math.isfinite(bound) and bound > 0 and beta > 0:
                slack = sign * (h - bound)
                barrier_time_slacks.append(slack)
                barrier -= tau * math.log(slack / max(bound, 1.0))
                gb[-1] -= tau * sign * h / slack / self.D[-1]
                if derivatives:
                    H0 += sparse.csc_matrix(
                        (
                            [tau * h * h / slack**2 / self.D[-1] ** 2],
                            ([self.n - 1], [self.n - 1]),
                        ),
                        shape=H0.shape,
                    )
                    if self.barrier_curvature == "exact_rank2":
                        correction += sparse.csc_matrix(
                            (
                                [-tau * sign * h / slack / self.D[-1] ** 2],
                                ([self.n - 1], [self.n - 1]),
                            ),
                            shape=H0.shape,
                        )
        if derivatives and self.barrier_curvature == "exact_rank2":
            H0 += correction
        finite = all(math.isfinite(a) for a in (f, barrier, overlap)) and all(
            np.all(np.isfinite(a)) for a in (g, gb, r, H0.data, Jr)
        )
        if not finite:
            return invalid("nonfinite_oracle")
        # The remainder estimators require a fixed-chart residual whose squared
        # norm is the complete physical objective. In free-time coordinates the
        # running-cost residual is nonlinear, so omitting it loses model error.
        # Keep its derivative sparse instead of stacking a large dense matrix.
        remainder_residual = np.concatenate((math.sqrt(2.0) * r, reg))
        remainder_jacobian = None
        if derivatives:

            def remainder_jacobian(step):
                return np.concatenate((math.sqrt(2.0) * (Jr @ step), Jreg @ step))

        return SimpleNamespace(
            valid=True,
            reason="valid",
            derivatives=derivatives,
            f=f,
            barrier=barrier,
            g=g,
            gb=gb,
            H0=H0.tocsc(),
            Jr=Jr,
            reg=reg,
            Jreg=Jreg,
            r=r,
            barrier_correction=correction,
            barrier_difference_data=dict(
                weight=tau,
                box=zbox,
                time_slacks=np.asarray(barrier_time_slacks, dtype=float),
            ),
            remainder_residual=remainder_residual,
            remainder_jacobian=remainder_jacobian,
            states=states,
            physical=physical,
            overlap=overlap,
            h=h,
            ell=max(0.0, 1 - overlap),
            gate_error=float(r @ r / b.Q),
            regularizer=float(reg @ reg / 2),
            margin=float(np.min(1 - np.abs(zbox))),
        )

    oracle = grn_oracle

    def direction(self, a, mu):
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError("Regularization mu must be finite and strictly positive")
        if not a.valid or not a.derivatives:
            raise ValueError("A valid derivative oracle is required")
        neq = self.A.shape[0]
        B = a.H0 + sparse.eye(self.n, format="csc") * mu
        V = math.sqrt(2.0) * a.Jr.T
        K = sparse.bmat([[B, self.A.T], [self.A, sparse.csc_matrix((neq, neq))]], format="csc")
        factor = splu(K)
        rhs = np.vstack((np.column_stack((a.g + a.gb, V)), np.zeros((neq, 1 + V.shape[1]))))
        solves = factor.solve(rhs)
        pg, PV = solves[: self.n, 0], solves[: self.n, 1:]
        small = np.eye(V.shape[1]) + V.T @ PV
        cf = linalg.cho_factor((small + small.T) / 2, lower=True, check_finite=False)
        p = -pg + PV @ linalg.cho_solve(cf, V.T @ pg, check_finite=False)
        residual = self.project_tangent(B @ p + V @ (V.T @ p) + a.g + a.gb)
        gradient_norm = max(float(np.linalg.norm(a.g + a.gb)), 1e-30)
        tangent_norm = max(float(np.linalg.norm(self.project_tangent(a.g + a.gb))), 1e-30)
        tangent_rel = float(np.linalg.norm(residual)) / tangent_norm
        for _ in range(2):
            if tangent_rel < 1e-7:
                break
            q = factor.solve(np.concatenate((residual, np.zeros(neq))))[: self.n]
            p += -q + PV @ linalg.cho_solve(cf, V.T @ q, check_finite=False)
            residual = self.project_tangent(B @ p + V @ (V.T @ p) + a.g + a.gb)
            tangent_rel = float(np.linalg.norm(residual)) / tangent_norm
        return SimpleNamespace(
            p=p,
            slope=float((a.g + a.gb) @ p),
            relative_residual=float(np.linalg.norm(residual)) / gradient_norm,
            tangent_relative_residual=tangent_rel,
            constraint_residual=_inf_norm(self.A @ p),
            step_norm=float(np.linalg.norm(p)),
        )

    def feasibility(self, y):
        s, b = self.unpack(y), self.base
        chain = _inf_norm(s.u[:, 1:] - s.u[:, :-1] - s.h * s.du[:, :-1])
        derivative_chain = _inf_norm(s.du[:, 1:] - s.du[:, :-1] - s.h * s.ddu[:, :-1])
        endpoints = max(_inf_norm(s.u[:, 0]), _inf_norm(s.u[:, -1]))
        amplitude = max(0.0, float(np.max(np.abs(s.u) - b.bound[:, None])))
        acceleration = max(0.0, float(np.max(np.abs(s.ddu) - b.acceleration_bound[:, None])))
        time_bounds = max(self.h_bounds[0] - s.h, s.h - self.h_bounds[1], 0.0)
        return dict(
            control_chain=chain,
            derivative_chain=derivative_chain,
            endpoints=endpoints,
            amplitude=amplitude,
            acceleration=acceleration,
            time_bounds=time_bounds,
            primal=max(
                chain,
                derivative_chain,
                endpoints,
                amplitude,
                acceleration,
                time_bounds,
                0.0,
            ),
        )

    def return_fields(self, y, final_raw):
        s, b = self.unpack(y), self.base
        checks = self.feasibility(y)
        checks["modeled_unitarity"] = max(
            float(np.linalg.norm(U.conj().T @ U - np.eye(b.dimension))) for U in final_raw.states
        )
        checks["redundant_state_box"] = max(
            0.0,
            max(float(np.max(np.maximum(np.abs(U.real), np.abs(U.imag)))) for U in final_raw.states)
            - 1,
        )
        return dict(
            states=final_raw.states,
            u=s.u,
            du=s.du,
            ddu=s.ddu,
            h=s.h,
            dt=np.full(b.K, s.h),
            times=np.arange(b.K) * s.h,
            regularizer=final_raw.regularizer,
            propagation=b.propagation,
            loss=self.loss,
            checks=checks,
        )


PICOProblem = GRNProblem


def solve_pico(problem, u, du, ddu, options=None, h=None, *, started=None):
    """Julia-style physical-input entry point, with exactly one restoration.

    The clock includes physical-input restoration and lazy controller import. ``h``
    optionally supplies an initial uniform step; fixed-step problems reject changes.
    The generic controller owns all stopping and adaptation rules.
    """
    started = perf_counter() if started is None else started
    from qgrn.compat import solve_grn

    if not isinstance(problem, GRNProblem):
        raise TypeError("problem must be GRNProblem/PICOProblem")

    class PhysicalInputAdapter:
        def __getattr__(self, name):
            return getattr(problem, name)

        def restore_initial(self, unused_coordinates):
            # Match Julia's physical-input order: move h into its interval
            # before forming h*du and h^2*ddu. A raw pack/unpack round trip
            # can overflow/underflow for a finite, recoverable initial h.
            initial = problem.initial_state(u, du, ddu, h=h)
            return initial.y, initial.diagnostics

    return solve_grn(PhysicalInputAdapter(), np.zeros(problem.n), options, started=started)
