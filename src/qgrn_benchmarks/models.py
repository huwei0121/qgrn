"""Five endpoint-control models on their published time grids.

`oracle` computes only the scalar value and analytic adjoint gradient. A separate
`residual_oracle` exposes the exact endpoint least-squares representation for GRN;
ordinary BFGS does not pay for its Jacobian. `rollout` uses independent scipy.expm.
All flattened controls/residual matrices use Julia-compatible Fortran ordering.
Generated starts use NumPy's RNG; paired benchmark starts are loaded from files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh, expm

DEFAULT_SUITE = Path(__file__).with_name("endpoint_models.json")
TASK_IDS = (
    "machnes_c4_t7",
    "machnes_c4_t12",
    "jandura_cz",
    "jandura_c2z",
    "heule_cnot3_alternating_t70",
)
PUBLISHED_GRIDS = dict(
    zip(TASK_IDS, ((128, 7.0), (128, 12.0), (99, 7.612), (399, 16.43), (70, 70.0)))
)


@dataclass(frozen=True)
class OracleResult:
    f: float
    g: np.ndarray


@dataclass(frozen=True)
class ResidualResult(OracleResult):
    r: np.ndarray
    Jr: np.ndarray


def _complex_data(array):
    a = np.asarray(array)
    return {
        "shape": list(a.shape),
        "real": a.real.ravel(order="F").tolist(),
        "imag": a.imag.ravel(order="F").tolist(),
        "order": "F",
    }


def _real_residual(array):
    v = np.asarray(array).ravel(order="F")
    return np.concatenate((v.real, v.imag))


def _spin_operators(n):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)

    def lift(a, j):
        result = np.array([[1.0]], dtype=complex)
        for k in range(n):
            result = np.kron(result, a if k == j else np.eye(2))
        return result

    return [[lift(a, j) for j in range(n)] for a in (x, y, z)]


@dataclass
class PaperTask:
    id: str
    specification: dict[str, Any]
    suite_path: Path
    H0: np.ndarray | None
    Hs: tuple[np.ndarray, ...]
    target: np.ndarray | None
    active: np.ndarray | None
    objective: str

    @property
    def N(self):
        return int(self.specification["intervals"])

    @property
    def M(self):
        return self.N

    @property
    def T(self):
        return float(self.specification["duration"])

    @property
    def h(self):
        return self.T / self.N

    @property
    def dimension(self):
        return int(self.specification["hilbert_or_logical_dimension"])

    @property
    def qubits(self):
        return int(self.specification["qubits"])

    @property
    def n_variables(self):
        return int(self.specification["solver_coordinates"])

    @property
    def channels(self):
        return 1 if self.objective == "rydberg_average" else self.active.shape[0]

    def _coordinates(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != self.n_variables:
            raise ValueError(f"{self.id} requires {self.n_variables} flat F-order coordinates")
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite control coordinates")
        return x

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        if self.objective == "rydberg_average":
            physical = rng.uniform(-np.pi, np.pi, self.N + 1)
            return self.phase_coordinates(physical)
        return rng.normal(size=self.n_variables)

    def phase_lift(self, x):
        x = self._coordinates(x)
        j = np.arange(1, self.N, dtype=float)
        scaled = x[:-1] / np.sqrt(j * (j + 1))
        phi = np.zeros(self.N)
        phi[:-1] = np.cumsum(scaled[::-1])[::-1]
        phi[1:] -= j * scaled
        return np.r_[phi, x[-1]]

    def phase_coordinates(self, physical):
        x = np.asarray(physical, dtype=float)
        if x.shape != (self.N + 1,) or not np.all(np.isfinite(x)):
            raise ValueError("Expected N physical phases followed by theta")
        j = np.arange(1, self.N, dtype=float)
        return np.r_[(np.cumsum(x[:-2]) - j * x[1:-1]) / np.sqrt(j * (j + 1)), x[-1]]

    def _project_phase_gradient(self, gradient):
        j = np.arange(1, self.N, dtype=float)
        return np.r_[
            (np.cumsum(gradient[: self.N - 1]) - j * gradient[1 : self.N]) / np.sqrt(j * (j + 1)),
            gradient[-1],
        ]

    def metadata(self):
        result = dict(self.specification)
        result.update(
            id=self.id,
            M=self.N,
            T=self.T,
            h=self.h,
            dimension=self.dimension,
            channels=self.channels,
            variables=self.n_variables,
            paper_grid=True,
            objective_kind=self.objective,
            coordinate_order="F",
            added_constraints="none",
            added_running_cost="none",
            initialization="NumPy default_rng(seed); saved paired arrays define benchmark initialization independently of the author RNG",
            author_native_optimizer=False,
            model_source="endpoint_models.json and the cited primary papers",
            scalar_oracle="Exact scalar analytic gradient, without a residual Jacobian",
            residual_oracle="Separate exact endpoint residual and its analytic Jacobian for GRN",
            return_propagation="Independent scipy.linalg.expm at every original propagation interval",
        )
        if self.objective == "rydberg_average":
            result.update(
                objective="1-(|s|^2+survival)/(d(d+1)); binomial multiplicities, leakage and common local Z angle retained",
                omega_max=1.0,
                blockade="infinite",
                decay=0.0,
                coordinate_map="N-1 zero-mean orthonormal Helmert laser-phase coordinates plus theta",
                block_dimensions=[2] * self.qubits,
                multiplicities=[comb(self.qubits, j) for j in range(1, self.qubits + 1)],
                controls="Unit laser amplitude; unbounded PWC phase coordinates and common local Z angle",
            )
        else:
            result.update(
                H0=_complex_data(self.H0),
                Hs=[_complex_data(h) for h in self.Hs],
                target=_complex_data(self.target),
                active_drives=self.active.tolist(),
                active_drive_index_base=0,
                objective="1-|tr(G^* U)|/d"
                if self.objective == "full_trace"
                else "1-|U10+U01|^2/4; computational input subspace only",
            )
            result["controls"] = (
                "First spin only; one unbounded scalar per interval, alternating x then y"
                if self.id.startswith("heule_")
                else "Two independent unbounded quadratures; zero detuning; physical leakage level retained"
                if self.id.startswith("motzoi_")
                else "Independent unbounded local x/y amplitudes on all spins"
            )
        return result

    def _hamiltonian(self, controls, k):
        hamiltonian = self.H0.copy()
        for j in range(self.channels):
            hamiltonian += controls[j, k] * self.Hs[self.active[j, k]]
        return hamiltonian

    def _terminal(self, U):
        if self.objective == "subspace_trace":
            v = U[:, :2]
            overlap = np.vdot(self.target, v)
            residual = (v - self.target * (overlap / 2)) / np.sqrt(2)
            survival = float(np.sum(abs(v[:2]) ** 2))
            metrics = dict(
                direct_f=1 - abs(overlap) ** 2 / 4,
                average_infidelity=1 - (abs(overlap) ** 2 + survival) / 6,
                leakage=float(np.sum(abs(v[2]) ** 2) / 2),
            )
        else:
            overlap = np.vdot(self.target, U)
            if abs(overlap) <= 1e-14 * self.dimension:
                raise ValueError("SINGULAR_OVERLAP_CHART")
            residual = (U - (overlap / abs(overlap)) * self.target) / np.sqrt(2 * self.dimension)
            metrics = dict(
                direct_f=1 - abs(overlap) / self.dimension,
                average_infidelity=self.dimension
                / (self.dimension + 1)
                * (1 - abs(overlap) ** 2 / self.dimension**2),
                leakage=0.0,
            )
        r = _real_residual(residual)
        return overlap, r, metrics

    def _unitary_forward(self, x):
        controls = x.reshape((self.channels, self.N), order="F")
        states = [np.eye(self.dimension, dtype=complex)]
        props, eigvecs, eigvals = [], [], []
        for k in range(self.N):
            ev, v = eigh(self._hamiltonian(controls, k), check_finite=False)
            prop = (v * np.exp(-1j * self.h * ev)) @ v.conj().T
            props.append(prop)
            eigvecs.append(v)
            eigvals.append(ev)
            states.append(prop @ states[-1])
        return states, props, eigvecs, eigvals

    def _frechet_multiplier(self, eigenvalues):
        e = eigenvalues
        return (
            -1j
            * self.h
            * np.exp(-1j * self.h * (e[:, None] + e[None, :]) / 2)
            * np.sinc(self.h * (e[:, None] - e[None, :]) / (2 * np.pi))
        )

    def oracle(self, x, order=1):
        x = self._coordinates(x)
        if order not in (0, 1):
            raise ValueError("Only value and first derivatives are supported")
        if self.objective == "rydberg_average":
            return self._rydberg_oracle(x, order, residual=False)
        states, props, vectors, values = self._unitary_forward(x)
        overlap, r, _ = self._terminal(states[-1])
        f = float(r @ r)
        if not order:
            return OracleResult(f, np.empty(0))
        d = self.dimension
        if self.objective == "subspace_trace":
            target = np.zeros((d, d), dtype=complex)
            target[:, :2] = self.target
            factor = np.conj(overlap) / 2
        else:
            target = self.target
            factor = np.conj(overlap) / (d * abs(overlap))
        gradient = np.empty(self.n_variables)
        suffix = np.eye(d, dtype=complex)
        for k in reversed(range(self.N)):
            v = vectors[k]
            contraction = suffix.conj().T @ target @ states[k].conj().T
            ce = v.conj().T @ contraction @ v
            multiplier = self._frechet_multiplier(values[k])
            for j in range(self.channels):
                b = v.conj().T @ self.Hs[self.active[j, k]] @ v
                dt = np.vdot(ce, multiplier * b)
                gradient[k * self.channels + j] = -np.real(factor * dt)
            suffix = suffix @ props[k]
        return OracleResult(f, gradient)

    def residual_oracle(self, x, order=1):
        x = self._coordinates(x)
        if order not in (0, 1):
            raise ValueError("Only value and first derivatives are supported")
        if self.objective == "rydberg_average":
            return self._rydberg_oracle(x, order, residual=True)
        states, props, vectors, values = self._unitary_forward(x)
        overlap, r, _ = self._terminal(states[-1])
        f = float(r @ r)
        if not order:
            return ResidualResult(f, np.empty(0), r, np.empty((0, 0)))
        jacobian = np.empty((len(r), len(x)))
        suffix = np.eye(self.dimension, dtype=complex)
        for k in reversed(range(self.N)):
            v = vectors[k]
            multiplier = self._frechet_multiplier(values[k])
            for j in range(self.channels):
                dp = v @ (multiplier * (v.conj().T @ self.Hs[self.active[j, k]] @ v)) @ v.conj().T
                du = suffix @ dp @ states[k]
                if self.objective == "subspace_trace":
                    dv = du[:, :2]
                    dt = np.vdot(self.target, dv)
                    dr = (dv - self.target * (dt / 2)) / np.sqrt(2)
                else:
                    dt = np.vdot(self.target, du)
                    dr = (
                        du - 1j * (overlap / abs(overlap)) * self.target * np.imag(dt / overlap)
                    ) / np.sqrt(2 * self.dimension)
                jacobian[:, k * self.channels + j] = _real_residual(dr)
            suffix = suffix @ props[k]
        return ResidualResult(f, 2 * jacobian.T @ r, r, jacobian)

    def _rydberg_forward(self, x):
        physical = self.phase_lift(x)
        phi = physical[:-1]
        theta = physical[-1]
        weights = np.array([comb(self.qubits, j) for j in range(1, self.qubits + 1)])
        states = np.empty((self.qubits, self.N + 1, 2), dtype=complex)
        props = np.empty((self.qubits, self.N, 2, 2), dtype=complex)
        ephase = np.exp(1j * phi)
        for j in range(1, self.qubits + 1):
            c, s = np.cos(np.sqrt(j) * self.h / 2), np.sin(np.sqrt(j) * self.h / 2)
            states[j - 1, 0] = [1, 0]
            for k in range(self.N):
                p = np.array([[c, -1j * s * ephase[k]], [-1j * s * ephase[k].conjugate(), c]])
                props[j - 1, k] = p
                states[j - 1, k + 1] = p @ states[j - 1, k]
        sv = np.zeros(1 + 2 * self.qubits, dtype=complex)
        vv = sv.copy()
        sv[0] = vv[0] = 1
        for j in range(1, self.qubits + 1):
            weight = np.sqrt(weights[j - 1])
            sign = -1 if j == self.qubits else 1
            sv[2 * j - 1 : 2 * j + 1] = weight * states[j - 1, -1]
            vv[2 * j - 1] = sign * weight * np.exp(1j * j * theta)
        overlap = np.vdot(vv, sv)
        d = self.dimension
        rb = (sv - vv * (overlap / d)) / np.sqrt(d)
        rl = np.sqrt(weights / d) * states[:, -1, 1]
        rc = np.r_[np.sqrt(d / (d + 1)) * rb, np.sqrt(1 / (d + 1)) * rl]
        return physical, weights, states, props, sv, vv, overlap, _real_residual(rc)

    def _rydberg_oracle(self, x, order, residual):
        physical, weights, states, props, sv, vv, overlap, r = self._rydberg_forward(x)
        f = float(r @ r)
        d = self.dimension
        if not order:
            return (
                ResidualResult(f, np.empty(0), r, np.empty((0, 0)))
                if residual
                else OracleResult(f, np.empty(0))
            )
        # No residual Jacobian is allocated on the BFGS/scalar path.
        ds = np.zeros((len(sv), self.N + 1), dtype=complex) if residual else None
        physical_gradient = np.zeros(self.N + 1)
        for j in range(1, self.qubits + 1):
            suffix = np.eye(2, dtype=complex)
            weight = weights[j - 1]
            a = states[j - 1, -1, 0]
            sign = -1 if j == self.qubits else 1
            factor = (
                np.conj(overlap) * sign * weight * np.exp(-1j * j * physical[-1])
                + weight * np.conj(a)
            ) / (d * (d + 1))
            sine = np.sin(np.sqrt(j) * self.h / 2)
            for k in reversed(range(self.N)):
                ep = np.exp(1j * physical[k])
                before = states[j - 1, k]
                delta = suffix @ np.array(
                    [sine * ep * before[1], -sine * ep.conjugate() * before[0]]
                )
                physical_gradient[k] += -2 * np.real(factor * delta[0])
                if residual:
                    ds[2 * j - 1 : 2 * j + 1, k] = np.sqrt(weight) * delta
                suffix = suffix @ props[j - 1, k]
            dt = -1j * j * sign * weight * np.exp(-1j * j * physical[-1]) * a
            physical_gradient[-1] += -2 * np.real(np.conj(overlap) * dt) / (d * (d + 1))
        gradient = self._project_phase_gradient(physical_gradient)
        if not residual:
            return OracleResult(f, gradient)
        dv = np.zeros(len(vv), dtype=complex)
        for j in range(1, self.qubits + 1):
            dv[2 * j - 1] = 1j * j * vv[2 * j - 1]
        drb = (ds - vv[:, None] * (vv.conj() @ ds)[None, :] / d) / np.sqrt(d)
        drb[:, -1] = -(dv * overlap + vv * np.vdot(dv, sv)) / (d * np.sqrt(d))
        drl = ds[2::2] / np.sqrt(d)
        drc = np.vstack((np.sqrt(d / (d + 1)) * drb, np.sqrt(1 / (d + 1)) * drl))
        physical_jacobian = np.vstack((drc.real, drc.imag))
        j = np.arange(1, self.N, dtype=float)
        jacobian = np.empty((len(r), self.N))
        jacobian[:, :-1] = (
            np.cumsum(physical_jacobian[:, : self.N - 1], axis=1)
            - j * physical_jacobian[:, 1 : self.N]
        ) / np.sqrt(j * (j + 1))
        jacobian[:, -1] = physical_jacobian[:, -1]
        return ResidualResult(f, 2 * jacobian.T @ r, r, jacobian)

    def rollout(self, x):
        x = self._coordinates(x)
        if self.objective == "rydberg_average":
            physical = self.phase_lift(x)
            phi = physical[:-1]
            theta = float(physical[-1])
            states = [np.tile(np.array([[1.0], [0.0]], dtype=complex), (1, self.qubits))]
            for k in range(self.N):
                following = np.empty_like(states[-1])
                for j in range(1, self.qubits + 1):
                    h = (
                        np.sqrt(j)
                        / 2
                        * np.array([[0, np.exp(1j * phi[k])], [np.exp(-1j * phi[k]), 0]])
                    )
                    following[:, j - 1] = expm(-1j * self.h * h) @ states[-1][:, j - 1]
                states.append(following)
            a = states[-1][0]
            w = np.array([comb(self.qubits, j) for j in range(1, self.qubits + 1)])
            sign = np.ones(self.qubits)
            sign[-1] = -1
            overlap = 1 + np.sum(sign * w * np.exp(-1j * np.arange(1, self.qubits + 1) * theta) * a)
            survival = 1 + np.sum(w * abs(a) ** 2)
            value = float(
                1 - (abs(overlap) ** 2 + survival) / (self.dimension * (self.dimension + 1))
            )
            unitarity = max(float(np.max(abs(np.sum(abs(s) ** 2, axis=0) - 1))) for s in states)
            controls = phi.reshape((1, self.N))
            metrics = {"leakage": float(np.sum(w * abs(states[-1][1]) ** 2) / self.dimension)}
        else:
            u = x.reshape((self.channels, self.N), order="F")
            states = [np.eye(self.dimension, dtype=complex)]
            for k in range(self.N):
                states.append(expm(-1j * self.h * self._hamiltonian(u, k)) @ states[-1])
            _, _, metrics = self._terminal(states[-1])
            value = float(metrics["direct_f"])
            unitarity = max(
                float(np.linalg.norm(s.conj().T @ s - np.eye(self.dimension))) for s in states
            )
            controls = np.zeros((len(self.Hs), self.N))
            for k in range(self.N):
                for j in range(self.channels):
                    controls[self.active[j, k], k] = u[j, k]
            theta = None
        return dict(
            states=states,
            controls=controls,
            times=np.linspace(0, self.T, self.N + 1),
            coordinates=x.copy(),
            theta=theta,
            final_objective=value,
            unitarity=unitarity,
            metrics=metrics,
            representation="complete_original_grid_independent_expm",
        )


def task_ids():
    return list(TASK_IDS)


def build_task(task_id, suite_path=None):
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown endpoint task: {task_id}")
    path = Path(suite_path) if suite_path is not None else DEFAULT_SUITE
    suite = json.loads(path.read_text(encoding="utf-8-sig"))
    if suite.get("schema") != "quantum_paper_nine_suite_v1":
        raise ValueError("Unsupported suite schema")
    if len(suite["tasks"]) != len(TASK_IDS) or {s["id"] for s in suite["tasks"]} != set(TASK_IDS):
        raise ValueError("Expected the five declared endpoint models")
    spec = dict(next(s for s in suite["tasks"] if s["id"] == task_id))
    n = int(spec["qubits"])
    N = int(spec["intervals"])
    if (N, float(spec["duration"])) != PUBLISHED_GRIDS[task_id]:
        raise ValueError(
            "This task identifier is reserved for its published full grid and duration"
        )
    spec["source_record"] = dict(suite["sources"][spec["source"]])
    if not np.isclose(spec["timestep"], spec["duration"] / N, rtol=1e-14, atol=0):
        raise ValueError("Declared duration/grid/timestep disagree")
    if task_id.startswith("jandura_"):
        task = PaperTask(task_id, spec, path, None, (), None, None, "rydberg_average")
        if task.n_variables != N:
            raise ValueError("Rydberg quotient dimension must be N")
        return task
    xs, ys, zs = _spin_operators(n)
    if task_id == "heule_cnot3_alternating_t70":
        H0 = sum(
            (xs[j] @ xs[j + 1] + ys[j] @ ys[j + 1] + zs[j] @ zs[j + 1]) / 4 for j in range(n - 1)
        )
        Hs = (xs[0] / 2, ys[0] / 2)
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        target = np.kron(np.eye(2), cnot)
        active = (np.arange(N) % 2).reshape((1, N))
    else:
        Hs = tuple(h / 2 for j in range(n) for h in (xs[j], ys[j]))
        active = np.tile(np.arange(2 * n)[:, None], (1, N))
        H0 = sum(zs[j] @ zs[k] / 2 for j in range(n - 1) for k in range(j + 1, n))
        Hcs = sum(zs[j] @ zs[k] / 2 for j, k in ((0, 1), (1, 2), (2, 3), (3, 0)))
        target = expm(-1j * np.pi * Hcs / 2)
    task = PaperTask(task_id, spec, path, H0, Hs, target, active, "full_trace")
    if task.n_variables != task.channels * N:
        raise ValueError("Declared control dimension differs from the model")
    return task


def initial(task, seed):
    return task.initial(seed)


def metadata(task):
    return task.metadata()


def oracle(task, x, order=1):
    return task.oracle(x, order)


def residual_oracle(task, x, order=1):
    return task.residual_oracle(x, order)


def rollout(task, x):
    return task.rollout(x)
