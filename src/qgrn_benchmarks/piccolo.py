"""Complete official Piccolo X/CNOT bridges, paired with native MadNLP.

Julia exports the paired full initial state. Python reads that saved input and
includes graph construction and initial restoration in its online time budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from qgrn.compat import GRNOptions

from .io import source_hashes
from .pico import array, clean, complex_array, pack, states_from_slice
from .smooth_control import PICOProblem, solve_pico

TASKS = ("first_gate", "two_qubit_zoh")
BUDGETS = dict(first_gate=30.0, two_qubit_zoh=60.0)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            clean(value),
            stream,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return perf_counter() - started


def load_input(path, task=None, seed=None):
    raw = read(path)
    if raw.get("schema") != "modern_piccolo_shared_input_v1" or raw.get("task") not in TASKS:
        raise ValueError("Expected complete official modern Piccolo input")
    if task is not None and raw["task"] != task or seed is not None and raw["seed"] != seed:
        raise ValueError("Shared input task or seed mismatch")
    digest = hashlib.sha256(np.asarray(raw["initial_primal"], dtype="<f8").tobytes()).hexdigest()
    if digest != raw["initial_sha256"]:
        raise ValueError("Complete initial primal hash differs")
    m = raw["model"]
    expected = dict(
        K=100 if raw["task"] == "first_gate" else 200,
        Q=100.0,
        R=0.01,
        acceleration=1.0,
        propagation="exact",
        regularizer="time_weighted",
        free_time=True,
        loss="process" if raw["task"] == "first_gate" else "average",
    )
    if any(m[k] != v for k, v in expected.items()):
        raise ValueError("Author model mismatch")
    return raw


def build_problem(raw):
    m = raw["model"]
    return PICOProblem(
        complex_array(m["H0"]),
        [complex_array(H) for H in m["Hs"]],
        complex_array(m["target"]),
        K=m["K"],
        h=m["h"],
        amplitude=m["amplitude"],
        acceleration=m["acceleration"],
        Q=m["Q"],
        R=m["R"],
        free_time=True,
        h_bounds=(0.0, np.inf),
        propagation="exact",
        loss=m["loss"],
        regularizer="time_weighted",
    )


def primal_slice(model, primal):
    meta = model["native_metadata"]
    z = np.asarray(primal, float).reshape((meta["node_dimension"], meta["N"]), order="F")
    components = meta["components"]

    def field(name):
        return z[np.asarray(components[name], int) - 1]

    drive, state = meta["drive_name"], meta["state_name"]
    # Both official tasks use named h/Delta-t components. Infer it from the
    # exact initial timestep metadata rather than hardcoding a Unicode label.
    used = {drive, "d" + drive, "dd" + drive, state, "t"}
    candidates = [
        name for name, indices in components.items() if name not in used and len(indices) == 1
    ]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous timestep component: {candidates}")
    return dict(
        states=pack(field(state)),
        u=pack(field(drive)),
        du=pack(field("d" + drive)),
        ddu=pack(field("dd" + drive)),
        dt=field(candidates[0]).ravel(),
        times=field("t").ravel(),
    )


def make_slice(problem, y, states):
    s = problem.unpack(np.asarray(y, float))
    iso = np.column_stack([np.vstack((U.real, U.imag)).ravel(order="F") for U in states])
    return dict(
        states=pack(iso),
        u=pack(s.u),
        du=pack(s.du),
        ddu=pack(s.ddu),
        dt=np.full(problem.base.K, s.h),
        times=np.arange(problem.base.K) * s.h,
    )


def inspect_slice(model, sample):
    H0, Hs, G = (
        complex_array(model["H0"]),
        [complex_array(H) for H in model["Hs"]],
        complex_array(model["target"]),
    )
    u, du, ddu = [array(sample[k]) for k in ("u", "du", "ddu")]
    dt, times = np.asarray(sample["dt"], float), np.asarray(sample["times"], float)
    states = states_from_slice(sample)
    d, K = G.shape[0], model["K"]

    def maximum(x):
        return float(np.max(np.abs(x), initial=0.0))

    def loss(U):
        return (
            float(abs(1 - abs(np.trace(G.conj().T @ U)) ** 2 / d**2))
            if model["loss"] == "process"
            else float(
                abs(
                    1
                    - (
                        abs(np.trace((G.conj().T @ U).conj().T @ (G.conj().T @ U)))
                        + abs(np.trace(G.conj().T @ U)) ** 2
                    )
                    / (d * (d + 1))
                )
            )
        )

    reg = 0.5 * model["R"] * sum(float(np.sum(dt * np.sum(x * x, axis=0))) for x in (u, du, ddu))
    bounds = np.asarray(model["amplitude"], float)[:, None]
    constraints = dict(
        initial_state=maximum(states[0] - np.eye(d)),
        control_chain=maximum(u[:, 1:] - u[:, :-1] - dt[:-1] * du[:, :-1]),
        derivative_chain=maximum(du[:, 1:] - du[:, :-1] - dt[:-1] * ddu[:, :-1]),
        endpoints=max(maximum(u[:, 0]), maximum(u[:, -1])),
        amplitude=max(0.0, float(np.max(np.abs(u[:, 1:-1]) - bounds))),
        acceleration=max(0.0, maximum(ddu) - model["acceleration"]),
        time_bounds=max(0.0, -float(np.min(dt))),
        equal_time=maximum(dt - dt[-1]),
        initial_time=abs(float(times[0])),
        time_chain=maximum(times[1:] - times[:-1] - dt[:-1]),
        state_bounds=max(
            0.0,
            max(maximum(U.real) for U in states[1:]) - 1.0,
            max(maximum(U.imag) for U in states[1:]) - 1.0,
        ),
    )
    U = np.eye(d, dtype=complex)
    defect = 0.0
    for k in range(K - 1):
        H = H0 + sum(u[j, k] * Hj for j, Hj in enumerate(Hs))
        P = expm(-1j * dt[k] * H)
        defect = max(defect, maximum(states[k + 1] - P @ states[k]))
        U = P @ U
    constraints["quantum_dynamics"] = defect
    constraints["primal"] = max(constraints.values())
    return dict(
        final_objective=model["Q"] * loss(states[-1]) + reg,
        terminal_loss=loss(states[-1]),
        regularizer=reg,
        physical_objective=model["Q"] * loss(U) + reg,
        physical_gate_error=loss(U),
        physical_process_infidelity=float(abs(1 - abs(np.trace(G.conj().T @ U)) ** 2 / d**2)),
        constraints=constraints,
        feasibility=constraints["primal"],
        physical_unitarity=float(np.linalg.norm(U.conj().T @ U - np.eye(d))),
    )


def run_python(task, seed, seconds, input_path, maxiter=100000000):
    hashes = source_hashes()
    started = perf_counter()
    raw = load_input(input_path, task, seed)
    problem = build_problem(raw)
    initial = inspect_slice(raw["model"], raw["initial"])
    options = GRNOptions(
        estimator="ratio",
        ratio_update="step_aware",
        seconds=float(seconds),
        maxiter=maxiter,
        stop_on_target=False,
        hardware_tangent=True,
        hardware_model="slack",
    )
    setup = perf_counter() - started
    tick = perf_counter()
    s = raw["initial"]
    result = solve_pico(
        problem,
        array(s["u"]),
        array(s["du"]),
        array(s["ddu"]),
        options,
        h=float(s["dt"][0]),
        started=started,
    )
    solver_seconds = perf_counter() - tick
    solve_return = perf_counter() - started
    tick = perf_counter()
    final_oracle = problem.grn_oracle(np.asarray(result["y"]), derivatives=False)
    final = make_slice(problem, result["y"], final_oracle.states)
    returned = inspect_slice(raw["model"], final)
    elapsed = perf_counter() - started
    parity = abs(returned["final_objective"] - final_oracle.f)
    return clean(
        dict(
            schema="modern_piccolo_probe_grn_v1",
            task=task,
            seed=seed,
            method="grn",
            budget_seconds=seconds,
            status=result["status"],
            stopping_policy="user_requested_time_budget_only",
            stop_on_target=False,
            time_budget_only=True,
            precision_stopping=False,
            options=asdict(options),
            initial_coordinate_sha256=raw["initial_sha256"],
            initial_sha256=raw["initial_sha256"],
            model=raw["model"],
            initial_slice=s,
            initial=initial,
            final_slice=final,
            returned=returned,
            history=result["history"],
            trials=result["trials"],
            restoration=result["restoration"],
            result={
                k: v for k, v in result.items() if k not in ("history", "trials", "states", "final")
            },
            final_objective=returned["final_objective"],
            failure=False,
            return_valid=parity <= 1e-7 * max(1.0, abs(final_oracle.f)),
            objective_parity=parity,
            setup_seconds=setup,
            solver_seconds=solver_seconds,
            solve_return_seconds=solve_return,
            verification_seconds=perf_counter() - tick,
            complete_seconds=elapsed,
            over_budget_seconds=max(0.0, elapsed - seconds),
            shared_input_generation_seconds=raw["generation_seconds"],
            source_hashes=hashes,
            source_unchanged=hashes == source_hashes(),
            input_file=str(Path(input_path).resolve()),
            timing_note="Complete online setup/restoration/solve/final audit included; shared generation, imports, warmup and saving separate",
        )
    )
