"""Complete 500-node SWAP13 model with the physical terminal gate loss.

The Julia interface generates paired inputs and runs the author Ipopt model.
Both methods use the same correction to the terminal loss and its derivatives
for the author's real/imaginary state layout.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from qgrn.compat import GRNOptions

from .io import source_hashes
from .smooth_control import PICOProblem, solve_pico


def array(a):
    return np.asarray(a["values"], dtype=float).reshape(a["shape"], order="F")


def complex_array(a):
    return (np.asarray(a["real"]) + 1j * np.asarray(a["imag"])).reshape(a["shape"], order="F")


def pack(a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return dict(
            shape=list(a.shape),
            real=a.real.ravel(order="F"),
            imag=a.imag.ravel(order="F"),
            order="F",
        )
    return dict(shape=list(a.shape), values=a.ravel(order="F"), order="F")


def clean(a):
    if isinstance(a, dict):
        return {str(k): clean(v) for k, v in a.items()}
    if isinstance(a, (tuple, list)):
        return [clean(v) for v in a]
    if isinstance(a, np.ndarray):
        return clean(pack(a)) if np.iscomplexobj(a) else clean(a.tolist())
    if isinstance(a, (float, np.floating)):
        return float(a) if np.isfinite(a) else None
    if isinstance(a, (np.integer, np.bool_)):
        return a.item()
    return a


def initial_hash(s):
    values = np.concatenate(
        [array(s[n]).ravel(order="F") for n in ("states", "u", "du", "ddu")] + [np.asarray(s["dt"])]
    )
    return hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()


def load_input(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["schema"] != "author2023_shared_input_v2" or raw["task"] != "author2023_swap13":
        raise ValueError("Expected original complete SWAP13 author input")
    m = raw["model"]
    expected = dict(
        K=500,
        h=0.4,
        Q=200.0,
        R=0.001,
        acceleration=0.05,
        free_time=True,
        propagation="pade22",
        loss="unsquared",
        regularizer="node_sum",
    )
    if any(m[k] != v for k, v in expected.items()) or m["h_bounds"] != [0.2, 0.4]:
        raise ValueError("Input does not use the complete published SWAP13 configuration")
    if complex_array(m["H0"]).shape != (8, 8) or len(m["Hs"]) != 6:
        raise ValueError("Expected dimension eight and six physical controls")
    if not np.allclose(m["amplitude"], 0.08 * np.pi, rtol=0, atol=1e-15):
        raise ValueError("Original amplitude bounds differ")
    if initial_hash(raw["initial"]) != raw["initial_sha256"]:
        raise ValueError("Original complete initial input hash mismatch")
    for name in ("u", "du", "ddu"):
        a = array(raw["initial"][name])
        if a.shape != (6, 500) or not np.all(np.isfinite(a)):
            raise ValueError("Invalid original control arrays")
    if array(raw["initial"]["states"]).shape != (128, 500):
        raise ValueError("All original collocation states are required")
    return raw


def build_problem(raw, problem_options=None):
    m = raw["model"]
    return PICOProblem(
        complex_array(m["H0"]),
        [complex_array(h) for h in m["Hs"]],
        complex_array(m["target"]),
        K=m["K"],
        h=m["h"],
        amplitude=m["amplitude"],
        acceleration=m["acceleration"],
        Q=m["Q"],
        R=m["R"],
        free_time=True,
        h_bounds=tuple(m["h_bounds"]),
        propagation="pade22",
        loss="unsquared",
        regularizer="node_sum",
        **dict(problem_options or {}),
    )


def states_from_slice(s):
    x = array(s["states"])
    d = int(np.sqrt(x.shape[0] / 2))
    return [
        x[:, k].reshape(2 * d, d, order="F")[:d] + 1j * x[:, k].reshape(2 * d, d, order="F")[d:]
        for k in range(x.shape[1])
    ]


def make_slice(problem, y, states):
    s = problem.unpack(y)
    iso = np.column_stack([np.vstack((u.real, u.imag)).ravel(order="F") for u in states])
    return dict(
        u=pack(s.u),
        du=pack(s.du),
        ddu=pack(s.ddu),
        states=pack(iso),
        dt=np.full(problem.base.K, s.h),
    )


def inspect_slice(model, s):
    """Independent all-node original constraints plus Pade and expm replay."""
    H0, Hs, target = (
        complex_array(model["H0"]),
        [complex_array(h) for h in model["Hs"]],
        complex_array(model["target"]),
    )
    u, du, ddu = [array(s[k]) for k in ("u", "du", "ddu")]
    dt, states = np.asarray(s["dt"], dtype=float), states_from_slice(s)
    d, K = H0.shape[0], u.shape[1]

    def maximum(v):
        return float(np.max(np.abs(v), initial=0.0))

    reg = 0.5 * model["R"] * sum(float(np.sum(v * v)) for v in (u, du, ddu))

    def correct_error(U):
        return float(abs(1 - abs(np.vdot(target, U)) / d))

    a = np.vstack((target.real, target.imag)).ravel(order="F")
    x = array(s["states"])[:, -1]
    b = np.concatenate((-a[len(a) // 2 :], a[: len(a) // 2]))
    half_loss = float(abs(1 - np.hypot(a @ x, b @ x) / d))
    constraints = dict(
        initial_state=maximum(states[0] - np.eye(d)),
        control_chain=maximum(u[:, 1:] - u[:, :-1] - dt[:-1] * du[:, :-1]),
        derivative_chain=maximum(du[:, 1:] - du[:, :-1] - dt[:-1] * ddu[:, :-1]),
        endpoints=max(maximum(u[:, 0]), maximum(u[:, -1])),
        amplitude=max(0.0, float(np.max(np.abs(u) - np.asarray(model["amplitude"])[:, None]))),
        acceleration=max(0.0, maximum(ddu) - model["acceleration"]),
        time_bounds=max(
            0.0,
            model["h_bounds"][0] - float(np.min(dt)),
            float(np.max(dt)) - model["h_bounds"][1],
        ),
        equal_time=maximum(dt - dt[0]),
    )
    Up, Ue, defect, pade_difference, expm_difference = (
        np.eye(d, dtype=complex),
        np.eye(d, dtype=complex),
        0.0,
        0.0,
        0.0,
    )
    for k in range(K - 1):
        H = H0 + sum(u[j, k] * h for j, h in enumerate(Hs))
        A = -1j * dt[k] * H
        B, C = np.eye(d) - A / 2 + A @ A / 12, np.eye(d) + A / 2 + A @ A / 12
        defect = max(defect, maximum(B @ states[k + 1] - C @ states[k]))
        Up, Ue = np.linalg.solve(B, C) @ Up, expm(A) @ Ue
        pade_difference = max(pade_difference, maximum(states[k + 1] - Up))
        expm_difference = max(expm_difference, maximum(states[k + 1] - Ue))
    constraints["quantum_dynamics"] = defect
    constraints["primal"] = max(constraints.values())
    return dict(
        final_objective=model["Q"] * correct_error(states[-1]) + reg,
        terminal_loss=correct_error(states[-1]),
        regularizer=reg,
        historical_half_objective=model["Q"] * half_loss + reg,
        historical_half_loss=half_loss,
        pade_objective=model["Q"] * correct_error(Up) + reg,
        pade_gate_error=correct_error(Up),
        physical_objective=model["Q"] * correct_error(Ue) + reg,
        physical_gate_error=correct_error(Ue),
        constraints=constraints,
        feasibility=constraints["primal"],
        stored_pade_difference=pade_difference,
        stored_expm_difference=expm_difference,
        physical_unitarity=float(np.linalg.norm(Ue.conj().T @ Ue - np.eye(d))),
    )


def run_case(input_path, *, seconds=30.0, maxiter=3000, grn_options=None, problem_options=None):
    if not np.isfinite(seconds) or seconds < 0 or isinstance(seconds, bool):
        raise ValueError("seconds must be finite and nonnegative")
    options = GRNOptions(
        **dict(
            grn_options or {},
            seconds=float(seconds),
            maxiter=maxiter,
            stop_on_target=False,
        )
    )
    estimator = getattr(options, "estimator", "ratio")
    method = "grn" if estimator == "ratio" else "grn_" + estimator
    hashes, started, stage = (
        source_hashes(),
        perf_counter(),
        "load_complete_shared_input",
    )
    raw, record = None, {}
    try:
        raw = load_input(input_path)
        record = dict(
            schema="pico_swap_candidate_v1",
            task="pico_swap13",
            seed=raw["seed"],
            method=method,
            budget_seconds=seconds,
            target_early_stopping=False,
            precision_early_stopping=False,
            stopping_policy="user_requested_time_budget_only",
            options=asdict(options),
            problem_options=dict(problem_options or {}),
            input_file=str(Path(input_path).resolve()),
            input_file_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            initial_coordinate_sha256=raw["initial_sha256"],
            initial_slice=raw["initial"],
            model=raw["model"],
            objective_semantics="physical_trace_unsquared_with_original_node_regularizer",
            objective_correction="Column-wise complex structure replaces the historical half-vector loss; same correction for both algorithms.",
            shared_input_generation_seconds=raw["generation_seconds"],
            source_hashes=hashes,
            timing_note="Complete clock starts before input loading/model construction; restoration, solve, full state return and independent all-node replay included. Shared input generation and process/import startup excluded and reported separately. Warmup is separate.",
        )
        stage = "build_complete_pico_model"
        problem = build_problem(raw, problem_options=problem_options)
        s = raw["initial"]
        # The raw input's endpoint and controls need not satisfy collocation.
        # Its physical replay is recorded separately from the native NLP value.
        initial = inspect_slice(raw["model"], s)
        record["initial"] = initial
        record["setup_seconds"] = perf_counter() - started
        stage = "restore_and_solve"
        result = solve_pico(
            problem,
            array(s["u"]),
            array(s["du"]),
            array(s["ddu"]),
            options,
            h=float(s["dt"][0]),
            started=started,
        )
        record["solve_return_seconds"] = perf_counter() - started
        stage = "full_return_and_independent_validation"
        y = result["y"]
        final_oracle = problem.grn_oracle(y, derivatives=False)
        if not final_oracle.valid:
            raise RuntimeError("Returned coordinates have no valid physical rollout")
        final_slice = make_slice(problem, y, final_oracle.states)
        returned = inspect_slice(raw["model"], final_slice)
        for h in result["history"]:
            h["feasibility"] = problem.feasibility(np.asarray(h["x"]))["primal"]
        record.update(
            status=result["status"],
            termination_reason=result.get("termination_reason"),
            history=result["history"],
            trials=result["trials"],
            restoration=result["restoration"],
            result={
                k: v for k, v in result.items() if k not in ("history", "trials", "states", "final")
            },
            returned=returned,
            final_slice=final_slice,
            return_valid=bool(
                abs(returned["final_objective"] - final_oracle.f)
                < 1e-7 * max(1.0, abs(final_oracle.f))
                and returned["feasibility"] < 1e-7
            ),
            returned_primal_feasible=returned["feasibility"] <= 1e-6,
            final_objective=returned["final_objective"],
            failure=False,
        )
    except Exception as exc:
        record.update(
            schema="pico_swap_candidate_v1",
            task="pico_swap13",
            method=method,
            seed=None if raw is None else raw["seed"],
            budget_seconds=seconds,
            status="EXCEPTION",
            failure=True,
            return_valid=False,
            stage=stage,
            error=str(exc),
            traceback=traceback.format_exc(),
            history=record.get("history", []),
        )
    record["complete_seconds"] = perf_counter() - started
    record["over_budget_seconds"] = max(0.0, record["complete_seconds"] - seconds)
    record["source_hashes"] = hashes
    record["source_unchanged"] = hashes == source_hashes()
    return clean(record)
