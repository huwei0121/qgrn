"""Offline, common feasible control reconstruction for PICO/Piccolo records.

This does not call an optimizer or choose the best historical point. It keeps
terminal derivative auxiliaries, rebuilds the original chains, uniformly scales
only if needed for the original acceleration bound, and propagates the original
discrete model. The original NLP state objective remains a separate quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def array(value):
    if isinstance(value, dict):
        return np.asarray(value["values"], float).reshape(
            value["shape"], order=value.get("order", "F")
        )
    return np.asarray(value, float)


def complex_array(value):
    return (np.asarray(value["real"]) + 1j * np.asarray(value["imag"])).reshape(
        value["shape"], order="F"
    )


def slice_controls(sample):
    return {key: array(sample[key]) for key in ("u", "du", "ddu", "dt")}


def history_controls(raw, event, modern_problem=None):
    if "slice" in event:
        return slice_controls(event["slice"])
    if "primal" in event:
        from .piccolo import primal_slice

        return slice_controls(primal_slice(raw["model"], event["primal"]))
    if raw["task"] == "pico_swap13":
        from .pico import build_problem

        value = build_problem(raw).unpack(np.asarray(event.get("y", event.get("x")), float))
        return dict(
            u=value.u,
            du=value.du,
            ddu=value.ddu,
            dt=np.full(raw["model"]["K"], value.h),
        )
    if modern_problem is None:
        from .piccolo import build_problem

        modern_problem = build_problem(raw)
    value = modern_problem.unpack(np.asarray(event["y"], float))
    return dict(u=value.u, du=value.du, ddu=value.ddu, dt=np.full(raw["model"]["K"], value.h))


def norm(value):
    return float(np.max(np.abs(value), initial=0.0))


def reconstruct_controls(model, coordinates):
    original = {key: np.asarray(coordinates[key], float) for key in ("u", "du", "ddu", "dt")}
    if not all(np.all(np.isfinite(v)) for v in original.values()):
        raise ValueError("Nonfinite controls cannot be reconstructed")
    u = original["u"].copy()
    n, nodes = u.shape
    if nodes != model["K"] or original["du"].shape != u.shape or original["ddu"].shape != u.shape:
        raise ValueError("Incomplete control/derivative arrays")
    amplitude = np.broadcast_to(np.asarray(model["amplitude"], float).reshape(-1), (n,))[:, None]
    acceleration = float(model["acceleration"])
    bounds = model.get("h_bounds", [0.0, None])
    lower, upper = bounds[0] or 0.0, bounds[1] if bounds[1] is not None else np.inf
    # Use all node time-step entries, including the regularized terminal node.
    proposed_h = float(np.mean(original["dt"]))
    h = float(np.clip(proposed_h, lower, upper))
    u = np.clip(u, -amplitude, amplitude)
    u[:, [0, -1]] = 0.0
    if h == 0:
        # On the h=0 feasible set, controls are identically zero, du is
        # constant, and bounded ddu is arbitrary; this branch is its identity.
        u.fill(0.0)
        du = np.repeat(original["du"][:, -1:], nodes, axis=1)
        ddu = np.clip(original["ddu"], -acceleration, acceleration)
        before, scale = norm(ddu), 1.0
    else:
        du = original["du"].copy()
        du[:, :-1] = np.diff(u, axis=1) / h
        ddu = original["ddu"].copy()
        ddu[:, :-1] = np.diff(du, axis=1) / h
        before = norm(ddu)
        scale = 1.0 if before <= acceleration else float(np.nextafter(acceleration / before, 0.0))
        # The common scale preserves both chains and keeps all terminal
        # auxiliary variables; it is exactly one on a feasible input.
        u *= scale
        du *= scale
        ddu *= scale
    reconstructed = dict(u=u, du=du, ddu=ddu, dt=np.full(nodes, h))
    checks = dict(
        control_chain=norm(np.diff(u, axis=1) - h * du[:, :-1]),
        derivative_chain=norm(np.diff(du, axis=1) - h * ddu[:, :-1]),
        endpoints=max(norm(u[:, 0]), norm(u[:, -1])),
        amplitude=max(0.0, float(np.max(np.abs(u) - amplitude))),
        acceleration=max(0.0, norm(ddu) - acceleration),
        time_bounds=max(0.0, lower - h, h - upper),
        equal_time=0.0,
    )
    details = dict(
        input_step_mean=proposed_h,
        reconstructed_step=h,
        common_scale=scale,
        acceleration_after_chain_rebuild=before,
        acceleration_bound=acceleration,
        chain_rebuild_without_scaling_hardware_feasible=before <= acceleration,
        terminal_auxiliary_policy="Keep original terminal du/ddu, then apply the same common scale as other nodes",
        max_coordinate_change={
            key: norm(reconstructed[key] - original[key]) for key in reconstructed
        },
        constraints=checks,
        control_feasibility=max(checks.values()),
    )
    return reconstructed, details


def evaluate(model, controls):
    """Original full objective on the reconstructed original discrete graph."""
    H0, G = complex_array(model["H0"]), complex_array(model["target"])
    drives = np.asarray([complex_array(h) for h in model["Hs"]])
    dim, nodes = len(G), model["K"]
    identity = np.eye(dim, dtype=complex)
    U, state_bound, dynamic = identity.copy(), 0.0, 0.0
    for k in range(nodes - 1):
        A = -1j * controls["dt"][k] * (H0 + np.tensordot(controls["u"][:, k], drives, axes=(0, 0)))
        if model.get("propagation") == "pade22":
            D, N = 12 * identity - 6 * A + A @ A, 12 * identity + 6 * A + A @ A
            next_U = np.linalg.solve(D, N @ U)
            dynamic = max(dynamic, norm((D @ next_U - N @ U) / 12))
        else:
            next_U = expm(A) @ U
        U = next_U
        state_bound = max(state_bound, norm(U.real) - 1.0, norm(U.imag) - 1.0)
    trace = abs(np.trace(G.conj().T @ U))
    if model["loss"] == "unsquared":
        loss = abs(1 - trace / dim)
    elif model["loss"] == "process":
        loss = abs(1 - trace * trace / (dim * dim))
    elif model["loss"] == "average":
        product = G.conj().T @ U
        loss = abs(
            1 - (abs(np.trace(product.conj().T @ product)) + trace * trace) / (dim * (dim + 1))
        )
    else:
        raise ValueError("Unknown original terminal loss")
    squared = sum(np.sum(controls[key] ** 2, axis=0) for key in ("u", "du", "ddu"))
    regularizer = (
        0.5
        * model["R"]
        * float(
            np.sum(squared if model.get("regularizer") == "node_sum" else controls["dt"] * squared)
        )
    )
    return dict(
        objective=float(model["Q"] * loss + regularizer),
        terminal_loss=float(loss),
        regularizer=regularizer,
        quantum_dynamics=dynamic,
        state_bounds=max(0.0, state_bound),
        terminal_unitarity=norm(U.conj().T @ U - identity),
        propagation=model.get("propagation", "exact"),
        objective_semantics="Full original objective on the commonly reconstructed feasible original discrete trajectory",
    )
