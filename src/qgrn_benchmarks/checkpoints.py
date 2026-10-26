"""Offline latest-candidate curves, common PICO reconstruction, paired selection.

Display-grid samples refer to preceding solver checkpoints. After a solver
exits, its returned candidate remains available. Analysis reads the original
records and saves reconstruction results separately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .io import read, write
from .reconstruction import (
    evaluate,
    history_controls,
    norm,
    reconstruct_controls,
    slice_controls,
)

ROOT = Path(__file__).parent
PICO_TASKS = {"first_gate", "two_qubit_zoh", "pico_swap13"}
EPSILON = 1e-12
EVALUATION_VERSION = "latest_candidate_common_reconstruction_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_grid(budget, count=150):
    if not 2 <= count <= 200:
        raise ValueError("Use 2 through 200 fixed display-grid points")
    tau = budget / 1000.0
    values = tau * np.expm1(np.linspace(0.0, np.log1p(budget / tau), count))
    values[0], values[-1] = 0.0, budget
    return values


def events(raw):
    result = [dict(seconds=0.0, kind="initial", source_index=None)]
    for index, event in enumerate(raw.get("history", [])):
        seconds = float(event["elapsed"])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid original checkpoint clock")
        result.append(dict(seconds=seconds, kind="history", source_index=index))
    # The RN adapter explicitly supplies a solver-clock return timestamp.
    clock = (
        raw.get("returned_elapsed_seconds")
        if raw.get("history_time_basis") == "solver"
        else raw.get("complete_seconds")
    )
    if clock is None:
        raise ValueError("No documented clock for the returned candidate")
    result.append(dict(seconds=float(clock), kind="return", source_index=None))
    if any(b["seconds"] < a["seconds"] for a, b in zip(result, result[1:])):
        raise ValueError("Recorded candidate times are not monotone")
    return result


def evaluate_pico_record(raw, event, problem):
    if event["kind"] in {"initial", "return"}:
        field = "initial_slice" if event["kind"] == "initial" else "final_slice"
        coordinates = slice_controls(raw[field])
    else:
        coordinates = history_controls(raw, raw["history"][event["source_index"]], problem)
    candidate, reconstruction = reconstruct_controls(raw["model"], coordinates)
    value = evaluate(raw["model"], candidate)
    repeated, _ = reconstruct_controls(raw["model"], candidate)
    identity_error = max(norm(repeated[k] - candidate[k]) for k in candidate)
    if (
        max(
            reconstruction["control_feasibility"],
            value["quantum_dynamics"],
            value["state_bounds"],
            identity_error,
        )
        > 1e-9
    ):
        raise ValueError("Common reconstruction did not produce a feasible fixed point")
    return dict(
        **value,
        reconstruction=reconstruction,
        idempotence_error=identity_error,
        objective_provenance="independent_original_discrete_rollout_after_common_reconstruction",
    )


def original_objective(raw, event):
    if event["kind"] == "initial":
        return float(raw["initial"]["final_objective"])
    if event["kind"] == "return":
        return float(raw["returned"]["final_objective"])
    record = raw["history"][event["source_index"]]
    return float(record.get("objective", record.get("f")))


def verify_reduced_domain(raw):
    """No extra recovery is needed for unconstrained reduced paper tasks."""
    if raw["task"].startswith("juqbox"):
        lo, hi = (
            np.asarray(raw["metadata"]["lower"]),
            np.asarray(raw["metadata"]["upper"]),
        )
        points = [
            np.asarray(raw["initial"]["coefficients"]),
            np.asarray(raw["returned"]["coefficients"]),
        ]
        points.extend(np.asarray(p["x"]) for p in raw["history"])
        maximum = max(max(0.0, float(np.max(lo - x)), float(np.max(x - hi))) for x in points)
        return dict(
            original_box_violation=maximum,
            controls_checked=len(points),
            reconstruction="componentwise_projection_to_original_coefficient_box",
        )
    model = raw["model"]
    if model.get("added_constraints") != "none":
        raise ValueError("Unknown reduced control-domain constraints")
    return dict(reconstruction="identity_on_unconstrained_original_coordinates")


def analyze_record(path, grid, cache):
    started = perf_counter()
    digest = sha(path)
    signature = dict(
        raw_sha256=digest,
        evaluation_version=EVALUATION_VERSION,
        reconstruction_sha256=sha(ROOT / "reconstruction.py"),
        grid_seconds=grid.tolist(),
    )
    if path.name.startswith("juqbox"):
        signature["rn_replay_sha256"] = sha(ROOT / "juqbox_replay.py")
    if cache.exists():
        saved = read(cache)
        if saved.get("cache_signature") == signature:
            return saved
    raw = read(path)
    if raw.get("return_valid") is not True or raw.get("failure") is True:
        raise ValueError(f"Invalid producer return: {path}")
    all_events = events(raw)
    times = np.array([p["seconds"] for p in all_events])
    indices = np.searchsorted(times, grid, side="right") - 1
    indices[0] = 0  # t=0 always describes the identical shared original input.
    first_update = next((i for i, p in enumerate(all_events) if p["kind"] == "history"), None)
    wanted = set(indices.tolist()) | {0, len(all_events) - 1}
    if first_update is not None:
        wanted.add(first_update)
    problem = None
    if raw["task"] in PICO_TASKS and raw["method"] == "grn" and raw["task"] != "pico_swap13":
        from .piccolo import build_problem

        problem = build_problem(raw)
    domain = None if raw["task"] in PICO_TASKS else verify_reduced_domain(raw)
    samples = {}
    for index in sorted(wanted):
        event = all_events[index]
        tick = perf_counter()
        if raw["task"] in PICO_TASKS:
            point = evaluate_pico_record(raw, event, problem)
        elif raw["task"].startswith("juqbox"):
            original = (
                raw["initial"]["coefficients"]
                if event["kind"] == "initial"
                else raw["returned"]["coefficients"]
                if event["kind"] == "return"
                else raw["history"][event["source_index"]]["x"]
            )
            original = np.asarray(original, float)
            candidate = np.clip(original, raw["metadata"]["lower"], raw["metadata"]["upper"])
            change = float(np.max(abs(candidate - original)))
            if change:
                from .juqbox_replay import replay

                point = replay(raw["metadata"], candidate)
            else:
                point = dict(
                    objective=original_objective(raw, event),
                    objective_provenance="recorded_original_reduced_objective",
                )
            point["reconstruction"] = dict(
                domain, coefficient_projection_change=change, restored_box_violation=0.0
            )
        else:
            point = dict(
                objective=original_objective(raw, event),
                objective_provenance="recorded_original_reduced_objective",
                reconstruction=domain,
            )
        if not np.isfinite(point["objective"]):
            raise ValueError("Nonfinite candidate quality")
        samples[str(index)] = dict(event, **point, offline_evaluation_seconds=perf_counter() - tick)
    values = np.array([samples[str(i)]["objective"] for i in indices])
    result = dict(
        schema="available_candidate_run_v1",
        cache_signature=signature,
        path=str(path.resolve()),
        task=raw["task"],
        method=raw["method"],
        seed=raw["seed"],
        status=raw["status"],
        initial_coordinate_sha256=raw.get("initial_coordinate_sha256"),
        budget_seconds=float(raw["budget_seconds"]),
        history_time_basis=raw.get("history_time_basis", "complete_online_cost"),
        seconds=grid,
        values=values,
        source_event_indices=indices,
        source_event_seconds=times[indices],
        original_event_count=len(all_events),
        samples=samples,
        independently_replayed=raw["task"] in PICO_TASKS,
        first_checkpoint_seconds=times[first_update] if first_update is not None else None,
        return_seconds=times[-1],
        actual_final_checkpoint_seconds=times[-2] if len(times) > 2 else 0.0,
        held_after_return=bool(times[-1] < grid[-1]),
        held_return_grid_count=int(np.sum(grid > times[-1])),
        raw_objective_history=[dict(p, objective=original_objective(raw, p)) for p in all_events]
        if raw["task"] not in PICO_TASKS
        else [
            dict(
                p,
                objective=(
                    raw["initial"].get("final_objective", raw["initial"].get("native_objective"))
                    if p["kind"] == "initial"
                    else raw["returned"].get("final_objective")
                    if p["kind"] == "return"
                    else raw["history"][p["source_index"]].get("objective")
                ),
            )
            for p in all_events
        ],
        offline_analysis_seconds=perf_counter() - started,
    )
    write(cache, result)
    return read(cache)
