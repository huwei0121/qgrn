"""Uniform nine-problem dispatch; optional Julia imports stay lazy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from time import perf_counter

from .io import read, write
from .runtime import native_command, native_environment, native_root, reproduction_root

_WARM_RECORDS = {}


def _warm_collocation(problem_id, seed, input_path):
    key = (problem_id, "qgrn")
    if key in _WARM_RECORDS:
        return dict(_WARM_RECORDS[key], reused=True)
    started = perf_counter()
    if problem_id == "pico_swap13":
        from .pico import run_case

        result = run_case(
            input_path,
            seconds=120.0,
            maxiter=1,
            grn_options=dict(
                estimator="ratio",
                ratio_update="step_aware",
                hardware_tangent=True,
                hardware_model="slack",
            ),
        )
    else:
        from .piccolo import run_python

        result = run_python(problem_id, seed, 120.0, input_path, maxiter=1)
    if result.get("return_valid") is not True:
        raise RuntimeError(f"Q-GRN warmup failed for {problem_id}: {result.get('status')}")
    record = dict(
        task=problem_id,
        method="qgrn",
        seed=seed,
        maxiter=1,
        seconds=perf_counter() - started,
        status=result["status"],
        return_valid=True,
        reused=False,
        semantics="Independent one-iteration warmup on a copy of the same retained input; excluded from formal history and budget",
    )
    _WARM_RECORDS[key] = record
    return record


def run_one(task, method, seed, output, *, seconds=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    budget = float(task["seconds"] if seconds is None else seconds)
    if budget <= 0:
        raise ValueError("seconds must be positive")
    if method not in task["methods"]:
        raise ValueError("Method is not configured for this problem")
    problem_id = task["id"]
    input_path = reproduction_root() / "data/inputs" / f"{problem_id}__{seed}.json"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    warm = None
    if method == "qgrn" and problem_id in ("first_gate", "two_qubit_zoh", "pico_swap13"):
        warm = _warm_collocation(problem_id, seed, input_path)
    if problem_id in ("first_gate", "two_qubit_zoh"):
        if method == "qgrn":
            from .piccolo import run_python

            result = run_python(problem_id, seed, budget, input_path)
            result["warm_record"] = warm
            result["warm_seconds"] = 0.0 if warm.get("reused") else warm["seconds"]
            write(output, result)
        else:
            command = native_command(
                "modern",
                "piccolo_native.jl",
                problem_id,
                seed,
                input_path,
                output,
                budget,
            )
            _native(command, output, "modern")
    elif problem_id == "pico_swap13":
        if method == "qgrn":
            from .pico import run_case

            result = run_case(
                input_path,
                seconds=budget,
                maxiter=100000,
                grn_options=dict(
                    estimator="ratio",
                    ratio_update="step_aware",
                    hardware_tangent=True,
                    hardware_model="slack",
                ),
            )
            result["warm_record"] = warm
            result["warm_seconds"] = 0.0 if warm.get("reused") else warm["seconds"]
            write(output, result)
        else:
            _native(
                native_command("pico2023", "pico_run.jl", input_path, output, budget),
                output,
                "pico2023",
            )
    elif problem_id.startswith("juqbox"):
        from .juqbox import run_case

        run_case(
            problem_id,
            seed,
            output,
            "grn" if method == "qgrn" else method,
            seconds=budget,
            input_json=input_path,
        )
    else:
        from .endpoint import run_case

        run_case(
            problem_id,
            seed,
            output,
            method=task["legacy_methods"][method],
            seconds=budget,
            input_json=input_path,
        )
    result = read(output)
    if (
        result.get("return_valid") is not True
        or result.get("failure")
        or result.get("source_unchanged") is False
    ):
        raise RuntimeError(f"Invalid solver return; complete diagnostic retained in {output}")
    return dict(
        task=problem_id,
        method=method,
        seed=seed,
        status=result["status"],
        return_valid=True,
        output=str(output),
    )


def _native(command, output, kind):
    log = output.with_suffix(".native.log")
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=native_root(),
            env=native_environment(kind),
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if result.returncode:
        raise RuntimeError(f"Native Julia returned {result.returncode}; see {log}")
