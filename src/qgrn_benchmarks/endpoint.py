"""Paired full-grid endpoint comparisons with complete online clocks."""

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from qgrn.compat import GRNOptions, solve_grn

from . import dynamo
from .io import source_hashes, write
from .models import build_task

LABELS = {
    "grn": "Q-GRN",
    "fminunc_bfgs": "BFGS",
    "dynamo_fminunc": "BFGS",
    "dynamo_krotov": "Krotov",
}


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    write(path, value)


def run_case(task_id, seed, output, *, method, seconds=30.0, input_json=None, grn_options=None):
    from .bfgs import FminuncOptions, minimize

    if method not in LABELS:
        raise ValueError(f"Unsupported endpoint method: {method}")
    output = Path(output).resolve()
    assets = Path(str(output) + ".assets")
    if output.exists() or assets.exists():
        raise FileExistsError(output)
    assets.mkdir(parents=True)
    hashes = source_hashes()
    raw = dict(
        schema="quantum_30s_five_run_v1",
        task=task_id,
        seed=seed,
        method=method,
        method_label=LABELS[method],
        budget_seconds=seconds,
        target_objective=None,
        stop_on_target=False,
        source_hashes=hashes,
        assets=str(assets),
        failures=[],
        precision_stopping=False,
        time_budget_only=True,
        gradient_mode="mathematical_abs_trace" if method.startswith("dynamo_") else "analytic",
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        created_utc=datetime.now(timezone.utc).isoformat(),
        timing_semantics="Measured build, initial independent rollout, solver, returned rollout. Imports/source serialization excluded. Checkpoints after budget_seconds are retained but excluded from within-budget comparison.",
    )
    try:
        t0 = perf_counter()
        task = build_task(task_id)
        if input_json is None:
            x0 = task.initial(seed)
        else:
            data = json.loads(Path(input_json).read_text(encoding="utf-8-sig"))
            if data["task"] != task_id or data["seed"] != seed:
                raise ValueError("Paired input task/seed differs")
            x0 = np.asarray(data["coordinates"], dtype=float)
        initial = task.rollout(x0)
        setup = perf_counter() - t0
        raw.update(
            model=task.metadata(),
            initial_coordinates=x0,
            initial=initial,
            setup_seconds=setup,
            initial_coordinate_sha256=hashlib.sha256(
                np.asarray(x0, dtype="<f8").tobytes()
            ).hexdigest(),
        )
        save_new(assets / "input.json", dict(task=task_id, seed=seed, coordinates=x0))
        start = perf_counter()
        remaining = max(0.0, seconds - setup)
        if remaining <= 0:
            result = dict(
                status="SETUP_TIME_LIMIT",
                x=x0,
                history=[],
                objective=initial["final_objective"],
            )
        elif method == "grn":
            from dataclasses import asdict

            overrides = dict(estimator="ratio", ratio_update="step_aware")
            overrides.update(grn_options or {})
            options = GRNOptions(
                **dict(overrides, seconds=remaining, maxiter=100000, stop_on_target=False)
            )
            raw["options"] = asdict(options)
            result = solve_grn(task, x0, options, started=start)
        elif method == "fminunc_bfgs":
            result = minimize(
                task.oracle,
                x0,
                options=FminuncOptions(
                    max_seconds=remaining,
                    max_iterations=100000,
                    max_evaluations=10000000,
                    tol_fun=1e-8,
                    tol_x=1e-8,
                    time_budget_only=True,
                ),
                started=start,
            )
            result = vars(result).copy()
            result["status_code"] = result["status"]
            result["status"] = "FMINUNC_COMPATIBLE_" + result["message"].upper().replace(" ", "_")
        else:
            oc = dynamo.configure(
                task.H0,
                task.Hs,
                task.target,
                task.T,
                initial["controls"].T,
                gradient_mode="mathematical_abs_trace",
            )
            configure = perf_counter() - start
            result = dynamo.optimize(
                oc,
                method="bfgs" if method == "dynamo_fminunc" else "krotov",
                maxiter=100000 if method == "dynamo_fminunc" else 10000000,
                seconds=max(0.0, remaining - configure),
                goal=None,
                history_mode="delta",
                bfgs_backend="fminunc_compatible",
                precision_stopping=False,
            )
            result["x"] = result["controls"].T.ravel(order="F")
            for event in result["history"]:
                if "controls" in event:
                    event["x"] = event.pop("controls").T.ravel(order="F")
                event["elapsed"] += configure
        solver_seconds = perf_counter() - start
        events = []
        for event in result.pop("history", []):
            event = dict(event)
            event["solver_elapsed"] = event["elapsed"]
            event["elapsed"] += setup
            if "x" not in event and "y" in event:
                event["x"] = event["y"]
            event.pop("y", None)
            events.append(event)
        t = perf_counter()
        returned = task.rollout(result["x"])
        return_seconds = perf_counter() - t
        value = result.get("objective", result.get("f", result.get("fun")))
        if value is None and events:
            value = events[-1].get("objective", events[-1].get("f"))
        parity = None if value is None else abs(returned["final_objective"] - value)
        raw.update(
            status=result["status"],
            result=result,
            history=events,
            returned=returned,
            solver_seconds=solver_seconds,
            return_seconds=return_seconds,
            complete_seconds=setup + solver_seconds + return_seconds,
            objective_parity_error=parity,
            return_valid=parity is not None and parity < 1e-8 and returned["unitarity"] < 1e-8,
            failures=result.get("failures", []),
        )
    except Exception as error:
        raw.update(status="EXCEPTION", return_valid=False)
        raw["failures"].append(dict(error=str(error), traceback=traceback.format_exc()))
    raw["source_hashes_after"] = source_hashes()
    raw["source_unchanged"] = hashes == raw["source_hashes_after"]
    raw["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_new(output, raw)
    return raw
