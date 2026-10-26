"""Fixed-cohort reference statistics and integrity checks."""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np

from .io import read


def aggregate_reference(root):
    root = Path(root)
    config = read(root / "configs/selected.json")
    output = deepcopy(read(root / "data/plotted_values.json"))
    for task in config["tasks"]:
        panel = output["panels"][task["id"]]
        for method in task["methods"]:
            runs = [
                read(root / "data/curves" / f"{task['id']}__{method}__{seed}.json")
                for seed in task["seeds"]
            ]
            values = np.asarray([x["values"] for x in runs])
            times = np.asarray(runs[0]["seconds"])
            if any(not np.array_equal(times, x["seconds"]) for x in runs):
                raise ValueError("Mismatched time grids")
            record_method = task["legacy_methods"][method]
            entry = panel["methods"][record_method]
            entry.update(
                seed_values=values.tolist(),
                q25=np.quantile(values, 0.25, axis=0).tolist(),
                median=np.quantile(values, 0.5, axis=0).tolist(),
                q75=np.quantile(values, 0.75, axis=0).tolist(),
                seconds=times.tolist(),
                seeds=task["seeds"],
                cohort_count=len(runs),
            )
    return output


def verify_reference(root):
    root = Path(root)
    config = read(root / "configs/selected.json")
    errors = []
    runs = 0
    expected = set()
    for task in config["tasks"]:
        if len(task["seeds"]) != 10 or len(set(task["seeds"])) != 10:
            errors.append(f"Invalid seeds: {task['id']}")
        for seed in task["seeds"]:
            hashes = set()
            initial = []
            for method in task["methods"]:
                path = root / "data/curves" / f"{task['id']}__{method}__{seed}.json"
                expected.add(path.name)
                data = read(path)
                runs += 1
                hashes.add(data["initial_coordinate_sha256"])
                initial.append(data["values"][0])
                if data["task"] != task["id"] or data["method"] != method or data["seed"] != seed:
                    errors.append(f"Identity differs: {path.name}")
                t = np.asarray(data["seconds"])
                v = np.asarray(data["values"])
                event = np.asarray(data["source_event_seconds"])
                if (
                    len(t) != 150
                    or len(v) != 150
                    or not np.isfinite(v).all()
                    or np.any(event > t + 1e-9)
                ):
                    errors.append(f"Invalid checkpoint data: {path.name}")
                if t[0] != 0 or abs(t[-1] - task["seconds"]) > 1e-9:
                    errors.append(f"Budget differs: {path.name}")
            if (
                len(hashes) != 1
                or None in hashes
                or np.ptp(initial) > 1e-9 * max(1, max(abs(x) for x in initial))
            ):
                errors.append(f"Pairing differs: {task['id']}/{seed}")
    if {x.name for x in (root / "data/curves").glob("*.json")} != expected:
        errors.append("Unexpected or missing reference records")
    manifest = root / "MANIFEST.json"
    if manifest.exists():
        for relative, digest in read(manifest)["files"].items():
            path = root.parent / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"File checksum differs: {relative}")
    return dict(
        valid=not errors,
        problems=len(config["tasks"]),
        runs=runs,
        paired_starts=90,
        errors=errors,
        cohort="advantage-selected ten of twenty per problem",
    )


def aggregate_runs(root, run_directory, output):
    """Reevaluate a completed rerun on the same causal checkpoint convention."""
    from .checkpoints import analyze_record, display_grid
    from .io import write

    root, run_directory, output = map(Path, (root, run_directory, output))
    plan = read(run_directory / "plan.json")
    reference = read(root / "data/plotted_values.json")
    full_config = read(root / "configs/selected.json")
    configs = {x["id"]: x for x in full_config["tasks"]}
    panels = {}
    for entry in plan["entries"]:
        task, method, seed = entry["task"], entry["method"], entry["seed"]
        path = run_directory / f"{task}__{method}__{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"Incomplete experiment: {path}")
        cache = output / "candidate_cache" / path.name
        record = analyze_record(path, display_grid(entry["seconds"]), cache)
        panel = panels.setdefault(task, dict(runs={}, budget_seconds=entry["seconds"]))
        panel["runs"].setdefault(method, []).append(record)
    result = dict(schema="selected_available_candidate_plot_v1", panels={})
    for task, group in panels.items():
        entry = deepcopy(reference["panels"][task])
        entry["budget_seconds"] = group["budget_seconds"]
        entry["methods"] = {}
        cohorts = [sorted(x["seed"] for x in runs) for runs in group["runs"].values()]
        if any(seeds != cohorts[0] for seeds in cohorts):
            raise ValueError("Missing paired starts in rerun")
        entry["selected_seeds"] = cohorts[0]
        for method, runs in group["runs"].items():
            runs.sort(key=lambda x: cohorts[0].index(x["seed"]))
            values = np.asarray([x["values"] for x in runs])
            record_method = configs[task]["legacy_methods"][method]
            entry["methods"][record_method] = dict(
                seeds=cohorts[0],
                cohort_count=len(runs),
                seconds=runs[0]["seconds"],
                seed_values=values.tolist(),
                q25=np.quantile(values, 0.25, axis=0).tolist(),
                median=np.quantile(values, 0.5, axis=0).tolist(),
                q75=np.quantile(values, 0.75, axis=0).tolist(),
                first_checkpoint_seconds=[x["first_checkpoint_seconds"] for x in runs],
                held_return_seeds=[x["seed"] for x in runs if x["held_after_return"]],
                return_seconds=[x["return_seconds"] for x in runs],
                fixed_cohort_at_every_time=True,
                history_minimum_applied=False,
            )
        result["panels"][task] = entry
    write(output / "plotted_values.json", result)
    return result
