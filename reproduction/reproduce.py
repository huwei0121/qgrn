#!/usr/bin/env python3
"""Verify, rerun, and plot the selected nine-problem Q-GRN experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT.parent / "src"))
os.environ["QGRN_REPRODUCTION_ROOT"] = str(ROOT)


def run_entry(entry, output):
    from qgrn_benchmarks.runner import run_one

    task, method, seed, seconds = entry
    return run_one(
        task,
        method,
        seed,
        Path(output) / f"{task['id']}__{method}__{seed}.json",
        seconds=seconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    setup = commands.add_parser("setup")
    setup.add_argument(
        "--runtimes",
        nargs="+",
        choices=["modern", "pico2023"],
        default=["modern", "pico2023"],
    )
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=ROOT / "configs/selected.json")
    run.add_argument("--problems", nargs="+")
    run.add_argument("--methods", nargs="+")
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--seconds", type=float)
    run.add_argument("--output", type=Path, default=ROOT / "outputs/run")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--plan", action="store_true")
    run.add_argument("--resume", action="store_true")
    plot = commands.add_parser("plot")
    plot.add_argument("--input", type=Path)
    plot.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    from qgrn_benchmarks.io import read, write

    if args.command == "verify":
        from qgrn_benchmarks.analysis import verify_reference

        report = verify_reference(ROOT)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    if args.command == "plot":
        from qgrn_benchmarks.analysis import aggregate_reference
        from qgrn_benchmarks.plotting import redraw_saved_values

        args.output.mkdir(parents=True, exist_ok=True)
        values = args.output / "plotted_values.json"
        if args.input:
            from qgrn_benchmarks.analysis import aggregate_runs

            write(values, aggregate_runs(ROOT, args.input, args.output))
        else:
            write(values, aggregate_reference(ROOT))
        redraw_saved_values(values, args.output)
        print(args.output.resolve())
        return 0
    if args.command == "setup":
        from qgrn_benchmarks.runtime import (
            native_command,
            native_environment,
            native_root,
        )

        for kind in args.runtimes:
            command = native_command(kind) + ["-e", "using Pkg; Pkg.instantiate()"]
            subprocess.run(command, cwd=native_root(), env=native_environment(kind), check=True)
        return 0
    config = read(args.config)
    tasks = config["tasks"]
    for requested, available, name in (
        (args.problems, {x["id"] for x in tasks}, "problem"),
        (args.methods, {m for x in tasks for m in x["methods"]}, "method"),
    ):
        if requested and not set(requested) <= available:
            parser.error(f"Unknown {name}")
    entries = [
        (task, method, seed, args.seconds)
        for task in tasks
        if not args.problems or task["id"] in args.problems
        for seed in task["seeds"]
        if not args.seeds or seed in args.seeds
        for method in task["methods"]
        if not args.methods or method in args.methods
    ]
    if not entries:
        parser.error("No matching configured paired starts")
    plan = dict(
        run_count=len(entries),
        nominal_seconds=sum(float(e[0]["seconds"] if e[3] is None else e[3]) for e in entries),
        entries=[
            dict(
                task=e[0]["id"],
                method=e[1],
                seed=e[2],
                seconds=e[0]["seconds"] if e[3] is None else e[3],
            )
            for e in entries
        ],
    )
    source_paths = (
        list((ROOT.parent / "src/qgrn").glob("*.py"))
        + list((ROOT.parent / "src/qgrn_benchmarks").glob("*.py"))
        + list((ROOT / "julia").rglob("*.jl"))
        + list((ROOT / "julia").rglob("*.toml"))
    )
    source_paths += [args.config, ROOT.parent / "src/qgrn_benchmarks/endpoint_models.json"]
    plan["source_sha256"] = {
        str(p.relative_to(ROOT.parent))
        if p.is_relative_to(ROOT.parent)
        else p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_paths)
        if p.is_file()
    }
    paired_inputs = {ROOT / "data/inputs" / f"{e[0]['id']}__{e[2]}.json" for e in entries}
    plan["input_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paired_inputs)
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and read(plan_path) != plan:
        parser.error("Existing output has a different plan")
    write(plan_path, plan)
    if args.resume:
        entries = [
            e for e in entries if not _completed(args.output / f"{e[0]['id']}__{e[1]}__{e[2]}.json")
        ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_entry, e, str(args.output.resolve())): e for e in entries}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                entry = futures[future]
                result = dict(
                    task=entry[0]["id"],
                    method=entry[1],
                    seed=entry[2],
                    error=str(error),
                    return_valid=False,
                )
            results.append(result)
            write(
                args.output / "status.json",
                dict(completed=len(results), total=len(entries), results=results),
            )
            print(json.dumps(result), flush=True)
    return int(any(not x["return_valid"] for x in results))


def _completed(path):
    if not path.exists():
        return False
    from qgrn_benchmarks.io import read

    data = read(path)
    if data.get("return_valid") is not True:
        raise RuntimeError(
            f"Existing invalid run must be inspected and moved before resume: {path}"
        )
    return True


if __name__ == "__main__":
    raise SystemExit(main())
