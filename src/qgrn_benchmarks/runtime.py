"""Explicit, relocatable configuration for native Julia backends."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def reproduction_root():
    return Path(
        os.environ.get(
            "QGRN_REPRODUCTION_ROOT",
            Path(__file__).resolve().parents[2] / "reproduction",
        )
    ).resolve()


def native_root():
    return reproduction_root() / "julia"


def native_environment(kind):
    env = dict(os.environ)
    env.update(JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    depot = os.environ.get("QGRN_JULIA_2023_DEPOT" if kind == "pico2023" else "QGRN_JULIA_DEPOT")
    if depot:
        env["JULIA_DEPOT_PATH"] = depot
    return env


def native_command(kind, script=None, *args):
    variable = "QGRN_JULIA_2023" if kind == "pico2023" else "QGRN_JULIA"
    executable = os.environ.get(variable) or shutil.which("julia")
    if not executable:
        raise RuntimeError(
            f"Set {variable} to Julia {'1.9.2' if kind == 'pico2023' else '1.12.7'} executable"
        )
    project = (
        native_root() / "third_party/QuantumCollocation2023.jl"
        if kind == "pico2023"
        else native_root()
    )
    command = [str(executable), "--startup-file=no", f"--project={project}"]
    if script:
        command.extend([str(native_root() / script), *map(str, args)])
    return command
