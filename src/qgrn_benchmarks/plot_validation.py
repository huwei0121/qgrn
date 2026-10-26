"""Validate unchanged fixed-cohort publication arrays and display-only tail cuts."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .plotting import METHOD_STYLES, read, write


def validate_saved_plot(output: Path, reference: Path | None = None) -> dict:
    """Check saved plot statistics without importing solvers or discarded seeds."""
    output = Path(output)
    plotted = read(output / "plotted_values.json")
    original = read(reference) if reference is not None else None
    checks = []
    for task, panel in plotted["panels"].items():
        for method, values in panel["methods"].items():
            key = f"{task}/{method}"
            matrix = np.asarray(values["seed_values"])
            times = np.asarray(values["seconds"])
            assert matrix.shape == (10, 150), f"{key}: fixed ten-seed cohort changed"
            assert values["seeds"] == panel["selected_seeds"], key
            assert times[0] == 0 and times[-1] == panel["budget_seconds"], key
            assert np.isfinite(matrix).all(), key
            for quantile, field in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
                assert np.array_equal(np.quantile(matrix, quantile, axis=0), values[field]), key
            assert not values["history_minimum_applied"], key
            if original is not None:
                assert values == original["panels"][task]["methods"][method], (
                    f"{key}: data or full-budget statistics changed"
                )
            cut = plotted["displayed_intervals"][task][method]
            end = cut["display_last_index"]
            assert cut["display_end_seconds"] == times[end], key
            if cut["tail_shortened"]:
                assert cut["meaningful_descent"], f"{key}: flat baseline removed"
                for field in ("q25", "median", "q75"):
                    tail = np.asarray(values[field])[end:]
                    same = tail == tail[-1]
                    roundoff = (
                        (np.abs(tail) <= 1e-12)
                        & (abs(tail[-1]) <= 1e-12)
                        & (np.abs(tail - tail[-1]) <= 1e-14)
                    )
                    assert np.all(same | roundoff), f"{key}: changing tail omitted"
            checks.append(
                dict(
                    task=task,
                    method=method,
                    full_array_preserved=True,
                    tail_shortened=cut["tail_shortened"],
                    display_end_seconds=cut["display_end_seconds"],
                )
            )
    tasks = list(plotted["panels"])
    for index in range(3):
        group = read(output / f"plotted_values_group_{index + 1}.json")
        assert list(group["panels"]) == tasks[index * 3 : (index + 1) * 3]
        for task, panel in group["panels"].items():
            assert panel == plotted["panels"][task], task
            assert group["displayed_intervals"][task] == plotted["displayed_intervals"][task], task
    labels = {entry["label"]: entry["color"] for entry in METHOD_STYLES.values()}
    assert set(labels) == {"Q-GRN", "BFGS", "Ipopt", "MadNLP", "Krotov"}
    assert len(set(labels.values())) == 5
    for path in output.glob("available_candidate_objective*.svg"):
        tree = ET.parse(path)
        text = " ".join(
            "".join(item.itertext()) for item in tree.findall(".//{http://www.w3.org/2000/svg}text")
        )
        assert "Q-GRN" in text
        for phrase in (
            "Advantage-selected",
            "Native",
            "Linear below",
            "Latest available",
            "Median and",
        ):
            assert phrase not in text, f"{path.name}: explanatory annotation remains"
    result = dict(
        valid=True,
        displayed_runs=200,
        panels=9,
        algorithm_groups=len(checks),
        exact_reference_match=original is not None,
        full_budget_arrays_preserved=True,
        unchanged_seed_selection=True,
        distinct_algorithm_colors=labels,
        zero_new_optimization_runs=True,
        checks=checks,
        plotted_values_sha256=hashlib.sha256(
            (output / "plotted_values.json").read_bytes()
        ).hexdigest(),
    )
    if reference is not None:
        result["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    write(output / "plot_validation.json", result)
    return result
