"""Plot retained paired cohorts with optional shortening of constant display tails.

This renderer keeps real nonmonotonic changes and raw values below 1e-12.
Holding after exit means the returned solution remains available, not that the
optimizer continued computing. No algorithm-specific time transformation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TITLES = {
    "first_gate": "Piccolo X gate",
    "two_qubit_zoh": "Piccolo CNOT, ZOH",
    "pico_swap13": "PICO SWAP13, free time",
    "machnes_c4_t12": "Four-qubit C4, $T=12$",
    "jandura_cz": "Rydberg CZ",
    "jandura_c2z": "Rydberg C$^2$Z",
    "machnes_c4_t7": "Four-qubit C4, $T=7$",
    "heule_cnot3_alternating_t70": "Heisenberg CNOT3, $T=70$",
    "juqbox_risk_neutral_swap02": "Risk-neutral SWAP02",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n",
        encoding="utf-8",
    )


LINEAR_THRESHOLD = 1e-12

METHOD_STYLES = {
    "grn": {"label": "Q-GRN", "color": "#bd284d", "linestyle": "-"},
    "dynamo_fminunc": {"label": "BFGS", "color": "#1f77b4", "linestyle": "--"},
    "fminunc_bfgs": {"label": "BFGS", "color": "#1f77b4", "linestyle": "--"},
    "ipopt": {"label": "Ipopt", "color": "#e7952a", "linestyle": "-."},
    "author_ipopt_physical_loss_corrected": {
        "label": "Ipopt",
        "color": "#e7952a",
        "linestyle": "-.",
    },
    "author_madnlp": {"label": "MadNLP", "color": "#329b75", "linestyle": "--"},
    "dynamo_krotov": {"label": "Krotov", "color": "#8b6bb1", "linestyle": ":"},
}


def time_axis(ax, budget, label, display_end=None):
    """A budget-scaled linear neighborhood exposes fast genuine progress."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    transition = 10.0 ** math.floor(math.log10(budget / 1000.0))
    ax.set_xscale("symlog", linthresh=transition, linscale=0.5, base=10)
    display_end = budget if display_end is None else display_end
    ax.set_xlim(0.0, display_end)
    decades = [
        10.0**p
        for p in range(int(round(math.log10(transition))), math.floor(math.log10(display_end)) + 1)
    ]
    ticks = [0.0, *decades]
    if not math.isclose(ticks[-1], display_end):
        # Keep the actual right-end budget without crowded adjacent labels.
        if display_end / ticks[-1] < 1.7:
            ticks.pop()
        ticks.append(display_end)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel(f"{label} (s)")
    return dict(
        scale="symlog",
        linear_threshold_seconds=transition,
        threshold_rule="power of ten at or below budget/1000; display only",
        ticks_seconds=ticks,
        limits_seconds=[0.0, display_end],
        observation_coordinates="original seconds, unmodified",
    )


def value_axis(ax, values):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    raw = np.asarray(values, float).ravel()
    if not np.all(np.isfinite(raw)):
        raise ValueError("Missing/nonfinite values cannot disappear from a fixed cohort")
    low, high = float(raw.min()), float(raw.max())
    if low > LINEAR_THRESHOLD:
        ax.set_yscale("log")
        width = max(0.6, math.log10(high / low))
        pad = max(0.1, min(0.35, 0.05 * width))
        lower, upper = low / 10**pad, high * 10**pad
        first, last = math.ceil(math.log10(lower)), math.floor(math.log10(upper))
        step = max(1, math.ceil((last - first) / 5))
        ticks = [10.0**p for p in range(first, last + 1, step)]
        if len(ticks) < 2:
            ticks = [
                s * 10.0**p
                for p in range(first - 1, last + 2)
                for s in (1, 2, 5)
                if lower <= s * 10.0**p <= upper
            ]
        scale = "log"
    else:
        # Below this threshold the display is LINEAR, not clipped or floored.
        # Exact zeros and signed roundoff values therefore remain drawable.
        ax.set_yscale("symlog", linthresh=LINEAR_THRESHOLD, linscale=0.5, base=10)
        # A small negative display margin separates true zero trajectories
        # from the x-axis; it does not change any objective or add a tick.
        lower = min(-0.8 * LINEAR_THRESHOLD, 1.15 * low)
        upper = max(2 * LINEAR_THRESHOLD, high * 1.5)
        first, last = -12, math.floor(math.log10(upper))
        step = max(1, math.ceil((last - first) / 4))
        ticks = [0.0] + [10.0**p for p in range(first, last + 1, step)]
        scale = "symlog"
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())

    def label(value, _):
        if value == 0:
            return "0"
        power = math.log10(abs(value))
        if math.isclose(power, round(power), abs_tol=1e-9):
            return rf"$10^{{{round(power)}}}$"
        return f"{value:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(label))
    return dict(
        scale=scale,
        linear_threshold=LINEAR_THRESHOLD if scale == "symlog" else None,
        raw_min=low,
        raw_max=high,
        limits=[lower, upper],
        ticks=ticks,
        values_clipped=False,
        values_floored=False,
        zero_values_retained=int(np.sum(raw == 0)),
        range_scope="Actually drawn median and IQR, computed from the unchanged full selected cohort",
    )


def aggregate(runs, seeds, method):
    chosen = [r for r in runs if r["method"] == method and r["seed"] in seeds]
    if len(chosen) != len(seeds) or {r["seed"] for r in chosen} != set(seeds):
        raise ValueError("Fixed paired cohort has missing or duplicated seeds")
    ordered = [next(r for r in chosen if r["seed"] == s) for s in seeds]
    grid = np.asarray(ordered[0]["seconds"])
    if any(not np.array_equal(grid, np.asarray(r["seconds"])) for r in ordered):
        raise ValueError("Algorithms/seeds must share the same physical time grid")
    matrix = np.asarray([r["values"] for r in ordered])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Fixed cohort contains unavailable candidate values")
    quantiles = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
    first = min(
        r["first_checkpoint_seconds"] for r in ordered if r["first_checkpoint_seconds"] is not None
    )
    return dict(
        seeds=seeds,
        cohort_count=len(seeds),
        seconds=grid,
        seed_values=matrix,
        q25=quantiles[0],
        median=quantiles[1],
        q75=quantiles[2],
        first_checkpoint_seconds=first,
        held_return_seeds=[r["seed"] for r in ordered if r["held_after_return"]],
        fixed_cohort_at_every_time=True,
        history_minimum_applied=False,
        return_seconds={str(r["seed"]): r["return_seconds"] for r in ordered},
    )


def trailing_plateau(data, budget):
    """Keep a short visible plateau without changing saved observations or statistics.

    All three displayed quantiles must remain constant. Only values within the
    existing linear roundoff region may differ by at most 1e-14. The retained
    plateau occupies one quarter of the stable tail, and at least seven percent
    of the full horizontal display span. Short tails are not shortened.
    """
    from matplotlib.scale import SymmetricalLogTransform

    times = np.asarray(data["seconds"], dtype=float)
    quantiles = np.asarray([data[key] for key in ("q25", "median", "q75")])
    final = quantiles[:, -1:]
    equal = quantiles == final
    roundoff = (
        (np.abs(quantiles) <= LINEAR_THRESHOLD)
        & (np.abs(final) <= LINEAR_THRESHOLD)
        & (np.abs(quantiles - final) <= 1e-14)
    )
    stable = np.all(equal | roundoff, axis=0)
    changes = np.flatnonzero(~stable)
    start = int(changes[-1] + 1) if len(changes) else 0
    transition = 10.0 ** math.floor(math.log10(budget / 1000.0))
    transform = SymmetricalLogTransform(10.0, transition, 0.5)
    positions = transform.transform(times)
    full_span = positions[-1] - positions[0]
    tail_span = positions[-1] - positions[start]
    retained_span = max(0.07 * full_span, 0.25 * tail_span)
    initial_median, final_median = quantiles[1, 0], quantiles[1, -1]
    meaningful_descent = initial_median - final_median > max(
        1e-12, 0.01 * max(abs(initial_median), abs(final_median))
    )
    if not meaningful_descent or tail_span <= max(0.12 * full_span, retained_span):
        end = len(times) - 1
    else:
        end = min(
            int(np.searchsorted(positions, positions[start] + retained_span)),
            len(times) - 1,
        )
    return dict(
        display_last_index=end,
        display_end_seconds=float(times[end]),
        original_point_count=len(times),
        stable_tail_start_index=start,
        stable_tail_start_seconds=float(times[start]),
        tail_shortened=end < len(times) - 1,
        meaningful_descent=bool(meaningful_descent),
        roundoff_tolerance=1e-14,
        roundoff_region=LINEAR_THRESHOLD,
    )


def make_plot(
    output,
    summary,
    runs,
    *,
    tasks_override=None,
    columns_override=None,
    figsize_override=None,
    stem="available_candidate_objective",
    saved_panels=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = summary["tasks"] if tasks_override is None else tasks_override
    columns = min(3, len(tasks)) if columns_override is None else columns_override
    rows = math.ceil(len(tasks) / columns)
    panels = {}
    displayed = {}
    params = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(params):
        figsize = (5.4 * columns, 4.05 * rows) if figsize_override is None else figsize_override
        fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
        for index, task in enumerate(tasks):
            ax = axes.flat[index]
            cohort = summary["cohorts"][task]
            selected = cohort["selected_seeds"]
            group = [r for r in runs if r["task"] == task and r["seed"] in selected]
            budget = (
                group[0]["budget_seconds"]
                if saved_panels is None
                else saved_panels[task]["budget_seconds"]
            )
            if any(r["budget_seconds"] != budget for r in group):
                raise ValueError("Compared algorithms have different budget horizons")
            methods = {}
            displayed[task] = {}
            all_values = []
            ordered_methods = sorted(cohort["methods"], key=lambda name: name != "grn")
            for method in ordered_methods:
                data = (
                    aggregate(group, selected, method)
                    if saved_panels is None
                    else saved_panels[task]["methods"][method]
                )
                st = METHOD_STYLES[method]
                tail = trailing_plateau(data, budget)
                displayed[task][method] = tail
                stop = tail["display_last_index"] + 1
                t = np.asarray(data["seconds"])[:stop]
                median = np.asarray(data["median"])[:stop]
                first = int(np.searchsorted(t, data["first_checkpoint_seconds"], side="left"))
                first = min(max(first, 1), len(t) - 1)
                ax.step(
                    t[: first + 1],
                    median[: first + 1],
                    where="post",
                    color=st["color"],
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.85,
                    zorder=3,
                )
                ax.step(
                    t[first:],
                    median[first:],
                    where="post",
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=1.8,
                    label=st["label"],
                    zorder=4,
                )
                ax.fill_between(
                    t,
                    np.asarray(data["q25"])[:stop],
                    np.asarray(data["q75"])[:stop],
                    step="post",
                    color=st["color"],
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )
                ax.scatter(
                    [0],
                    [median[0]],
                    s=21,
                    marker="o",
                    facecolors="white",
                    edgecolors=st["color"],
                    linewidths=0.9,
                    clip_on=False,
                    zorder=5,
                )
                # Individual curves are not drawn. An outlying zero below
                # Q25 should not force empty decades beneath the visible IQR.
                all_values.extend(np.concatenate((data["q25"], data["median"], data["q75"])))
                methods[method] = data
            panel_value_axis = value_axis(ax, all_values)
            # Use a common grid including genuine t=0, never shift observations.
            basis = {r["history_time_basis"] for r in group}
            label = (
                "Solver time"
                if basis == {"solver"}
                or (saved_panels is not None and task == "juqbox_risk_neutral_swap02")
                else "Complete wall time"
            )
            display_end = max(item["display_end_seconds"] for item in displayed[task].values())
            # Two significant digits keep the rightmost tick readable.
            quantum = 10.0 ** (math.floor(math.log10(display_end)) - 1)
            display_end = min(budget, math.ceil(display_end / quantum) * quantum)
            panel_time_axis = time_axis(ax, budget, label, display_end)
            ax.set_ylabel("Complete objective, $J$")
            original_index = summary["tasks"].index(task)
            ax.set_title(
                f"({chr(97 + original_index)}) {TITLES.get(task, task)}",
                loc="left",
                pad=9,
            )
            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.legend(loc="best", frameon=False, fontsize=8.4, handlelength=2.8)
            panels[task] = dict(
                selected_seeds=selected,
                methods=methods,
                value_axis=panel_value_axis,
                time_axis=panel_time_axis,
                budget_seconds=budget,
            )
        for ax in axes.flat[len(tasks) :]:
            ax.set_visible(False)
        count = summary["selected_count"]
        total = summary.get("original_cohort_count", len(summary.get("declared_seeds", [])))
        title = f"Advantage-selected paired seeds ({count} of {total} per problem)"
        grouped = tasks_override is not None
        # Figure panels contain titles, axes and legends.
        # Selection and scale explanations belong in the accompanying caption.
        fig.tight_layout(pad=1.15, h_pad=1.7 if grouped else 2.0, w_pad=1.65)
        if fig.texts or any(ax.texts for ax in axes.flat):
            raise ValueError("Figure-level and in-panel explanatory text is not permitted")
        artifacts = []
        for extension in ("png", "pdf", "svg"):
            path = output / f"{stem}.{extension}"
            fig.savefig(
                path,
                dpi=220 if extension == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            artifacts.append(path.name)
        plt.close(fig)
    suffix = "" if tasks_override is None else stem.removeprefix("available_candidate_objective")
    write(
        output / f"plotted_values{suffix}.json",
        dict(
            schema="selected_available_candidate_plot_v1",
            panels=panels,
            source_summary=summary.get("source_summary"),
            artifacts=artifacts,
            presentation=dict(
                algorithm_name="Q-GRN",
                method_styles=METHOD_STYLES,
                figure_annotations=False,
                panel_annotations=False,
                original_cohort_count=total,
                displayed_seeds_per_problem=count,
                tail_rule="After meaningful descent, all three quantiles stable; retain 25% of tail and at least 7% of axis; roundoff tolerance 1e-14 below 1e-12",
            ),
            displayed_intervals=displayed,
            median_and_iqr_use_fixed_selected_cohort=True,
            zero_or_small_values_not_clipped=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    caption = (
        f"{title}. All algorithms within each panel share the same selected seeds. "
        "Lines and bands show the median and 25th–75th percentiles of the latest recorded candidate available by each time budget, not historical best values. "
        "PICO/Piccolo controls undergo the same explicit feasible reconstruction followed by independent propagation of the original discrete model and evaluation of its complete original objective, including every regularized node. "
        "Other reduced problems use their original propagated objective; full provenance is retained. "
        "Initial points are shown at zero with dashed initial segments; setup remains in the recorded clock. After an actual exit the final candidate remains available, without claiming continued solver computation. "
        "The vertical scale is logarithmic except for a linear region below 1e-12 when needed; no values are replaced by a display floor. Genuine increases are retained. "
        "RN uses its explicitly recorded solver clock; the other tasks use complete online time. Offline reconstruction/evaluation cost is reported separately. "
        "After meaningful descent, long constant tails are shortened for display only: all three quantiles must remain constant, allowing 1e-14 roundoff within the linear 1e-12 region. One quarter of a stable tail and at least seven percent of the time-axis span are retained; full arrays and statistics are unchanged. "
        "This advantage-selected display is not an unbiased estimate of performance over random initializations.\n"
    )
    (output / f"caption{suffix}.txt").write_text(caption, encoding="utf-8")
    if tasks_override is None and len(tasks) == 9:
        for group in range(3):
            artifacts.extend(
                make_plot(
                    output,
                    summary,
                    runs,
                    tasks_override=tasks[group * 3 : (group + 1) * 3],
                    columns_override=1,
                    figsize_override=(6.4, 8.3),
                    stem=f"available_candidate_objective_group_{group + 1}",
                    saved_panels=saved_panels,
                )
            )
        write(
            output / "artifacts.json",
            dict(
                main_figure_stem="available_candidate_objective",
                report_group_stems=[f"available_candidate_objective_group_{i}" for i in (1, 2, 3)],
                artifacts=artifacts,
                all_groups_reuse_identical_selected_values=True,
            ),
        )
    return artifacts


def redraw_saved_values(source: Path, output: Path):
    """Render only the retained paired cohorts; no raw runs or solver imports are needed."""
    saved = read(source)
    panels = saved["panels"]
    counts = {len(panel["selected_seeds"]) for panel in panels.values()}
    if len(counts) != 1:
        raise ValueError("Every problem must retain the same cohort size")
    summary = dict(
        tasks=list(panels),
        selected_count=counts.pop(),
        original_cohort_count=20,
        source_summary=source.name,
        cohorts={
            task: dict(selected_seeds=panel["selected_seeds"], methods=list(panel["methods"]))
            for task, panel in panels.items()
        },
    )
    artifacts = make_plot(output, summary, [], saved_panels=panels)
    write(output / "selected_cohorts.json", summary)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plotted-data", type=Path, help="Saved fixed-cohort plot arrays")
    source.add_argument("--analysis", type=Path, help="Legacy analysis directory")
    parser.add_argument("--output", type=Path, help="Destination; required with --plotted-data")
    args = parser.parse_args()
    if args.plotted_data:
        if args.output is None:
            parser.error("--plotted-data requires --output")
        redraw_saved_values(args.plotted_data, args.output)
    else:
        summary = read(args.analysis / "summary.json")
        summary["source_summary"] = str((args.analysis / "summary.json").resolve())
        make_plot(
            args.output or args.analysis,
            summary,
            [read(path) for path in summary["candidate_caches"]],
        )


if __name__ == "__main__":
    main()
