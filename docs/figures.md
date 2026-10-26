# Figure conventions

The nine-panel figure and three grouped figures use the same data, method
colors, and axis conventions. Each panel shows its problem title, axis labels,
and method legend.

| Method | Label | Color |
| --- | --- | --- |
| Structured gradient regularization | Q-GRN | #bd284d |
| BFGS comparison implementation | BFGS | #1f77b4 |
| Native Ipopt | Ipopt | #e7952a |
| Native MadNLP | MadNLP | #329b75 |
| DYNAMO Krotov translation | Krotov | #8b6bb1 |

## Curves and time

Each problem uses ten paired random seeds. Lines show the median objective
and shading shows the 25th to 75th percentiles. Each record has 150 time
samples; a sample uses the latest candidate available at that time.
Objective increases and floating-point values are retained.

For the three collocation problems, the candidate is restored to the original
control-chain and hardware constraints before evaluating the full objective.
Other problems use their reduced physical objective. The risk-neutral problem
uses its recorded solver clock; the others use complete online time.

The horizontal scale is symmetric logarithmic and includes zero. Its linear
threshold is the largest power of ten no greater than budget / 1000.
The vertical scale is logarithmic when all visible values exceed 1e-12;
otherwise its interval from -1e-12 to 1e-12 is linear.

## Stable tails

A display interval can end before the full budget when the median and both
quartiles have stopped changing. The stored data and statistics retain the
complete budget.

1. The median must first decrease by more than
   max(1e-12, 0.01 * max(abs(initial), abs(final))).
2. All three quantile curves must be constant in the tail. A difference of
   at most 1e-14 is allowed only when both values are in the sub-1e-12
   linear region.
3. The displayed tail retains one quarter of its transformed horizontal span
   and at least 7% of the total axis span. A tail shorter than 12% of the
   total span is displayed in full.

The interval for each method is saved in the displayed_intervals field of
plotted_values.json.

## Implementation

The plotting entry point is redraw_saved_values in
src/qgrn_benchmarks/plotting.py. It reads saved records and does not evaluate
the physical models. The independent plot validator checks cohort sizes,
time grids, quantiles, tail stability, method colors, and SVG labels.

~~~console
python reproduction/reproduce.py plot --output outputs/reference
~~~
