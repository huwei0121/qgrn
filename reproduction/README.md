# Reproduce the nine Q-GRN benchmarks

This bundle contains nine complete quantum-control problems, Q-GRN and their
comparison methods, 90 saved paired initializations, and 200 recorded objective
curves. Each problem uses ten random seeds, with the same saved initialization
for every method at each seed. Seeds can differ between problems. The file
[configs/selected.json](configs/selected.json) specifies the
methods, seeds, time budgets, and stopping rules.

Saved arrays define the initial points exactly. Input and curve provenance,
source record identifiers, and SHA-256 checksums are recorded in
[data/provenance.json](data/provenance.json).

## Install, verify, and redraw

Run these commands from the repository root, which contains `pyproject.toml`:

```console
python -m pip install -e ".[plots]"
python reproduction/reproduce.py verify
python reproduction/reproduce.py plot --output outputs/reference
```

Verification and plotting require only Python. The tested reference environment
uses 64-bit Python 3.13.6, NumPy 2.3.2, SciPy 1.17.1, and Matplotlib 3.10.9;
[requirements-reproducibility.txt](../requirements-reproducibility.txt) records
the package versions. The core solver supports Python 3.10 and later. Timing
and floating-point iteration histories depend on the platform and numerical
libraries.

Plotting recomputes medians and interquartile bands from the 200 individual
records. The reference display preserves all 150 samples per record, including
increases in the objective. Stable final plateaus may be shortened in the
display; full-budget data remain in `plotted_values.json`, along with each
displayed interval. See [the plotting convention](../docs/figures.md).

## Run the comparisons

```console
python reproduction/reproduce.py run --plan
python reproduction/reproduce.py run --config reproduction/configs/smoke.json --output outputs/smoke --workers 2
python reproduction/reproduce.py run --output outputs/full --workers 2
python reproduction/reproduce.py plot --input outputs/full --output outputs/full_figures
```

The complete plan has twenty problem/method combinations and ten starts per
combination: 200 solves. The smoke configuration uses short budgets to check
the interfaces; it is not a performance comparison.

To run one paired subset:

```console
python reproduction/reproduce.py run --problems jandura_cz --methods qgrn bfgs --seeds 85308 --output outputs/cz
```

Seeds must belong to the configured set for the requested problem.
`--seconds` overrides the time budget without changing the model or constraints.
`--resume` skips successful records under the same recorded plan. Existing run
records are not overwritten; move failed records and their associated files
to a separate directory before retrying them.

The X, CNOT, and SWAP13 trajectory models use the standalone Q-GRN core. Five
endpoint models use an independent BFGS implementation; the C4 tasks also use
a Python translation of the author's DYNAMO Krotov method. Both risk-neutral
methods use the same native Juqbox propagation kernel. The full objectives,
constraints, dimensions, and sources are specified in [models.md](docs/models.md).

## Install the native Julia backends

The reproduction bundle includes five pinned author libraries, with their
licenses, under `julia/third_party`. Commit identifiers are recorded in
[julia/sources.json](julia/sources.json). The native environments have separate
`Project.toml` and `Manifest.toml` files.

Set these environment variables to executable paths on your machine:

- `QGRN_JULIA`: Julia 1.12.7, used for Piccolo/MadNLP and Juqbox/Ipopt.
- `QGRN_JULIA_2023`: Julia 1.9.2, used for the original PICO SWAP13/Ipopt model.
- Optional `QGRN_JULIA_DEPOT` and `QGRN_JULIA_2023_DEPOT`: separate dependency
  caches for the two runtimes. If omitted, Julia uses its default cache.

Then instantiate the pinned environments:

```console
python reproduction/reproduce.py setup --runtimes modern pico2023
```

The first setup downloads dependencies. Compilation and warmup can take longer
than the optimization budget and are recorded separately. Eight tasks use
complete online time; the Juqbox risk-neutral task uses solver time and records
startup, warmup, and return verification separately. Numerical libraries run
with one thread per worker. `--workers` controls parallel independent solves;
the default is one. Native collocation solves can require substantial memory.

## Comparison methods and stopping

- **MadNLP**: native Piccolo X/CNOT models, exact derivatives, MUMPS, and the
  author's barrier update.
- **Ipopt, SWAP13**: native PICO model, sparse Jacobian and exact Hessian.
  Both methods use the same correction to the author's terminal loss and its
  derivatives for the real/imaginary state layout.
- **Ipopt, risk-neutral SWAP02**: native Juqbox model, discrete-adjoint gradient,
  limited-memory Hessian with history length five, and the author's
  five-term Neumann approximation in Störmer-Verlet propagation.
- **BFGS**: independent Python implementation of the published mathematical
  method, with fminunc-compatible scaling and an independent strong-Wolfe
  search. Reported times are Python times.
- **Krotov**: Python translation of DYNAMO's sequential update method, retaining
  the full time grid and using the mathematically consistent absolute-trace
  gradient.

Objective-target, gradient-tolerance, and step-tolerance stopping are disabled
for the time-budget comparisons. Iteration limits, numerical stagnation, and
solver failures remain explicit termination conditions. Improvements recorded
after the budget are retained in raw data and excluded from within-budget curves.

## Files and provenance

| Path | Contents |
| --- | --- |
| `data/inputs` | Ten complete saved initializations per problem |
| `data/curves` | 200 time-indexed candidate curves and reconstruction diagnostics |
| `data/plotted_values.json` | Reference statistical arrays and panel definitions |
| `figures` | Nine-panel figure and three grouped figures |
| `julia` | Native interfaces, locked environments, and author sources |
| `../src/qgrn` | Standalone nonlinear solver |
| `../src/qgrn_benchmarks` | Models, baselines, experiment runner, and analysis |

The public reproduction archive contains the code, 90 input files, and 200
curve records needed to verify the reference data, redraw the figures, and
run the comparisons. It operates independently of the original research
directory. Complete development-run archives remain local and are not required
by these commands.

Q-GRN and the benchmark interfaces use the root license. The DYNAMO translation
uses LGPL-3.0-only, and each Julia library retains its own license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
