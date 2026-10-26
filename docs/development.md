# Development conventions

The maintained solver lives in `src/qgrn`. The independent reproduction layer
lives in `src/qgrn_benchmarks` and `reproduction`.

Use `snake_case` for modules and functions, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Name modules after their responsibility.
Use names that describe behavior. Standard mathematical symbols such as `x`,
`r`, `mu` and `rho` are appropriate when their definitions are clear.

Public API changes must update examples, documentation and affected tests.
Maintain explicit boundaries between physical modeling, direction computation,
parameter adaptation, budget accounting, results, baseline adapters and plotting.
Document changes to numerical behavior separately from interface refactoring.

## Verification

```console
python -m pytest
python -m ruff check src/qgrn tests/test_qgrn_solver.py examples/qgrn_gate_control.py
python -m build
python tools/check_distribution.py dist
```

Research-checkout equivalence checks compare complete accepted and trial
trajectories with the frozen implementation using a controlled clock.
Standalone tests exercise residual contracts, sparse directions, constraints,
budget accounting, stopping semantics and serialization. Only standalone tests
enter the core source distribution.

Install the wheel into a new environment and run the example and standalone
tests from outside the research checkout. Build checks reject undeclared
historical imports, private machine paths and unrelated archive members.
Numerical comparisons use the pinned recorded dependencies; other supported
versions may change floating-point trajectories.

`docs/ci/tests.yml` provides a GitHub Actions template for Python 3.10 and 3.13
on Linux and Python 3.13 on Windows. To enable it, copy it into
`.github/workflows/tests.yml` using repository credentials with workflow scope.

## Data and figures

Each of the nine benchmark problems is tested with ten random seeds. Algorithms
within a problem share the same initial conditions. Exact seeds, input hashes,
and data provenance are included with the benchmark configuration and records.

Plots use consistent method colors and short labels: Q-GRN, BFGS, Krotov,
Ipopt and MadNLP. Their arrays and statistics remain intact when long final
constant tails are visually shortened. Real descent and changing interquartile
bands must remain visible. Plotting existing records does not rerun optimization.

## Releases

The core wheel contains only `qgrn` and its metadata. The source distribution
adds public documentation, a standalone example and relevant tests. The full
reproduction ZIP additionally contains benchmark models, required third-party
code and licenses, native solver environments, ten-seed configurations and data.
Third-party components retain their own licenses and attribution.

Build archives and verify their contents before publishing a tagged release.
Export the public source tree with `tools/export_public_repository.py`; its file
manifest covers the solver, benchmark software, and reproduction data.
