# Installation

The core package supports Python 3.10 or later and depends on NumPy and SciPy.
From a source checkout or unpacked source distribution:

```console
python -m pip install .
python examples/qgrn_gate_control.py
```

For development, use `python -m pip install -e ".[dev]"`.
For a built wheel, pass its filename to `python -m pip install`.
No Julia, MATLAB, GPU or quantum hardware is required for the core example.

`requirements-reproducibility.txt` records the numerical versions used for
the research comparisons under Python 3.13. Generic package installation uses
the compatible version ranges in `pyproject.toml` instead.

The full nine-problem reproduction bundle has additional dependencies and
fixed Julia source environments for native Ipopt/MadNLP and Juqbox models.
Its own README documents setup, small-budget checks, exact inputs, execution,
plotting and verification. Plotting saved data needs only the analysis
dependencies; it does not initialize native physical solvers.
