# Third-party sources

The standalone Q-GRN core is separate from the benchmark comparison code.

- `src/qgrn_benchmarks/dynamo.py` translates DYNAMO v1.2, copyright Shai Machnes 2010, under **LGPL-3.0-only**. `DYNAMO_LICENSE.txt` retains the original LGPL-3.0 and CC-BY-SA-3.0 text; `GPL-3.0.txt` supplies the GNU GPL text incorporated by LGPL-3.0. Both files accompany the translation. Original archive: https://www.tau.ac.il/~quantum/qlib/DYNAMO_v_1_2.zip (SHA-256 `95c3609cad7ae74151a8a1c39b48d61b83afe7cf729cf0d235013ce5640d8da2`). Numerical changes include the explicitly selected mathematical abs-trace gradient and an independently implemented fminunc-compatible BFGS backend. The translated module is distributed as editable Python source.
- `julia/third_party/Piccolo.jl`, `DirectTrajOpt.jl`, `NamedTrajectories.jl`, `QuantumCollocation2023.jl`, and `Juqbox.jl` retain their original source licenses. Exact commit identities and license paths are listed in `julia/sources.json`. The copied source files are unchanged; local Julia adapters live outside those directories.
- The translated BFGS implementation in `src/qgrn_benchmarks/bfgs.py` implements public mathematical mechanisms. It is not a distribution of MATLAB fminunc or its protected line-search implementation.

The Julia environment manifests adjust only local source locations from the research checkout to the bundled `third_party` directory. The original PICO environment remains locked to Julia 1.9.2. Other dependencies are downloaded by Julia's package manager under their own licenses.

The SWAP adapter in `julia/pico_native.jl` corrects the terminal complex-vector layout when constructing the physical trace loss and its derivatives. The same corrected loss is used by both comparison methods. Benchmark time limits and iteration recording are implemented in the local adapters. The bundled author sources retain their original contents.
