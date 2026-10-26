# Method and model sources

- A. Trowbridge et al., *Direct Collocation for Quantum Optimal Control*,
  IEEE QCE (2023). <https://arxiv.org/abs/2305.03261>.
- K. Mishchenko, *Regularized Newton Method with Global O(1/k²) Convergence*,
  SIAM Journal on Optimization 33(3), 1440-1462 (2023).
  <https://doi.org/10.1137/22M1488752>.
- S. Machnes et al., *Comparing, Optimising and Benchmarking Quantum Control
  Algorithms in a Unifying Programming Framework*, Physical Review A 84,
  022305 (2011). <https://arxiv.org/abs/1011.4874>.
- S. Jandura and G. Pupillo, *Time-Optimal Two- and Three-Qubit Gates for
  Rydberg Atoms*, Quantum 6, 712 (2022). <https://arxiv.org/abs/2202.00903>.
- R. Heule et al., *Local quantum control of Heisenberg spin chains*,
  Physical Review A 82, 052333 (2010). <https://arxiv.org/abs/1007.2572>.

The reproduction package separately records original paper tasks, later
software examples, Python translations and local corrections. The `BFGS` label
identifies an independent implementation with documented fminunc-compatible
settings. DYNAMO attribution and license texts accompany the translated code.

Ipopt and MadNLP are separate implementations in the primal-dual interior-point
family. MadNLP describes its method as closely related to Ipopt; they are shown
with different colors. See [MadNLP documentation](https://madnlp.github.io/MadNLP.jl/stable/)
and [Ipopt documentation](https://coin-or.github.io/Ipopt/).

Q-GRN applies gradient-dependent regularization to structured Gauss-Newton
models of quantum-control trajectories.
