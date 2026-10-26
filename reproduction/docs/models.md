# Nine quantum-control models

The model definitions below specify the original objectives, discretizations,
and constraints. [selected.json](../configs/selected.json) supplies the methods,
paired starts, and experimental time budgets. Inputs are stored in
[../data/inputs](../data/inputs). Source revisions are listed in
[../julia/sources.json](../julia/sources.json).

The first two models come from Piccolo tutorials. SWAP13 uses the 2023 PICO
paper configuration. The remaining models use the cited papers or author
examples. BFGS and Krotov are Python implementations; MadNLP and Ipopt are
called through the native author interfaces described below.

## Notation and propagation

`G` is the target gate; `U_k` contains the full propagator or all computational
input columns. `u`, `v`, and `a` denote controls and their first and second
derivatives. `N` denotes collocation nodes and `M` denotes propagation intervals.

Piccolo uses `HermitianExponentialIntegrator` with exact exponentials for
piecewise-constant controls. PICO SWAP13 uses Padé [2/2]. Juqbox uses the author's
Störmer-Verlet (SV) method and compatible path-leakage quadrature. The Machnes,
Jandura, and Heule models use exact piecewise-constant propagation. Eliminating
states by these recurrences preserves the original discrete dynamical constraints.

## 1. Piccolo X gate: `first_gate`

Source: [Piccolo first_gate.jl, commit 5e8f209](https://github.com/harmoniqs/Piccolo.jl/blob/5e8f209a4a2216a94e3cf1b755dd451ad3403d99/docs/literate/first_gate.jl),
lines approximately 28–125.

The two-dimensional system has two real control channels, 100 nodes, and 99
intervals. In the author's dimensionless units,

```text
H(u) = 0.5 Z + u_1 X + u_2 Y,    G = X.
```

Pauli drive matrices have no additional factor of `1/2` or `2π`. The initial
duration is 10, so `h_0 = 10/99`. The common timestep is an optimization variable.

The full variables `(U,u,v,a,h,t)` comprise 1,600 real values. Q-GRN uses 601
trajectory coordinates with 400 affine chain/endpoint equalities, giving a
201-dimensional tangent space. The objective is

```text
J = 100 ell(U_N) + 0.005 sum_{k=1}^N h_k (||u_k||² + ||v_k||² + ||a_k||²),
ell(U) = abs(1 - |tr(X† U)|²/4).
```

For `k=1,...,N-1`, the constraints are

```text
U_{k+1} = exp(-i h_k H(u_k)) U_k,
u_{k+1} - u_k = h_k v_k,
v_{k+1} - v_k = h_k a_k,
t_{k+1} - t_k = h_k.
```

Initial and endpoint conditions are `U_1=I`, `t_1=0`, and `u_1=u_N=0`.
Interior controls obey `|u|≤1`; all node accelerations obey `|a|≤1`.
Velocities are unbounded. All timesteps are equal, nonnegative, and have no
finite upper bound. Real and imaginary state components at nodes 2 through N
have bounds `[-1,1]`. The final node contributes to the regularizer and bounds.
There is no separate time-minimization term or hard fidelity threshold.

**Baseline:** native MadNLP 0.9.2 with `MumpsSolver`, `SparseKKTSystem`, analytic
quantum derivatives, and the exact Lagrangian Hessian. The control chain uses
ForwardDiff. The experiment changes the time budget, stopping rules, and
observation callbacks. Q-GRN's logarithmic time coordinate covers `h>0`;
independent returned-trajectory checks use the original `h≥0` constraints and
all state-component bounds.

## 2. Piccolo CNOT with zero-order hold: `two_qubit_zoh`

Source: the first stage of [Piccolo two_qubit_gate_validation.jl, commit 5e8f209](https://github.com/harmoniqs/Piccolo.jl/blob/5e8f209a4a2216a94e3cf1b755dd451ad3403d99/docs/literate/two_qubit_gate_validation.jl),
lines approximately 24–149.

Two transmons are each truncated to two levels: `d=4`, four real controls,
200 nodes, and 199 intervals. The author parameters are frequencies `[4.0,4.1]`
GHz, anharmonicities `[0.2,0.2]` GHz, and exchange coupling `0.1` GHz. In the
rotating frame and two-level truncation used by the tutorial,

```text
H0 = 2π·0.1 (a_1 a_2† + a_1† a_2) = π·0.1 (X⊗X + Y⊗Y),
H_drives = 2π [X⊗I, -Y⊗I, I⊗X, -I⊗Y].
```

Time is in ns. The initial duration is 10, with `h_0=10/199`. The target is
`EmbeddedOperator(CX)` on the full four-dimensional space. This tutorial model
uses strong coupling and excludes higher transmon levels.

There are 9,200 full real variables, 2,401 trajectory coordinates, and 1,600
affine chain/endpoint equalities. The objective is the same time-weighted sum
as in problem 1, with a different terminal loss:

```text
B = G† U_N,
ell(U_N) = abs(1 - (abs(tr(B†B)) + |tr(B)|²)/20).
```

On unitary trajectories this equals `(4/5)(1-|tr(G†U_N)|²/16)`. The native
collocation loss retains the norm term for infeasible state iterates.

Dynamics, initial conditions, endpoint conditions, state bounds, and common
time constraints match problem 1. Control bounds are `|u|≤0.1`, acceleration
bounds are `|a|≤1`, and velocities are unbounded. The final node contributes
to the regularizer. There is no separate time term or hard fidelity bound.

**Baseline:** the same native MadNLP configuration as problem 1. The benchmark
uses a 60-second budget with precision stopping disabled; the tutorial's
original `max_iter=150` is recorded as source information.

## 3. PICO three-qubit outer SWAP: `pico_swap13`

Source: [Trowbridge et al., Sec. IV-C](https://arxiv.org/html/2305.03261v3), and
QuantumCollocation.jl commit `d3b33912ebaf7b335f9e9a8619399cbc59ec89b2`,
`examples/three_qubit_swap/cold_start.jl`, with `load_trajectory=false`.
The interfaces are `julia/pico_input.jl`, `julia/pico_native.jl`,
`qgrn_benchmarks/pico.py`, and `qgrn_benchmarks/reconstruction.py`.

The system has dimension eight, six control channels, 500 nodes, and 499
intervals. The target exchanges qubits 1 and 3. With two-level lowering
operators `a_j`, the Hamiltonian is

```text
H0 = 2π sum_j (omega_j - 5.12) a_j† a_j
     + 2π·0.005 sum_{(i,j)=(1,2),(2,3)} (a_i† a_j + a_i a_j†),
omega = [5.18, 5.12, 5.06] GHz,
H_drives = [a_j+a_j†]_{j=1}^3 followed by [i(a_j-a_j†)]_{j=1}^3.
```

The full model has 73,500 real variables. Q-GRN uses 9,001 scaled trajectory
coordinates `(u,hv,h²a,log(h))` and returns all trajectory nodes. The common
step starts at `0.4 ns` and obeys `0.2≤h≤0.4`. The objective is

```text
J = 200 abs(1 - |tr(G†U_N)|/8)
    + 0.0005 sum_{k=1}^N (||u_k||² + ||v_k||² + ||a_k||²).
```

This is a node-sum regularizer, without timestep weighting. Set
`A_k=-i h_k H(u_k)`, `D_k=I-A_k/2+A_k²/12`, and
`P_k=I+A_k/2+A_k²/12`. The constraints are

```text
D_k U_{k+1} - P_k U_k = 0,    U_1=I,
u_{k+1}-u_k = h_k v_k,       v_{k+1}-v_k = h_k a_k,
u_1=u_N=0,                  |u|≤0.08π,    |a|≤0.05.
```

Velocities have no extra bounds. All node timesteps are equal and satisfy the
stated interval. Unitarity follows from the feasible Padé dynamics.

**Baseline:** native Ipopt, identified in raw records as
`author_ipopt_physical_loss_corrected`, with the author's sparse Jacobian and
exact Lagrangian Hessian. Both methods use a common correction to the terminal
loss and its derivatives for the author's real/imaginary layout. The historical
layout-dependent loss is retained separately in original records.

## 4. Four-qubit C4 gate, T=12: `machnes_c4_t12`

Source: [Machnes et al.](https://arxiv.org/pdf/1011.4874v3), Eqs. (42)–(49),
Table IV row 14, and Table V. Model definitions are in
`qgrn_benchmarks/endpoint_models.json` and `qgrn_benchmarks/models.py`.

There are 16 Hilbert-space dimensions, 128 intervals, 129 state nodes, eight
local X/Y drives, and 1,024 independent real amplitudes. The coupling is one;
time is measured in inverse-coupling units, with `h=12/128`.

```text
H0 = (1/2) sum_{1≤j<l≤4} Z_j Z_l,
H(u) = H0 + (1/2) sum_j (u_xj X_j + u_yj Y_j),
H_CS = (Z_1Z_2 + Z_2Z_3 + Z_3Z_4 + Z_4Z_1)/2,
G = exp(-i π H_CS/2),
J = 1 - |tr(G†U_M)|/16.
```

The drift couples all six pairs; the target Hamiltonian has four ring edges.
The full gate starts from `U_0=I` and follows exact exponential propagation.
Controls are unbounded, with no running cost, endpoint restrictions,
derivative chain, or hardware box constraints.

**Baselines:** BFGS and Krotov through the DYNAMO v1.2 Python translation in
`qgrn_benchmarks/dynamo.py`. BFGS uses the independent fminunc-compatible
implementation. Krotov retains sequential time-slice updates. Both use the
mathematically consistent absolute-trace gradient; the original PSU wrapper
convention remains separately available. Source attribution and LGPL licensing
are in `DYNAMO_TRANSLATION.txt` and `DYNAMO_LICENSE.txt`.

## 5. Rydberg CZ at fixed duration: `jandura_cz`

Source: [Jandura and Pupillo](https://arxiv.org/html/2202.00903v2), Eqs. (5),
(7), (8), (16)–(18), Sec. 3.1, and Fig. 1.

The model has `n=2` qubits, logical dimension `d=4`, 99 intervals, duration
`T=7.612`, and `Omega_max=1`. It assumes infinite blockade and no decay.
The global drive has constant amplitude and phase `phi_k`. The block associated
with a computational string containing `m` ones is

```text
H_m(phi) = sqrt(m)/2 · [[0, exp(i phi)], [exp(-i phi), 0]],
psi_m(0) = (1,0)^T,    m=1,...,n.
```

Each of the two complete 2×2 blocks follows exact piecewise-constant
propagation. The zero computational string is dark. Let `a_m` denote the first
component of the terminal block state and `theta` the common local-Z correction:

```text
s = 1 + sum_{m=1}^{n-1} C(n,m) exp(-i m theta) a_m
      - exp(-i n theta) a_n,
P = 1 + sum_{m=1}^n C(n,m) |a_m|²,
J = 1 - (|s|² + P)/(d(d+1)).
```

`P` retains computational-subspace survival and hence leakage. The 99 phases
and `theta` give 100 physical variables; removing the common laser-phase
symmetry gives 99 coordinates. The model has constant-amplitude phase controls,
the stated block dynamics and initial states, and no additional phase bounds,
bandwidth constraints, or endpoint restrictions.

**Baseline:** analytic GRAPE gradient with an independent Python BFGS method
using fminunc-compatible conventions. The author released pulse arrays;
optimizer-library and line-search details are unspecified. This task is one
fixed-duration subproblem of the paper's minimum-time scan.

## 6. Rydberg C2Z at fixed duration: `jandura_c2z`

Source: the same paper, Sec. 3.3 and Fig. 2.

Here `n=3`, `d=8`, and all three 2×2 symmetric blocks are propagated. There are
399 intervals, `T=16.43`, and `h=T/399`. The objective is the average-infidelity
expression in problem 5, with multiplicities `C(3,m)`. The 399 phases and
`theta` give 400 physical variables and 399 coordinates after phase reduction.
Assumptions, constraints, and baseline match problem 5.

The benchmark uses the printed 399-interval, `T=16.43` configuration. The
public pulse CSV has 398 intervals and duration `16.4295`; it defines a
separate discretized problem.

## 7. Four-qubit C4 gate, T=7: `machnes_c4_t7`

Source: Machnes et al., Table IV row 13. This uses the complete model and
comparison methods from problem 4 with `T=7` and `h=7/128`. The 128-interval
grid, target, Hamiltonians, and 1,024 variables are unchanged.

## 8. Alternating end-controlled CNOT: `heule_cnot3_alternating_t70`

Source: [Heule et al.](https://arxiv.org/html/1007.2572v2), Eqs. (2), (3),
(5)–(7), Sec. III, and Fig. 6.

The three-spin chain has dimension eight, 70 intervals, 71 state nodes, and
total duration 70. Each interval has duration one; the paper's figure caption
uses `T` for this interval duration. Coupling is one and `S=Pauli/2`.

```text
H0 = (1/4) sum_{j=1,2} (X_jX_{j+1} + Y_jY_{j+1} + Z_jZ_{j+1}),
H_k = H0 + u_k X_1/2 for odd intervals,
H_k = H0 + u_k Y_1/2 for even intervals,
G = I_2⊗CNOT,    J = 1 - |tr(G†U_M)|/8.
```

There are 70 real amplitudes and full 8×8 propagation from `U_0=I`. Controls
have no additional amplitude, derivative, or zero-endpoint constraints. The
paper's post-optimization filtering is outside this optimization model.

**Baseline:** independent Python BFGS with fminunc-compatible conventions,
reconstructing the paper's BFGS method. The author's complete executable and
line-search settings are unavailable in the recorded source material.

## 9. Risk-neutral SWAP02: `juqbox_risk_neutral_swap02`

Source: [Juqbox Risk_Neutral examples, commit 3e6279d](https://github.com/LLNL/Juqbox.jl/tree/3e6279d550afc214f5eccf99ba837cfc1d46b836/examples/Risk_Neutral),
combining `run_all.jl` with `swap-02-risk-neutral.jl`.
Related paper: [arXiv:2106.14310](https://arxiv.org/pdf/2106.14310v2),
DOI [10.1137/21M1429618](https://doi.org/10.1137/21M1429618).

The system has four physical levels and three computational input columns.
The target exchanges levels 0 and 2 and preserves level 1; level 3 is the
protected level. `U_0` contains the first three identity columns and
`V_target=U_0[:,(3,2,1)]` in one-based indexing. The duration is 300 ns, with
7,937 SV steps, 7,938 nodes, and nine normalized Gauss–Legendre scenarios
sharing the same control.

```text
n = diag(0,1,2,3),
H0 = -π·0.2198 (n²-n),
H_epsilon - H0 = 2π diag(0, epsilon/100, epsilon/10, epsilon),
epsilon in [-0.01,0.01] GHz.
```

The rotating frequency is 4.10336 GHz. The control uses the author's B-spline
carrier expansion, with angular frequencies 0 and `-2π·0.2198`. There are
12 coefficients per envelope, 48 real coefficients in total, and 32 free
variables after `zero_start_end!` fixes endpoint coefficients. Every free
coefficient obeys `|alpha_i|≤2π·0.012/2 rad/ns`. These are coefficient bounds.
The protected-state weight is `W=diag(0,0,0,1)`.

```text
J(alpha) = sum_q w_q (J_gate,q + J_leak,q) + 0.01 ||alpha||²/48,
J_gate,q = 1 - |tr(V_target† U_q(T))|²/9.
```

The path-leakage term uses the author's SV-compatible quadrature. For real
endpoint components `R_k`, imaginary half-step components `V_{k+1/2}`, and
real diagonal `W`,

```text
J_leak = (h/(2T)) sum_k [tr(R_k^T W R_k) + tr(R_{k+1}^T W R_{k+1})
                                  + 2 tr(V_{k+1/2}^T W V_{k+1/2})].
```

All scenarios, steps, and input columns contribute. SV is not exactly unitary;
the native gate loss includes its column-norm defect. The Tikhonov coefficient
0.01 is the pinned author's `objparams` default.

Constraints are the complete SV dynamics, shared initial state, coefficient
box, and fixed endpoint coefficients. State elimination uses that same
recurrence, including the author's five-term Neumann implicit solve. There
is no additional hard fidelity or leakage upper bound.

**Baseline:** native `setup_ipopt_problem`/`IpoptSolve`, discrete-adjoint
gradient, and limited-memory Hessian with history length five. The benchmark
replaces precision stopping with a time budget. Q-GRN uses full endpoint and
path-leakage Gauss-Newton sensitivities; the timings record their cost separately
from the direction solve.
