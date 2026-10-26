# Q-GRN algorithm

Q-GRN (**Quantum Gradient-Regularized Newton**) solves smooth quantum-control
problems with available model derivatives. It combines a structured
Gauss-Newton model, gradient-dependent regularization and adaptive
backtracking feedback. Full discrete trajectories can be represented in
control coordinates by propagating states through the prescribed dynamics.

## Local model and direction

Let `f(x)` be the complete physical objective, `B_beta(x)` the barrier and
`A p = 0` the affine tangent constraints. The adapter supplies the physical
gradient `g`, barrier gradient `gb`, endpoint residual Jacobian `Jr` and a
symmetric positive semidefinite base curvature `H0`. With `P_A` the orthogonal
projector onto `ker(A)`, define

```text
H = 2 Jr.T Jr + H0
chi = ||P_A (g + gb)||_2
mu = max(sqrt(kappa * max(chi, 1e-30)), 1e-12)

minimize    (g + gb).T p + 0.5 p.T (H + mu I) p
subject to A p = 0
           C p <= slack_fraction * slack    [optional hardware constraints]
```

The initial scale uses average model curvature:

```text
c0 = max(trace(H) / max(1, min(number_of_residuals, number_of_coordinates)), 1e-9)
mu0 = initial_regularization_ratio * c0
kappa0 = mu0**2 / max(chi, 1e-30)
```

Physical running costs and barrier curvature form `H0`. The endpoint residual
contributes low-rank curvature. A sparse KKT factorization handles the base
matrix and affine constraints; a Woodbury correction accounts for the endpoint
term. The implementation applies inverse actions through solves and keeps the
null-space basis implicit. For a pure residual objective without base curvature
or affine constraints, an endpoint Gram solve reduces cancellation. A complete
physical gradient with additional terms uses the general KKT/Woodbury path.

## Hardware constraints

For inequalities `c(x) <= 0`, the slack model uses `slack = -c(x) > 0` and
`C = Dc(x)`. A primal feasible working-set method starts at `p=0`, reuses the
base inverse action, and solves dense Schur systems for the working set.
Work limits may return an inexact step; the solver requires that step to
satisfy the linearized inequalities and decrease the quadratic model. QP
stationarity, dual, complementarity and feasibility residuals are recorded.

The optional `face` model adds nearby hardware normals to the affine tangent
space and releases rows with negative model multipliers. For this model,
`chi` uses the enlarged tangent projector. Both direction models rely on the
physical oracle to check nonlinear feasibility during backtracking.

## Acceptance and regularization feedback

For a trial displacement `s = alpha * p`, the physical curvature factor
`Jreg` describes supplementary running costs. Define

```text
barrier_change = B_beta(x) - B_beta(x + s)
pred = -g.T s - ||Jr s||**2 - 0.5 ||Jreg s||**2 + barrier_change
actual = f(x) - f(x + s) + barrier_change
rho = actual / pred
```

The trial is accepted when it is physically valid, `pred > 0`, `actual > 0`
and `rho >= 0.1`. Artificial damping is excluded from the prediction. Complete
residual differences and logarithmic slack ratios preserve small improvements
when objective values contain large constant contributions.

The default accepted-step update is

```text
kappa_next = max(1e-24, kappa * clip(1 / (2 * alpha**2), 1/4, 16))
```

A full accepted step halves `kappa`; an accepted half step doubles it.
If backtracking encountered an invalid physical-domain query, the update uses
reduction-ratio feedback: divide `kappa` by four for `rho > 0.75`, multiply it
by four for `rho < 0.25`, and otherwise retain it. This separates step lengths
limited by physical feasibility from those limited by model accuracy.

An invalid or inaccurate direction multiplies `kappa` by 100. A rejected
backtracking search multiplies it by 16. At most 30 direction attempts and
25 backtracking evaluations per attempt are made for one accepted step.
Every attempted direction and search outcome is included in the diagnostics.

## Barrier continuation and optional estimators

The default barrier schedule reduces `beta` by a factor of ten when an accepted
step has linear decrease below `0.1 * beta`, or after 30 accepted steps at the
same parameter. The `adaptive` schedule requires at least three accepted steps
and either a fourfold reduction in the affine-projected gradient norm or
balance between the physical and barrier gradients. Derivatives are refreshed
after each parameter change.

The optional `residual` and `split` estimators measure the Gauss-Newton error
using a complete physical residual `F`, with `f = 0.5 * ||F||**2`. For
`e = F(x+s) - F(x) - DF(x)s`, the model error is

```text
E = <F, e> + <DF s, e> + 0.5 ||e||**2
```

The `residual` estimator scales `||e|| / ||s||**2`. The `split` estimator
separates the first term from the remaining terms and uses
`mu = delta + sqrt(kappa * chi)`, subject to the same numerical floors.
Accepted observations are combined with decayed previous estimates.
`remainder_feedback` selects unsigned, underprediction-only or relative-error
observations. These estimators are selected explicitly; the default estimator
uses reduction and backtracking feedback. Exact option names and adapter
contracts are documented in [the API reference](api.md).

## Scope and interpretation

Causal state elimination preserves a discrete trajectory model when its
dynamics uniquely determine the states from the controls. This permits
structured directions in control coordinates while retaining the modeled
objective and constraints. Discretization accuracy remains a property of the
chosen physical model and integration rule.

The method uses a Gauss-Newton curvature approximation. Its empirical behavior
and conditional geometric identities are developed in the research report.
Mishchenko's gradient-regularized Newton method motivates the scale choice;
its convergence rates require the assumptions of that analysis. This
constrained, nonconvex implementation is assessed through numerical experiments
and the reported local-model diagnostics.
