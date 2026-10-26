# Python API

```python
from qgrn import Options, ResidualProblem, minimize
```

`minimize(problem, x0, options=None, *, started=None)` returns an
`OptimizationResult`. `Options` and `Result` are aliases for `SolverOptions`
and `OptimizationResult`. The input `x0` is a finite, nonempty, one-dimensional
real vector. Complex quantum states, residuals and derivatives must be split
into real and imaginary components by the model adapter.

## Residual problems

`ResidualProblem(residual, jacobian)` minimizes `sum(residual(x)**2)`.
The residual has shape `(m,)` and its dense Jacobian has shape `(m, n)`, where
`n` is the number of optimization coordinates. Both callables describe the same
model. The solver uses the exact residual gradient and Gauss-Newton curvature.

```python
import numpy as np
from qgrn import Options, ResidualProblem, minimize

problem = ResidualProblem(
    residual=lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]]),
    jacobian=lambda x: np.array([[-20 * x[0], 10], [-1, 0]]),
)
result = minimize(problem, [-1.2, 1.0], Options(time_limit=5, max_iterations=100))
print(result.fun, result.status, result.nfev)
```

An application may instead implement `ResidualOracle.residual_oracle(x, order)`.
For `order=0`, return `f` and `r`; for `order=1`, also return `Jr` and
`g = 2 * Jr.T @ r`. Return fields in a mapping or an attribute-bearing object.
An out-of-domain query returns `valid=False` and a descriptive `reason`.

## Structured trajectory problems

`ConstrainedOracle.grn_oracle(x, *, derivatives, beta)` supplies a complete
physical objective, derivatives, constraint information and an optional
barrier at the requested parameter `beta`. If both oracle methods are present,
`minimize` uses `grn_oracle`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `f` | scalar | Complete physical objective, excluding the barrier |
| `r` | `(m,)` | Residual defining the endpoint curvature model |
| `g` | `(n,)` | Complete physical gradient, including running or linear costs |
| `Jr` | `(m, n)` | Dense residual Jacobian or endpoint curvature factor |
| `barrier` | scalar | Barrier value at the requested `beta`; default zero |
| `gb` | `(n,)` | Barrier gradient; default zero |
| `H0` | `(n, n)` | Symmetric positive semidefinite supplementary curvature; default zero |
| `Jreg` | `(q, n)` | Factor for supplementary physical curvature; default empty |
| `gate_error` | scalar | Application error used by optional target stopping; default `f` |

Value queries require `f` and `r`. Derivative queries additionally require `g`
and `Jr`. `H0` and `Jreg` accept dense or SciPy sparse matrices. The direction
model is `2 * Jr.T @ Jr + H0`, where `H0` includes supplementary physical and
barrier curvature. The physical prediction uses
`2 * Jr.T @ Jr + Jreg.T @ Jreg`; the barrier change is evaluated directly.
`Jreg` therefore contains physical cost curvature only. The solver adds its
own regularization separately. A linear physical term contributes to `f` and
`g` and requires no curvature factor.

The following adapter members provide geometry and application diagnostics:

| Member | Contract |
| --- | --- |
| `A` | Full-row-rank matrix of shape `(k, n)` defining tangent constraints `A @ p = 0` |
| `restore_initial(x)` | Return restored coordinates, optionally as `(coordinates, diagnostics)` |
| `project_tangent(v)` | Apply the orthogonal projector onto `ker(A)`; optional |
| `feasibility(x)` | Return a nonnegative violation measure, or an object/mapping with a `primal` field |
| `hardware_geometry(x)` | Return an object with `slack`, `jacobian` and `ids` |
| `return_fields(x, final_oracle)` | Return additional application fields without replacing solver fields |

By default, initialization projects onto `A @ x = 0`. For constraints
`A @ x = b` with nonzero `b`, supply `restore_initial` and `feasibility`.
`A` must contain independent rows even when a custom projector is supplied.

For hardware inequalities `c(x) <= 0`, `hardware_geometry` returns positive
slacks `-c(x)`, the Jacobian `Dc(x)`, and one stable identifier per inequality.
The initial point must be strictly interior for a slack-constrained direction.
Every trial oracle checks the nonlinear physical domain. State propagation,
control restoration and physical constraints belong to the model adapter.
The generic `ResidualProblem` supplies an unconstrained residual model.

### Complete residuals and stable differences

The optional fields `remainder_residual=F` and `remainder_jacobian=DF` represent
the complete physical objective as `f = 0.5 * F.T @ F`. The Jacobian may be a
dense/sparse matrix, a linear operator, or a callable applying `DF` to a step.
These fields support stable objective differences and are required by the
`residual` and `split` curvature estimators for constrained adapters. Their
gradient and Gauss-Newton curvature must agree with the physical prediction.
Barrier terms remain separate.

For a logarithmic barrier of normalized box coordinates `z` and positive
time-bound slacks `s`, an optional `barrier_difference_data` mapping contains
`box`, `time_slacks` and `weight`. It represents
`-weight * (sum(log(1 - z**2)) + sum(log(s)))`. Supplying this representation
allows stable differences between nearby iterates using `log1p`.

## Options and stopping

| Option | Default | Meaning |
| --- | --- | --- |
| `time_limit` | 60 | Wall-clock budget in seconds |
| `max_iterations` | 3000 | Maximum accepted steps |
| `curvature_estimator` | `ratio` | Scale estimator: `ratio`, `residual` or `split` |
| `scale_update` | `step_aware` | Backtracking feedback for `ratio`; alternatively `ratio` |
| `initial_regularization_ratio` | 0.01 | Initial shift relative to average model curvature |
| `stop_on_target` | false | Enable objective, gate-error and feasibility stopping |
| `objective_target` | none | With target stopping, defaults to `1e-3 * problem.objective_weight`, whose default weight is 1 |
| `gate_error_target` | 0.001 | Gate-error threshold when target stopping is enabled |
| `feasibility_tolerance` | 1e-6 | Primal-violation threshold for target stopping |
| `direction_tolerance` | 1e-6 | Relative direction residual and linearized-feasibility tolerance |
| `use_hardware_constraints` | false | Use the adapter's hardware geometry in the direction model |
| `hardware_model` | `slack` | Linearized slack inequalities; alternatively `face` tangent constraints |
| `slack_fraction` | 0.9 | Fraction of available hardware slack used by the direction |
| `max_active_constraints` | 32 | Maximum hardware rows in the working set |
| `hardware_qp_updates` | 64 | Maximum working-set updates per slack QP |
| `active_margin` | 0.01 | Slack threshold for selecting nearby tangent faces |
| `barrier_initial` | 0.001 | Initial barrier parameter |
| `barrier_minimum` | 1e-9 | Lower bound for the barrier parameter |
| `barrier_update` | `decrement` | Linear-decrease/stage-length schedule; alternatively `adaptive` |
| `remainder_feedback` | `underprediction` | Eligible remainder observations: `underprediction`, `all` or `relative` |
| `estimator_decay` | 0.5 | Decay of the previous remainder scale after an accepted step |
| `objective_scale` | none | Residual-estimator normalization, inferred at initialization by default |
| `step_scale` | 1 | Coordinate scale for residual-estimator normalization |

The default uses time, iteration and numerical-progress stopping. Target
stopping is enabled explicitly and checks all three application thresholds.
These thresholds measure solution quality; they do not certify KKT stationarity
or global optimality.

Restoration, oracle calls, rejected attempts and the final physical evaluation
count toward the wall-clock budget. An operation already running is completed,
and its actual duration is recorded. Initialization and the final evaluation
also run with a zero budget. Pass a `time.perf_counter()` timestamp as `started`
to include caller setup in the same budget.

## Results and diagnostics

`x` is the last accepted coordinate vector. `fun` is its complete physical
objective evaluated with zero barrier. `nit` counts accepted steps; `nfev`
counts value and derivative oracle calls, including failed queries; `njev`
counts derivative calls. `elapsed` is the total measured duration.

| Status | Meaning |
| --- | --- |
| `TARGET_REACHED` | All requested application targets passed |
| `TIME_LIMIT` | A wall-clock budget checkpoint was reached |
| `ITERATION_LIMIT` | The maximum accepted-step count was reached |
| `NUMERICAL_STAGNATION` | A direction or repeated objective changes could not be resolved numerically |
| `MODEL_FAILURE` | Direction/model retries were exhausted |
| `NUMERICAL_FAILURE` | A nonfinite model quantity or invalid derivative query prevented progress |
| `NONDIFFERENTIABLE_POINT` | The adapter reported a singular zero-overlap residual chart |

`success` is true only for `TARGET_REACHED`; `message` gives the detailed
termination reason. If the final physical evaluation is invalid, the solver
raises `RuntimeError` because no finite result can be returned.

`history` records the initial point and accepted iterates with cumulative
timestamps. `trials` records each direction attempt and its line-search outcome;
it is not a trace of every backtracking evaluation. Mapping access provides
additional records, including `calls`, `parameters`, `checks`, `restoration`,
`oracle_seconds` and `direction_seconds`.

`to_dict()` returns a JSON-compatible copy and converts nonfinite diagnostics
to `null`. It omits the raw final oracle object by default because that object
may contain sparse matrices or callables. `to_dict(include_oracle=True)` also
serializes the raw payload and raises `TypeError` for unsupported objects.

## Compatibility interface

`qgrn.compat.GRNOptions` and `qgrn.compat.solve_grn` translate the earlier
option and record schema. Examples are `seconds -> time_limit`, `maxiter ->
max_iterations`, `ratio_update -> scale_update` and `hardware_tangent ->
use_hardware_constraints`. The compatibility defaults enable target stopping
and reduction-ratio updates; `SolverOptions` defaults to budget-based stopping
and accepted-step feedback. Specify these options explicitly when migrating a
recorded configuration. Existing dataset identifiers remain unchanged.
