"""Public solver contract, constrained steps and numerical precision behavior."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from qgrn import Options, ResidualProblem, Result, SolverOptions, minimize
from qgrn.direction import solve_direction
from qgrn.geometry import AffineGeometry
from qgrn.oracle import _normalize


def rosenbrock(scale=1.0):
    root = np.sqrt(scale)
    return ResidualProblem(
        lambda x: root * np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]]),
        lambda x: root * np.array([[-20 * x[0], 10.0], [-1.0, 0.0]]),
    )


class UpperBox:
    def grn_oracle(self, x, *, derivatives=True, beta=0.0):
        slack = 1 - x
        if np.any(slack <= 0):
            return dict(valid=False, reason="hardware_bounds")
        r = (x - 2) / np.sqrt(2)
        return dict(
            f=float(r @ r),
            r=r,
            Jr=np.eye(len(x)) / np.sqrt(2),
            g=x - 2,
            barrier=-beta * float(np.sum(np.log(slack))),
            gb=beta / slack,
            H0=sparse.diags(beta / slack**2),
        )

    def hardware_geometry(self, x):
        return SimpleNamespace(
            slack=1 - x,
            jacobian=sparse.eye(len(x), format="csc"),
            ids=[f"upper:{i}" for i in range(len(x))],
        )

    def feasibility(self, x):
        return max(0.0, float(np.max(x - 1)))


def test_public_defaults_are_the_selected_mechanism():
    options = Options()
    assert isinstance(options, SolverOptions)
    assert options.scale_update == "step_aware"
    assert options.curvature_estimator == "ratio"
    assert not options.stop_on_target


def test_accepted_steps_and_serialized_diagnostics():
    result = minimize(rosenbrock(), [-1.2, 1], Options(max_iterations=8))
    assert isinstance(result, Result)
    assert result.nit == 8
    assert not result.success
    assert result.fun < result.history[0]["objective"]
    assert result.nfev == result["calls"]["value"] + result.njev
    assert any(t["alpha"] < 1 for t in result.trials if t["accepted"])
    assert all(t["actual"] > 0 and t["rho"] >= 0.1 for t in result.trials if t["accepted"])
    serialized = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert serialized["x"] == result.x.tolist()
    assert "final" not in serialized
    serialized["x"][0] = 100
    assert result.x[0] != 100


def test_objective_scale_covariance():
    options = Options(max_iterations=8)
    left = minimize(rosenbrock(), [-1.2, 1], options)
    right = minimize(rosenbrock(49), [-1.2, 1], options)
    np.testing.assert_allclose(
        [h["x"] for h in left.history],
        [h["x"] for h in right.history],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [49 * t["mu"] for t in left.trials], [t["mu"] for t in right.trials], rtol=1e-10
    )


def test_slack_qp_preserves_original_hardware_domain():
    result = minimize(
        UpperBox(),
        [0.995, 0.98, 0.9],
        Options(
            max_iterations=3,
            use_hardware_constraints=True,
            barrier_initial=0,
            barrier_minimum=0,
        ),
    )
    assert result.nit == 3
    assert all(np.all(h["x"] < 1) for h in result.history)
    for trial in result.trials:
        assert trial["alpha"] == 1
        assert trial["kappa_after"] == trial["kappa_before"] / 2
        assert trial["linearized_constraint_violation"] <= 1e-12


def test_limited_work_qp_reports_inexactness():
    result = minimize(
        UpperBox(),
        [0.995, 0.98, 0.9],
        Options(
            max_iterations=1,
            use_hardware_constraints=True,
            max_active_constraints=1,
            barrier_initial=0,
            barrier_minimum=0,
        ),
    )
    trial = result.trials[0]
    assert result.nit == 1
    assert not trial["qp_exact"]
    assert trial["qp_stationarity_residual"] > 1e-6
    assert trial["linearized_constraint_violation"] <= 1e-12
    assert trial["model_change"] < 0


def test_sparse_direction_matches_dense_kkt():
    rng = np.random.default_rng(17)
    n = 12
    jacobian = rng.normal(size=(3, n))
    residual = rng.normal(size=3)
    matrix = sparse.diags(np.linspace(0.1, 0.3, n), format="csc")
    affine = sparse.csc_matrix(rng.normal(size=(3, n)))
    raw = dict(
        f=float(residual @ residual),
        r=residual,
        Jr=jacobian,
        g=2 * jacobian.T @ residual,
        H0=matrix,
    )
    packet = _normalize(raw, n, True)
    direction = solve_direction(packet, 0.4, AffineGeometry(SimpleNamespace(A=affine), n))
    hessian = 2 * jacobian.T @ jacobian + matrix.toarray() + 0.4 * np.eye(n)
    kkt = np.block([[hessian, affine.toarray().T], [affine.toarray(), np.zeros((3, 3))]])
    expected = np.linalg.solve(kkt, np.r_[-packet.g, np.zeros(3)])[:n]
    np.testing.assert_allclose(direction.p, expected, rtol=1e-11, atol=1e-12)
    assert direction.constraint_residual < 1e-12


def test_large_constant_residual_does_not_hide_improvement():
    problem = ResidualProblem(lambda x: np.array([1e8, x[0]]), lambda x: np.array([[0.0], [1.0]]))
    result = minimize(
        problem, [0.01], Options(max_iterations=5, barrier_initial=0, barrier_minimum=0)
    )
    assert result.nit == 5
    assert abs(result.x[0]) < 1e-9
    assert all(h["f"] == 1e16 for h in result.history)
    assert all(t["actual"] > 0 for t in result.trials)


def test_direction_includes_additional_linear_objective_gradient():
    jacobian = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0]])
    residual = np.array([0.3, -0.2])
    linear_gradient = np.array([0.1, 0.2, -0.4])
    gradient = 2 * jacobian.T @ residual + linear_gradient
    packet = _normalize(
        dict(f=float(residual @ residual), r=residual, Jr=jacobian, g=gradient), 3, True
    )
    direction = solve_direction(packet, 0.4, AffineGeometry(SimpleNamespace(), 3))
    expected = np.linalg.solve(2 * jacobian.T @ jacobian + 0.4 * np.eye(3), -gradient)
    assert not direction.endpoint_system
    np.testing.assert_allclose(direction.p, expected, rtol=1e-12, atol=1e-13)
    assert direction.tangent_relative_residual < 1e-12


def test_numerical_stagnation_is_not_success():
    problem = ResidualProblem(lambda x: x, lambda x: np.eye(x.size))
    result = minimize(problem, [0.0])
    assert result.status == "NUMERICAL_STAGNATION"
    assert not result.success
    assert not result["checks"]["kkt_certified"]


def test_explicit_targets_enable_success():
    result = minimize(
        ResidualProblem(lambda x: x, lambda x: np.eye(x.size)),
        [0.0],
        Options(stop_on_target=True),
    )
    assert result.success
    assert result.nit == 0


def test_zero_budget_still_records_initial_and_return_queries():
    result = minimize(rosenbrock(), [-1.2, 1], Options(time_limit=0))
    assert result.status == "TIME_LIMIT"
    assert result.nit == 0
    assert result.nfev == 2
    assert result.njev == 1
    assert len(result.history) == 1


@pytest.mark.parametrize(
    "settings",
    [
        dict(time_limit=-1),
        dict(max_iterations=True),
        dict(barrier_minimum=0.1),
        dict(scale_update="unknown"),
        dict(slack_fraction=1),
        dict(estimator_decay=0),
    ],
)
def test_invalid_options_fail_before_queries(settings):
    with pytest.raises(ValueError):
        Options(**settings)


def test_complex_residual_requires_explicit_realification():
    with pytest.raises(ValueError, match="realified"):
        minimize(ResidualProblem(lambda x: np.array([1j]), lambda x: np.ones((1, 1))), [0.0])


def test_empty_coordinates_are_rejected():
    with pytest.raises(ValueError, match="Initial coordinates"):
        minimize(rosenbrock(), [])
