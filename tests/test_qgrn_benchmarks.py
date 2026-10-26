"""Physics, pairing, and release-boundary checks for the benchmark package."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from qgrn_benchmarks.analysis import aggregate_reference, verify_reference
from qgrn_benchmarks.endpoint import run_case
from qgrn_benchmarks.io import read
from qgrn_benchmarks.models import build_task, task_ids
from qgrn_benchmarks.reconstruction import evaluate, reconstruct_controls, slice_controls

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkReleaseTests(unittest.TestCase):
    def test_endpoint_grn_options_override_defaults(self):
        observed = {}

        def evaluate_initial(task, x, options, *, started):
            observed.update(estimator=options.estimator, ratio_update=options.ratio_update)
            return dict(status="ITERATION_LIMIT", x=x, objective=task.oracle(x).f, history=[])

        with TemporaryDirectory(prefix="qgrn-endpoint-") as directory:
            with patch("qgrn_benchmarks.endpoint.solve_grn", side_effect=evaluate_initial):
                result = run_case(
                    "jandura_cz",
                    85308,
                    Path(directory) / "run.json",
                    method="grn",
                    seconds=30,
                    grn_options={"estimator": "residual", "ratio_update": "legacy"},
                )
        self.assertTrue(result["return_valid"], result["failures"])
        self.assertEqual(observed, {"estimator": "residual", "ratio_update": "legacy"})

    def test_unknown_endpoint_method_creates_no_output(self):
        with TemporaryDirectory(prefix="qgrn-endpoint-") as directory:
            output = Path(directory) / "run.json"
            with self.assertRaisesRegex(ValueError, "Unsupported endpoint method"):
                run_case("jandura_cz", 85308, output, method="unknown")
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".assets").exists())

    def test_complete_endpoint_models_directional_derivatives(self):
        for name in task_ids():
            with self.subTest(problem=name):
                task = build_task(name)
                x = task.initial(85301)
                direction = np.random.default_rng(17).normal(size=x.size)
                direction /= np.linalg.norm(direction)
                value = task.oracle(x)
                step = 2e-6
                finite = (
                    task.oracle(x + step * direction).f - task.oracle(x - step * direction).f
                ) / (2 * step)
                self.assertAlmostEqual(finite, float(value.g @ direction), delta=2e-7)
                self.assertLess(abs(task.rollout(x)["final_objective"] - value.f), 1e-10)

    def test_reference_contains_only_paired_selected_ten(self):
        result = verify_reference(ROOT / "reproduction")
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["runs"], 200)

    def test_statistics_recompute_from_individual_records(self):
        result = aggregate_reference(ROOT / "reproduction")
        original = read(ROOT / "reproduction/data/plotted_values.json")
        for task, panel in result["panels"].items():
            for method, curve in panel["methods"].items():
                for field in ("seed_values", "q25", "median", "q75"):
                    np.testing.assert_array_equal(
                        curve[field], original["panels"][task]["methods"][method][field]
                    )

    def test_common_reconstruction_preserves_original_constraints(self):
        config = read(ROOT / "reproduction/configs/selected.json")
        for spec in config["tasks"][:3]:
            with self.subTest(problem=spec["id"]):
                data = read(
                    ROOT / f"reproduction/data/inputs/{spec['id']}__{spec['seeds'][0]}.json"
                )
                candidate, diagnostic = reconstruct_controls(
                    data["model"], slice_controls(data["initial"])
                )
                second, _ = reconstruct_controls(data["model"], candidate)
                for key in candidate:
                    np.testing.assert_allclose(candidate[key], second[key], rtol=0, atol=1e-13)
                physics = evaluate(data["model"], candidate)
                self.assertLess(diagnostic["control_feasibility"], 1e-10)
                self.assertLess(physics["quantum_dynamics"], 1e-10)


if __name__ == "__main__":
    unittest.main()
