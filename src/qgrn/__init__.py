"""Q-GRN: Quantum Gradient-Regularized Newton optimization."""

from .options import SolverOptions
from .problem import ConstrainedOracle, ResidualOracle, ResidualProblem
from .result import OptimizationResult
from .solver import minimize

__version__ = "0.1.0"
Options = SolverOptions
Result = OptimizationResult

__all__ = [
    "ConstrainedOracle",
    "OptimizationResult",
    "Options",
    "ResidualOracle",
    "ResidualProblem",
    "Result",
    "SolverOptions",
    "minimize",
]
