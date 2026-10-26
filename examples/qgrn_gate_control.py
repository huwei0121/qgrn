"""Optimize an X rotation with exact propagation and Frechet derivatives.

Run after installing qgrn: python examples/qgrn_gate_control.py
This small standalone example checks the public API; it is not a paper benchmark.
"""

import numpy as np
from scipy.linalg import expm, expm_frechet

from qgrn import Options, ResidualProblem, minimize


def main():
    steps, dt = 12, 0.15
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    target = expm(-0.5j * np.pi * pauli_x)

    def propagate(controls, derivatives=False):
        matrices = [-1j * dt * (0.08 * pauli_z + u * pauli_x) for u in controls]
        propagators = [expm(matrix) for matrix in matrices]
        prefix = [np.eye(2, dtype=complex)]
        for propagator in propagators:
            prefix.append(propagator @ prefix[-1])
        difference = (prefix[-1] - target).ravel() / np.sqrt(2)
        residual = np.r_[difference.real, difference.imag]
        if not derivatives:
            return residual
        suffix = np.eye(2, dtype=complex)
        jacobian = np.empty((8, steps))
        for k in reversed(range(steps)):
            derivative = expm_frechet(matrices[k], -1j * dt * pauli_x, compute_expm=False)
            column = (suffix @ derivative @ prefix[k]).ravel() / np.sqrt(2)
            jacobian[:, k] = np.r_[column.real, column.imag]
            suffix = suffix @ propagators[k]
        return jacobian

    problem = ResidualProblem(propagate, lambda u: propagate(u, derivatives=True))
    start = np.random.default_rng(17).normal(0.8, 0.1, steps)
    result = minimize(problem, start, Options(max_iterations=30, time_limit=5))
    print(
        f"Q-GRN: {result.status}, objective={result.fun:.6g}, "
        f"accepted steps={result.nit}, oracle calls={result.nfev}"
    )
    assert result.fun < 1e-5


if __name__ == "__main__":
    main()
