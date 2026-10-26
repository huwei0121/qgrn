"""Residual adapters and the contract for application-supplied quantum models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


@runtime_checkable
class ResidualOracle(Protocol):
    """An oracle for f=||r||^2, g=2 Jr.T r in real coordinates.

    Return a mapping or object with fields ``f``, ``r``, and, when order=1,
    ``Jr`` and ``g``. Return ``valid=False, reason=...`` outside the domain.
    Optional ``gate_error`` is distinct from the complete objective.
    """

    def residual_oracle(self, x: Vector, order: int = 1) -> Mapping[str, Any]: ...


@runtime_checkable
class ConstrainedOracle(Protocol):
    """Complete residual model with optional hardware constraints.

    Value queries require ``f`` and ``r``. Derivative queries additionally
    require ``g`` and ``Jr``. For n coordinates and m residuals, r has shape
    (m,), g has shape (n,), and Jr has shape (m, n). The complete physical
    gradient g includes any additional running or linear costs.

    The direction Hessian is ``2 Jr.T Jr + H0``. H0 is a symmetric positive
    semidefinite (n, n) matrix containing supplementary physical curvature and
    any barrier curvature. Jreg factors the physical part of that supplementary
    model, so the predicted physical curvature is ``2 Jr.T Jr + Jreg.T Jreg``.
    ``barrier`` and ``gb`` are the barrier value and gradient at fixed beta;
    the line search evaluates the barrier change exactly.

    Optional full-row-rank A defines affine tangent constraints A p = 0.
    The default initial restoration projects onto A x = 0. A nonzero affine
    offset requires restore_initial and feasibility methods. Hardware models
    expose hardware_geometry(x) as an object with slack=-c(x)>0, jacobian=Dc
    and stable ids. Each trial oracle checks the nonlinear physical domain.

    For residual/split estimates, remainder_residual F must represent the full
    physical objective as .5||F||^2, with remainder_jacobian=DF. A Jacobian
    callable applying DF to a vector is supported. Barrier terms are excluded.
    """

    def grn_oracle(self, x: Vector, *, derivatives: bool, beta: float) -> Mapping[str, Any]: ...


class ResidualProblem:
    """Create an unconstrained real residual problem from two callables.

    The minimized objective is ``sum(residual(x)**2)``. The Jacobian has one
    row per residual and one column per coordinate. Complex residuals must be
    split into real and imaginary parts explicitly. Constrained physical
    models use the ConstrainedOracle interface.
    """

    def __init__(
        self,
        residual: Callable[[Vector], ArrayLike],
        jacobian: Callable[[Vector], ArrayLike],
    ) -> None:
        if not callable(residual) or not callable(jacobian):
            raise TypeError("residual and jacobian must be callable")
        self.residual = residual
        self.jacobian = jacobian

    def residual_oracle(self, x: Vector, order: int = 1) -> dict[str, Any]:
        raw = np.asarray(self.residual(x))
        if np.iscomplexobj(raw):
            raise ValueError("Residuals must be explicitly realified")
        r = np.asarray(raw, dtype=float)
        if r.ndim != 1:
            raise ValueError("Residuals must be a vector")
        value: dict[str, Any] = {"f": float(r @ r), "r": r}
        if order:
            raw_jacobian = np.asarray(self.jacobian(x))
            if np.iscomplexobj(raw_jacobian):
                raise ValueError("The Jacobian must be explicitly realified")
            jacobian = np.asarray(raw_jacobian, dtype=float)
            if jacobian.shape != (r.size, x.size):
                raise ValueError("The Jacobian must have shape (residuals, coordinates)")
            value.update(Jr=jacobian, g=2 * jacobian.T @ r)
        return value
