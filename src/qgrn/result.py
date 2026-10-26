"""Typed access to optimization outcomes and complete diagnostic records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize result value of type {type(value).__name__}")


class OptimizationResult(Mapping[str, Any]):
    """Last accepted iterate with numerical history and termination reason.

    ``success`` means the explicitly requested gate/objective/feasibility targets
    were reached. Time limits and numerical stagnation have separate statuses.
    Mapping access provides diagnostics
    such as ``trials``, ``checks``, ``restoration`` and oracle-specific fields.
    Arrays remain available in memory; :meth:`to_dict` returns a serializable
    copy, excluding the raw oracle payload by default.
    """

    def __init__(self, records: Mapping[str, Any]) -> None:
        self._records = dict(records)

    def __getitem__(self, key: str) -> Any:
        return self._records[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def x(self) -> NDArray[np.float64]:
        """Last accepted optimization coordinates."""
        return self._records["x"]

    @property
    def fun(self) -> float:
        """Complete physical objective, evaluated without the barrier."""
        return self._records["objective"]

    @property
    def status(self) -> str:
        """Termination status, such as TIME_LIMIT or TARGET_REACHED."""
        return self._records["status"]

    @property
    def message(self) -> str:
        """Detailed termination reason recorded by the solver."""
        return self._records["termination_reason"]

    @property
    def success(self) -> bool:
        """Whether the explicitly requested application targets were reached."""
        return self.status == "TARGET_REACHED"

    @property
    def nit(self) -> int:
        """Number of accepted steps."""
        return self._records["accepted"]

    @property
    def nfev(self) -> int:
        """All oracle calls, including derivative calls and failed queries."""
        return self._records["calls"]["value"] + self._records["calls"]["jacobian"]

    @property
    def njev(self) -> int:
        """Number of derivative oracle calls, including failed calls."""
        return self._records["calls"]["jacobian"]

    @property
    def elapsed(self) -> float:
        """Wall-clock duration, including restoration and the final evaluation."""
        return self._records["elapsed"]

    @property
    def history(self) -> list[dict[str, Any]]:
        """Initial and accepted-iterate records with cumulative timestamps."""
        return self._records["history"]

    @property
    def trials(self) -> list[dict[str, Any]]:
        """Direction attempts and their line-search outcomes."""
        return self._records["trials"]

    def to_dict(self, *, include_oracle: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible copy; nonfinite diagnostics become null.

        The optional raw oracle payload can contain sparse factors, callables
        or custom objects. Set ``include_oracle=True`` only if that payload is
        serializable; unsupported objects raise TypeError. The mapping interface
        retains all objects.
        """
        records = dict(self._records)
        if not include_oracle:
            records.pop("final", None)
        records.update(
            fun=self.fun,
            success=self.success,
            nit=self.nit,
            nfev=self.nfev,
            njev=self.njev,
        )
        return _json_value(records)
