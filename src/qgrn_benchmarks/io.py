"""JSON serialization, NumPy encoding, and numerical-source checksums.

``write`` replaces its destination. Run entry points check for existing records
before writing so that independent solver runs keep separate output files.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def encode(value):
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(
                shape=list(value.shape),
                real=value.real.ravel(order="F").tolist(),
                imag=value.imag.ravel(order="F").tolist(),
                order="F",
            )
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(encode(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def source_hashes():
    """Hash the actual Python solver, models, baseline and configuration sources."""
    import hashlib

    import qgrn

    roots = {"qgrn": Path(qgrn.__file__).parent, "qgrn_benchmarks": Path(__file__).parent}
    return {
        f"{name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for name, directory in roots.items()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in (".py", ".json")
    }
