"""Small numerical and record utilities shared by solver components."""

import math
from collections.abc import Mapping

import numpy as np


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _norm_inf(x):
    return float(np.max(np.abs(x), initial=0.0))


def _trace(matrix):
    return float(np.sum(matrix.diagonal()))


def _finite_or_none(x):
    return float(x) if math.isfinite(x) else None
