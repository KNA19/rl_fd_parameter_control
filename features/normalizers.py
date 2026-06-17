from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


FloatDict = Dict[str, float]


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.

    If conversion fails or gives NaN/inf, return default.
    """
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(value_float):
        return float(default)

    return float(value_float)


def clip01(value: float) -> float:
    """
    Clip a value to [0, 1].
    """
    value_float = safe_float(value)

    if value_float < 0.0:
        return 0.0

    if value_float > 1.0:
        return 1.0

    return value_float


def normalize_linear(
    value: float,
    scale: float,
) -> float:
    """
    Normalize value by a positive scale and clip to [0, 1].
    """
    value_float = safe_float(value)
    scale_float = safe_float(scale, default=1.0)

    if scale_float <= 0.0:
        return 0.0

    return clip01(value_float / scale_float)


def normalize_log1p(
    value: float,
    scale: float,
) -> float:
    """
    Log-normalize non-negative values using log1p.
    """
    value_float = max(0.0, safe_float(value))
    scale_float = max(1e-12, safe_float(scale, default=1.0))

    normalized = np.log1p(value_float) / np.log1p(scale_float)

    return clip01(float(normalized))


def normalize_signed_unit(
    value: float,
) -> float:
    """
    Map a value expected in [-1, 1] into [0, 1].
    """
    value_float = safe_float(value)
    value_float = max(-1.0, min(1.0, value_float))

    return float(0.5 * (value_float + 1.0))


def dictionary_to_vector(
    data: Mapping[str, float],
    names: Iterable[str],
) -> np.ndarray:
    """
    Convert a dictionary into a fixed-order float32 vector.

    Missing values become 0.0.
    """
    values = []

    for name in names:
        values.append(safe_float(data.get(name, 0.0)))

    return np.asarray(values, dtype=np.float32)


def clean_feature_dict(
    data: Mapping[str, float],
) -> FloatDict:
    """
    Return a finite float dictionary.
    """
    cleaned: FloatDict = {}

    for key, value in data.items():
        cleaned[str(key)] = safe_float(value)

    return cleaned