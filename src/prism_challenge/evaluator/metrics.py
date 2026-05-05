from __future__ import annotations

import math
from collections.abc import Iterable


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_exp_neg(value: float) -> float:
    return clamp(math.exp(-max(0.0, min(value, 20.0))))


def harmonic_mean(values: Iterable[float]) -> float:
    vals = [max(0.0, v) for v in values]
    positives = [v for v in vals if v > 0]
    if not positives or len(positives) != len(vals):
        return 0.0
    return len(positives) / sum(1.0 / v for v in positives)


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
