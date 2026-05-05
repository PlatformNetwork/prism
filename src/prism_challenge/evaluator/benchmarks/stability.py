from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.metrics import clamp, variance


@dataclass(frozen=True)
class StabilityResult:
    score: float
    variance: float


def benchmark_stability(seed_scores: list[float]) -> StabilityResult:
    var = variance(seed_scores)
    return StabilityResult(clamp(1.0 / (1.0 + var)), var)
