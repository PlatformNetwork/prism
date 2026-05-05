from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.metrics import harmonic_mean


@dataclass(frozen=True)
class GeneralismResult:
    score: float


def benchmark_generalism(domain_scores: list[float]) -> GeneralismResult:
    return GeneralismResult(harmonic_mean(domain_scores))
