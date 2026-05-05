from __future__ import annotations

import math
from dataclasses import dataclass

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.l2_proxy import score_l2
from prism_challenge.evaluator.metrics import clamp


@dataclass(frozen=True)
class EfficiencyResult:
    score: float
    parameter_count: int
    flops_estimate: float
    activation_memory_mb: float


def benchmark_efficiency(
    code: str, ctx: PrismContext, cfg: BenchConfig, quality: float
) -> EfficiencyResult:
    l2 = score_l2(code, ctx)
    cost = math.log10(max(10.0, l2.parameter_count + l2.flops_estimate + l2.activation_memory_mb))
    score = clamp((quality + l2.q_proxy) / (1.0 + cost / 10.0))
    return EfficiencyResult(score, l2.parameter_count, l2.flops_estimate, l2.activation_memory_mb)
