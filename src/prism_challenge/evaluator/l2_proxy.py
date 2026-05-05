from __future__ import annotations

import math
from dataclasses import dataclass

from .interface import PrismContext
from .l1_syntax import validate_l1


@dataclass(frozen=True)
class L2Result:
    q_proxy: float
    parameter_count: int
    flops_estimate: float
    activation_memory_mb: float


def score_l2(code: str, ctx: PrismContext) -> L2Result:
    l1 = validate_l1(code, ctx)
    if not l1.valid:
        return L2Result(0.0, 0, 0.0, 0.0)
    params = max(l1.parameter_count, 1)
    flops = float(params * ctx.sequence_length * 2)
    activation_mb = float(ctx.sequence_length * math.sqrt(params) * 4 / 1_000_000)
    efficiency = 1.0 / (1.0 + math.log10(params))
    memory_score = 1.0 / (1.0 + activation_mb)
    q_proxy = max(0.0, min(1.0, 0.65 * efficiency + 0.35 * memory_score))
    return L2Result(q_proxy, params, flops, activation_mb)
