from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import clamp
from prism_challenge.evaluator.training import train_language_model


@dataclass(frozen=True)
class LearningSpeedResult:
    score: float
    initial_loss: float
    final_loss: float
    tokens_seen: int


def benchmark_learning_speed(
    code: str, ctx: PrismContext, cfg: BenchConfig, seed: int
) -> LearningSpeedResult:
    run = train_language_model(code, ctx, cfg, seed)
    score = clamp((run.initial_loss - run.final_loss) / max(run.initial_loss, 1e-6))
    return LearningSpeedResult(score, run.initial_loss, run.final_loss, run.tokens_seen)
