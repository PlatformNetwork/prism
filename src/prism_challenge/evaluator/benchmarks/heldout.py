from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import clamp, safe_exp_neg
from prism_challenge.evaluator.training import train_language_model


@dataclass(frozen=True)
class HeldoutResult:
    score: float
    train_loss: float
    val_loss: float
    gap: float


def benchmark_heldout(code: str, ctx: PrismContext, cfg: BenchConfig, seed: int) -> HeldoutResult:
    run = train_language_model(code, ctx, cfg, seed)
    gap = max(0.0, run.val_loss - run.final_loss)
    score = clamp(safe_exp_neg(run.val_loss) * safe_exp_neg(gap))
    return HeldoutResult(score, run.final_loss, run.val_loss, gap)
