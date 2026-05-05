from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.benchmarks.efficiency import benchmark_efficiency
from prism_challenge.evaluator.benchmarks.generalism import benchmark_generalism
from prism_challenge.evaluator.benchmarks.heldout import benchmark_heldout
from prism_challenge.evaluator.benchmarks.learning_speed import benchmark_learning_speed
from prism_challenge.evaluator.benchmarks.long_context import benchmark_long_context
from prism_challenge.evaluator.benchmarks.reasoning import benchmark_reasoning
from prism_challenge.evaluator.benchmarks.stability import benchmark_stability
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import clamp


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    q_arch: float
    metrics: dict[str, float]


def run_benchmark_suite(code: str, ctx: PrismContext, cfg: BenchConfig) -> BenchmarkSuiteResult:
    seed_scores: list[float] = []
    first_metrics: dict[str, float] = {}
    for seed in cfg.seeds:
        learn = benchmark_learning_speed(code, ctx, cfg, seed)
        heldout = benchmark_heldout(code, ctx, cfg, seed)
        long_ctx = benchmark_long_context(code, ctx, cfg, seed)
        reasoning = benchmark_reasoning(code, ctx, cfg, seed)
        domain_scores = [learn.score, heldout.score, long_ctx.score, reasoning.score]
        generalism = benchmark_generalism(domain_scores)
        partial_quality = clamp(
            0.28 * learn.score
            + 0.24 * heldout.score
            + 0.24 * long_ctx.score
            + 0.24 * reasoning.score
        )
        efficiency = benchmark_efficiency(code, ctx, cfg, partial_quality)
        seed_score = clamp(
            0.22 * learn.score
            + 0.18 * heldout.score
            + 0.18 * long_ctx.score
            + 0.18 * reasoning.score
            + 0.12 * generalism.score
            + 0.05 * efficiency.score
        )
        seed_scores.append(seed_score)
        if not first_metrics:
            first_metrics = {
                "learning_speed": learn.score,
                "initial_loss": learn.initial_loss,
                "final_loss": learn.final_loss,
                "heldout_generalization": heldout.score,
                "val_loss": heldout.val_loss,
                "generalization_gap": heldout.gap,
                "long_context": long_ctx.score,
                "long_context_collapse": long_ctx.collapse_penalty,
                "reasoning": reasoning.score,
                "multi_domain": generalism.score,
                "efficiency": efficiency.score,
                "parameter_count": float(efficiency.parameter_count),
                "flops_estimate": efficiency.flops_estimate,
                "activation_memory_mb": efficiency.activation_memory_mb,
            }
    stability = benchmark_stability(seed_scores)
    mean_without_stability = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
    q_arch = clamp(mean_without_stability + 0.07 * stability.score)
    first_metrics.update(
        {
            "stability": stability.score,
            "score_variance": stability.variance,
            "q_arch": q_arch,
        }
    )
    return BenchmarkSuiteResult(q_arch=q_arch, metrics=first_metrics)
