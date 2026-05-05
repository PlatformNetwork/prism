from __future__ import annotations

from dataclasses import dataclass

from .bench_config import DEFAULT_BENCH_CONFIG, BenchConfig
from .benchmarks.suite import run_benchmark_suite
from .interface import PrismContext
from .lium_client import LiumClient
from .metrics import clamp


@dataclass(frozen=True)
class L4Result:
    q_arch: float
    scale_collapse: bool
    metrics: dict[str, float]


async def score_l4(
    code: str,
    ctx: PrismContext,
    lium: LiumClient,
    cfg: BenchConfig = DEFAULT_BENCH_CONFIG,
) -> L4Result:
    suite = run_benchmark_suite(code, ctx, cfg)
    if not lium.enabled():
        return L4Result(q_arch=suite.q_arch, scale_collapse=False, metrics=suite.metrics)
    job = await lium.submit_job(
        {
            "challenge": "prism",
            "level": "l4",
            "benchmarks": [
                "learning_speed",
                "heldout",
                "long_context",
                "reasoning",
                "generalism",
                "stability",
                "efficiency",
            ],
            "config": {
                "train_steps": cfg.train_steps,
                "eval_steps": cfg.eval_steps,
                "seeds": list(cfg.seeds),
                "sequence_lengths": list(cfg.sequence_lengths),
                "batch_size": cfg.batch_size,
                "vocab_size": cfg.vocab_size,
            },
            "local_metrics": suite.metrics,
        },
        idempotency_key=str(abs(hash(code))),
    )
    metrics = {**suite.metrics, **dict(job.metrics)}
    q_arch = clamp(float(metrics.get("q_arch", suite.q_arch)))
    small_loss = float(metrics.get("final_loss", 1.0))
    large_loss = float(metrics.get("val_loss", small_loss))
    scale_collapse = large_loss > small_loss * 1.15
    if scale_collapse:
        q_arch = 0.0
    metrics.update({"scale_small_loss": small_loss, "scale_large_loss": large_loss})
    return L4Result(q_arch, scale_collapse, metrics)
