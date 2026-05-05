from __future__ import annotations

from conftest import VALID_CODE

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.benchmarks.efficiency import benchmark_efficiency
from prism_challenge.evaluator.benchmarks.heldout import benchmark_heldout
from prism_challenge.evaluator.benchmarks.learning_speed import benchmark_learning_speed
from prism_challenge.evaluator.benchmarks.long_context import benchmark_long_context
from prism_challenge.evaluator.benchmarks.reasoning import benchmark_reasoning
from prism_challenge.evaluator.benchmarks.suite import run_benchmark_suite
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.synthetic import needle_text, reasoning_corpus
from prism_challenge.evaluator.tokenizer import HashTokenizer


def small_cfg() -> BenchConfig:
    return BenchConfig(
        train_steps=2,
        eval_steps=1,
        seeds=(13,),
        sequence_lengths=(16, 24),
        batch_size=1,
        vocab_size=256,
    )


def small_ctx() -> PrismContext:
    return PrismContext(vocab_size=256, sequence_length=16, max_parameters=1_000_000)


def test_tokenizer_and_synthetic_tasks_are_deterministic():
    tokenizer = HashTokenizer(256)
    assert tokenizer.encode("abc", 8) == tokenizer.encode("abc", 8)
    assert len(needle_text(32)) > 0
    assert len(reasoning_corpus()) == 3


def test_individual_benchmarks_return_finite_scores():
    cfg = small_cfg()
    ctx = small_ctx()
    learn = benchmark_learning_speed(VALID_CODE, ctx, cfg, seed=1)
    heldout = benchmark_heldout(VALID_CODE, ctx, cfg, seed=1)
    long_context = benchmark_long_context(VALID_CODE, ctx, cfg, seed=1)
    reasoning = benchmark_reasoning(VALID_CODE, ctx, cfg, seed=1)
    efficiency = benchmark_efficiency(VALID_CODE, ctx, cfg, quality=0.5)
    scores = [
        learn.score,
        heldout.score,
        long_context.score,
        reasoning.score,
        efficiency.score,
    ]
    for score in scores:
        assert 0 <= score <= 1
    assert set(long_context.accuracy_by_length) == {16, 24}


def test_benchmark_suite_combines_architecture_scores():
    result = run_benchmark_suite(VALID_CODE, small_ctx(), small_cfg())
    assert 0 <= result.q_arch <= 1
    for key in [
        "learning_speed",
        "heldout_generalization",
        "long_context",
        "reasoning",
        "multi_domain",
        "stability",
        "efficiency",
        "q_arch",
    ]:
        assert key in result.metrics
