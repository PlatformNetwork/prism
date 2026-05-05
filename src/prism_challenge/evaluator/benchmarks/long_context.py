from __future__ import annotations

from dataclasses import dataclass

import torch

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import clamp
from prism_challenge.evaluator.sandbox import SubmissionRuntime, load_submission_runtime
from prism_challenge.evaluator.synthetic import copy_text, needle_text
from prism_challenge.evaluator.tokenizer import HashTokenizer
from prism_challenge.evaluator.training import logits_for, set_seed


@dataclass(frozen=True)
class LongContextResult:
    score: float
    collapse_penalty: float
    accuracy_by_length: dict[int, float]


def _accuracy(runtime: SubmissionRuntime, tokens: torch.Tensor) -> float:
    with torch.no_grad():
        logits = logits_for(runtime, tokens)
        pred = logits.argmax(dim=-1)
        target = tokens[:, 1:] % logits.shape[-1]
        return float((pred == target).float().mean().cpu())


def benchmark_long_context(
    code: str, ctx: PrismContext, cfg: BenchConfig, seed: int
) -> LongContextResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = load_submission_runtime(code, ctx)
    model = runtime.model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    model.to(device).eval()
    tokenizer = HashTokenizer(ctx.vocab_size)
    acc: dict[int, float] = {}
    for length in cfg.sequence_lengths:
        texts = [needle_text(length), copy_text(length)]
        tokens = tokenizer.batch(texts, length, device)
        acc[length] = _accuracy(runtime, tokens)
    values = list(acc.values())
    collapse = max(0.0, values[0] - values[-1]) if values else 0.0
    score = clamp((sum(values) / len(values)) * (1.0 - collapse)) if values else 0.0
    return LongContextResult(score, collapse, acc)
