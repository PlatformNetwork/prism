from __future__ import annotations

from dataclasses import dataclass

import torch

from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import clamp
from prism_challenge.evaluator.sandbox import load_submission_runtime
from prism_challenge.evaluator.synthetic import reasoning_corpus
from prism_challenge.evaluator.tokenizer import HashTokenizer
from prism_challenge.evaluator.training import logits_for, set_seed


@dataclass(frozen=True)
class ReasoningResult:
    score: float
    task_accuracy: dict[str, float]


def benchmark_reasoning(
    code: str, ctx: PrismContext, cfg: BenchConfig, seed: int
) -> ReasoningResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = load_submission_runtime(code, ctx)
    model = runtime.model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    model.to(device).eval()
    tokenizer = HashTokenizer(ctx.vocab_size)
    names = ["parentheses", "addition", "pattern"]
    scores: dict[str, float] = {}
    with torch.no_grad():
        for name, text in zip(names, reasoning_corpus(), strict=True):
            tokens = tokenizer.batch([text], cfg.sequence_lengths[0], device)
            logits = logits_for(runtime, tokens)
            pred = logits.argmax(dim=-1)
            target = tokens[:, 1:] % logits.shape[-1]
            scores[name] = float((pred == target).float().mean().cpu())
    return ReasoningResult(clamp(sum(scores.values()) / len(scores)), scores)
