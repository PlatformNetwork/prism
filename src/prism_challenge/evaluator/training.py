from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bench_config import BenchConfig
from .dataset import fineweb_edu_samples
from .interface import PrismBatch, PrismContext, TrainingRecipe
from .sandbox import SubmissionRuntime, load_submission_runtime
from .tokenizer import HashTokenizer


@dataclass(frozen=True)
class TrainRun:
    initial_loss: float
    final_loss: float
    val_loss: float
    tokens_seen: int
    parameter_count: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _batch_from_tokens(tokens: torch.Tensor) -> PrismBatch:
    return PrismBatch(tokens=tokens[:, :-1], targets=tokens[:, 1:], metadata={})


def logits_for(
    model_or_runtime: torch.nn.Module | SubmissionRuntime, tokens: torch.Tensor
) -> torch.Tensor:
    if isinstance(model_or_runtime, SubmissionRuntime):
        runtime = model_or_runtime
        custom_infer = getattr(runtime.module, "inference_logits", None) or getattr(
            runtime.module, "infer", None
        )
        if callable(custom_infer):
            out = custom_infer(runtime.model, _batch_from_tokens(tokens), runtime.ctx)
        else:
            out = runtime.model(tokens[:, :-1])
    else:
        out = model_or_runtime(tokens[:, :-1])
    if not isinstance(out, torch.Tensor):
        raise TypeError("inference must return a tensor")
    if out.ndim == 2:
        out = out.unsqueeze(1).expand(-1, tokens.shape[1] - 1, -1)
    if out.ndim != 3:
        raise ValueError("model logits must have shape [batch, seq, vocab]")
    return out


def lm_loss(
    model_or_runtime: torch.nn.Module | SubmissionRuntime, tokens: torch.Tensor
) -> torch.Tensor:
    if isinstance(model_or_runtime, SubmissionRuntime):
        custom_loss = getattr(model_or_runtime.module, "compute_loss", None)
        if callable(custom_loss):
            loss = custom_loss(
                model_or_runtime.model, _batch_from_tokens(tokens), model_or_runtime.ctx
            )
            if not isinstance(loss, torch.Tensor):
                raise TypeError("compute_loss must return a torch.Tensor")
            return loss
    logits = logits_for(model_or_runtime, tokens)
    targets = tokens[:, 1:]
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1) % vocab)


def make_batch(
    tokenizer: HashTokenizer,
    texts: list[str],
    cfg: BenchConfig,
    device: torch.device,
    offset: int = 0,
) -> torch.Tensor:
    selected = [texts[(offset + i) % len(texts)] for i in range(cfg.batch_size)]
    return tokenizer.batch(selected, cfg.sequence_lengths[0], device)


def train_language_model(
    code: str, ctx: PrismContext, cfg: BenchConfig, seed: int, texts: list[str] | None = None
) -> TrainRun:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = load_submission_runtime(code, ctx)
    model = runtime.model
    recipe = runtime.recipe
    if not isinstance(runtime.recipe, TrainingRecipe):
        recipe = TrainingRecipe()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    model.to(device)
    model.train()
    params = sum(p.numel() for p in model.parameters())
    tokenizer = HashTokenizer(ctx.vocab_size)
    corpus = texts or fineweb_edu_samples(max(cfg.batch_size * 2, 4))
    custom_optimizer = getattr(runtime.module, "configure_optimizer", None)
    if callable(custom_optimizer):
        opt = custom_optimizer(model, recipe, ctx)
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=min(recipe.learning_rate, cfg.learning_rate),
            weight_decay=recipe.weight_decay,
        )
    initial = float(lm_loss(runtime, make_batch(tokenizer, corpus, cfg, device)).detach().cpu())
    final = initial
    for step in range(cfg.train_steps):
        batch = make_batch(tokenizer, corpus, cfg, device, step)
        custom_train_step = getattr(runtime.module, "train_step", None)
        if callable(custom_train_step):
            loss = custom_train_step(model, _batch_from_tokens(batch), opt, ctx)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("train_step must return a torch.Tensor")
        else:
            opt.zero_grad(set_to_none=True)
            loss = lm_loss(runtime, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
        final = float(loss.detach().cpu())
    model.eval()
    with torch.no_grad():
        val_texts = fineweb_edu_samples(max(cfg.batch_size * 2, 4))[::-1]
        val = float(lm_loss(runtime, make_batch(tokenizer, val_texts, cfg, device)).cpu())
    tokens_seen = cfg.train_steps * cfg.batch_size * cfg.sequence_lengths[0]
    if not all(math.isfinite(v) for v in [initial, final, val]):
        raise ValueError("non-finite training loss")
    return TrainRun(initial, final, val, tokens_seen, params)
