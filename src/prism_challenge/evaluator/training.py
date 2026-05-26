from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .bench_config import BenchConfig
from .checkpoints import (
    CHECKPOINT_METADATA_API_VERSION,
    CHECKPOINT_METADATA_FILENAME,
    CheckpointWorkspaceError,
    checkpoint_artifact_logical_size,
    load_checkpoint_metadata,
    metadata_path_for_checkpoint,
    resolve_checkpoint_artifact_path,
    write_checkpoint_metadata,
)
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
    checkpoint_path: str | None = None
    checkpoint_metadata_path: str | None = None
    resume_checkpoint_path: str | None = None
    load_checkpoint_metadata: dict[str, object] | None = None


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
    resume_metadata = _load_requested_resume_metadata(code, ctx)
    ctx = _enriched_checkpoint_context(ctx, device, resume_metadata)
    _prepare_current_checkpoint_dir(ctx)
    runtime = load_submission_runtime(code, ctx)
    model = runtime.model
    recipe = runtime.recipe
    if not isinstance(runtime.recipe, TrainingRecipe):
        recipe = TrainingRecipe()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    recipe_fingerprint = training_recipe_fingerprint(recipe)
    _validate_recipe_provenance(ctx, recipe_fingerprint)
    if resume_metadata is not None:
        if ctx.resume_checkpoint_dir is None:
            raise CheckpointWorkspaceError(
                "resume checkpoint metadata requires resume_checkpoint_dir"
            )
        resume_checkpoint_dir = Path(ctx.resume_checkpoint_dir)
        _validate_checkpoint_provenance(
            resume_metadata,
            code=code,
            ctx=ctx,
            runtime=runtime,
            recipe_fingerprint=recipe_fingerprint,
            checkpoint_dir=resume_checkpoint_dir,
        )
        load_result = _invoke_load_checkpoint(runtime, model, ctx, resume_checkpoint_dir)
    else:
        load_result = None
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
    checkpoint_record = _invoke_save_checkpoint(
        runtime,
        model,
        ctx,
        code=code,
        recipe_fingerprint=recipe_fingerprint,
    )
    return TrainRun(
        initial,
        final,
        val,
        tokens_seen,
        params,
        checkpoint_path=checkpoint_record["checkpoint_path"] if checkpoint_record else None,
        checkpoint_metadata_path=(
            checkpoint_record["checkpoint_metadata_path"] if checkpoint_record else None
        ),
        resume_checkpoint_path=(
            str(resume_metadata["checkpoint_path"]) if resume_metadata else None
        ),
        load_checkpoint_metadata=load_result,
    )


def training_recipe_fingerprint(recipe: TrainingRecipe) -> str:
    payload = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enriched_checkpoint_context(
    ctx: PrismContext,
    device: torch.device,
    resume_metadata: dict[str, Any] | None,
) -> PrismContext:
    metadata = dict(ctx.checkpoint_metadata)
    if resume_metadata is not None:
        metadata["resume_checkpoint_metadata"] = resume_metadata
        metadata["resume_checkpoint_path"] = resume_metadata["checkpoint_path"]
    return replace(
        ctx,
        checkpoint_dir=Path(ctx.checkpoint_dir) if ctx.checkpoint_dir is not None else None,
        resume_checkpoint_dir=(
            Path(ctx.resume_checkpoint_dir) if ctx.resume_checkpoint_dir is not None else None
        ),
        checkpoint_api_version=CHECKPOINT_METADATA_API_VERSION,
        is_resume=ctx.resume_checkpoint_dir is not None,
        device=device.type,
        checkpoint_metadata=metadata,
    )


def _prepare_current_checkpoint_dir(ctx: PrismContext) -> None:
    if ctx.checkpoint_dir is None:
        return
    checkpoint_dir = Path(ctx.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_dir.iterdir()):
        raise CheckpointWorkspaceError(
            f"current checkpoint directory must be empty before training: {checkpoint_dir}"
        )


def _load_requested_resume_metadata(code: str, ctx: PrismContext) -> dict[str, Any] | None:
    if ctx.resume_checkpoint_dir is None:
        return None
    resume_dir = Path(ctx.resume_checkpoint_dir)
    checkpoint_artifact_logical_size(resume_dir)
    metadata_files = tuple(sorted(resume_dir.rglob(CHECKPOINT_METADATA_FILENAME)))
    if not metadata_files:
        raise CheckpointWorkspaceError("requested resume checkpoint is missing metadata")
    if len(metadata_files) != 1:
        raise CheckpointWorkspaceError("requested resume checkpoint has multiple metadata files")
    metadata = load_checkpoint_metadata(metadata_files[0])
    _validate_checkpoint_file_from_metadata(metadata, resume_dir, metadata_files[0], ctx)
    if metadata["code_hash"] != _expected_code_hash(code, ctx):
        raise CheckpointWorkspaceError("checkpoint metadata code_hash does not match requested run")
    for field in ("submission_id", "arch_hash", "recipe_fingerprint"):
        expected = ctx.checkpoint_metadata.get(field)
        if expected is not None and metadata[field] != expected:
            raise CheckpointWorkspaceError(
                f"checkpoint metadata {field} does not match requested run"
            )
    if ctx.attempt > 1 and metadata["attempt"] != ctx.attempt - 1:
        raise CheckpointWorkspaceError("checkpoint metadata attempt is not the previous attempt")
    return metadata


def _invoke_load_checkpoint(
    runtime: SubmissionRuntime,
    model: torch.nn.Module,
    ctx: PrismContext,
    checkpoint_dir: Path,
) -> dict[str, object] | None:
    hook = getattr(runtime.module, "load_checkpoint", None)
    if not callable(hook):
        raise CheckpointWorkspaceError(
            "resume checkpoint requested but load_checkpoint hook is absent"
        )
    result = hook(model, checkpoint_dir, ctx)
    if result is None:
        return None
    if not isinstance(result, dict):
        raise TypeError("load_checkpoint must return None or a JSON-serializable dict")
    _require_json_serializable(result, "load_checkpoint return")
    return result


def _invoke_save_checkpoint(
    runtime: SubmissionRuntime,
    model: torch.nn.Module,
    ctx: PrismContext,
    *,
    code: str,
    recipe_fingerprint: str,
) -> dict[str, str] | None:
    if ctx.checkpoint_dir is None or ctx.rank != 0:
        return None
    hook = getattr(runtime.module, "save_checkpoint", None)
    if not callable(hook):
        return None
    checkpoint_dir = Path(ctx.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = hook(model, checkpoint_dir, ctx)
    relative_path, hook_return = _normalize_save_checkpoint_return(result)
    if relative_path is None:
        return None
    checkpoint_path = resolve_checkpoint_artifact_path(checkpoint_dir, relative_path)
    bytes_total = checkpoint_artifact_logical_size(checkpoint_dir)
    checkpoint_artifact_logical_size(checkpoint_dir, [relative_path])
    artifact_checkpoint_path = _artifact_relative_path(ctx, checkpoint_path)
    artifact_checkpoint_dir = _artifact_relative_path(ctx, checkpoint_dir)
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    metadata = {
        "checkpoint_api_version": CHECKPOINT_METADATA_API_VERSION,
        "submission_id": _metadata_value(ctx, "submission_id", "local"),
        "attempt": ctx.attempt,
        "code_hash": _expected_code_hash(code, ctx),
        "arch_hash": _metadata_value(ctx, "arch_hash", _runtime_arch_hash(runtime)),
        "recipe_fingerprint": recipe_fingerprint,
        "created_at": _utc_now(),
        "checkpoint_path": artifact_checkpoint_path,
        "hook_return": hook_return,
        "world_size": ctx.world_size,
        "rank_writer": 0,
        "checkpoint_dir": artifact_checkpoint_dir,
        "bytes_total": bytes_total,
    }
    write_checkpoint_metadata(metadata_path, metadata)
    checkpoint_artifact_logical_size(checkpoint_dir)
    validated = load_checkpoint_metadata(metadata_path)
    _validate_checkpoint_provenance(
        validated,
        code=code,
        ctx=ctx,
        runtime=runtime,
        recipe_fingerprint=recipe_fingerprint,
        checkpoint_dir=checkpoint_dir,
    )
    return {
        "checkpoint_path": artifact_checkpoint_path,
        "checkpoint_metadata_path": _artifact_relative_path(ctx, metadata_path),
    }


def _normalize_save_checkpoint_return(result: Any) -> tuple[str | None, dict[str, object] | None]:
    if result is None:
        return None, None
    if isinstance(result, str):
        return result, {"path": result}
    if isinstance(result, dict):
        if set(result) != {"path", "metadata"}:
            raise TypeError("save_checkpoint dict return must contain exactly path and metadata")
        if not isinstance(result["path"], str):
            raise TypeError("save_checkpoint return path must be a string")
        if not isinstance(result["metadata"], dict):
            raise TypeError("save_checkpoint return metadata must be a dict")
        _require_json_serializable(result, "save_checkpoint return")
        return result["path"], result
    raise TypeError("save_checkpoint must return None, str, or {'path': str, 'metadata': dict}")


def _validate_checkpoint_provenance(
    metadata: dict[str, Any],
    *,
    code: str,
    ctx: PrismContext,
    runtime: SubmissionRuntime,
    recipe_fingerprint: str,
    checkpoint_dir: Path,
) -> None:
    expected = {
        "submission_id": _metadata_value(ctx, "submission_id", "local"),
        "code_hash": _expected_code_hash(code, ctx),
        "arch_hash": _metadata_value(ctx, "arch_hash", _runtime_arch_hash(runtime)),
        "recipe_fingerprint": recipe_fingerprint,
    }
    for field, value in expected.items():
        if metadata[field] != value:
            raise CheckpointWorkspaceError(
                f"checkpoint metadata {field} does not match current run"
            )
    if metadata["checkpoint_dir"] != _artifact_relative_path(ctx, checkpoint_dir):
        raise CheckpointWorkspaceError(
            "checkpoint metadata checkpoint_dir does not match current run"
        )


def _validate_checkpoint_file_from_metadata(
    metadata: dict[str, Any], metadata_root: Path, metadata_path: Path, ctx: PrismContext
) -> None:
    artifact_root = _artifact_root(ctx, metadata_root)
    checkpoint_path = artifact_root / str(metadata["checkpoint_path"])
    try:
        checkpoint_path.resolve(strict=False).relative_to(metadata_root.resolve(strict=False))
    except ValueError as exc:
        raise CheckpointWorkspaceError(
            "checkpoint metadata path escapes requested resume dir"
        ) from exc
    resolve_checkpoint_artifact_path(metadata_root, checkpoint_path.relative_to(metadata_root))
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise CheckpointWorkspaceError("checkpoint metadata points to a missing checkpoint file")
    resolved_expected_metadata = metadata_path_for_checkpoint(checkpoint_path).resolve(strict=False)
    resolved_metadata = metadata_path.resolve(strict=False)
    if resolved_expected_metadata != resolved_metadata:
        raise CheckpointWorkspaceError("checkpoint metadata is not next to the checkpoint")
    if metadata["checkpoint_dir"] != _artifact_relative_path(ctx, metadata_root):
        raise CheckpointWorkspaceError(
            "checkpoint metadata checkpoint_dir does not match resume dir"
        )


def _validate_recipe_provenance(ctx: PrismContext, recipe_fingerprint: str) -> None:
    expected = ctx.checkpoint_metadata.get("recipe_fingerprint")
    if expected is not None and expected != recipe_fingerprint:
        raise CheckpointWorkspaceError(
            "checkpoint metadata recipe_fingerprint does not match recipe"
        )


def _metadata_value(ctx: PrismContext, field: str, default: str) -> str:
    value = ctx.checkpoint_metadata.get(field, default)
    if not isinstance(value, str) or not value:
        raise CheckpointWorkspaceError(
            f"checkpoint provenance {field} must be a non-empty string"
        )
    return value


def _expected_code_hash(code: str, ctx: PrismContext) -> str:
    return _metadata_value(ctx, "code_hash", hashlib.sha256(code.encode("utf-8")).hexdigest())


def _runtime_arch_hash(runtime: SubmissionRuntime) -> str:
    basis = "\n".join(sorted(runtime.report.ast_fingerprint))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _artifact_root(ctx: PrismContext, path: Path) -> Path:
    for field in ("artifact_output_path", "artifact_output", "artifact_root"):
        value = ctx.checkpoint_metadata.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise CheckpointWorkspaceError(
                f"checkpoint metadata {field} must be a non-empty string"
            )
        return Path(value)
    parts = path.resolve(strict=False).parts
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        return Path(*parts[:index])
    return Path(ctx.checkpoint_dir or ctx.resume_checkpoint_dir or path).resolve(strict=False)


def _artifact_relative_path(ctx: PrismContext, path: Path) -> str:
    root = _artifact_root(ctx, path).resolve(strict=False)
    target = Path(path).resolve(strict=False)
    try:
        return target.relative_to(root).as_posix()
    except ValueError as exc:
        raise CheckpointWorkspaceError(
            f"checkpoint artifact is outside artifact root: {path}"
        ) from exc


def _require_json_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except TypeError as exc:
        raise TypeError(f"{label} must be JSON serializable") from exc


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
