from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from prism_challenge.config import PrismSettings

from .bench_config import BenchConfig
from .dataset import (
    FINEWEB_EDU_SUBSETS,
    FineWebEduConfig,
    fineweb_edu_manifest_fields_for_mode,
    load_fineweb_edu_contract,
)
from .interface import PrismContext
from .metrics import collect_diagnostics, diagnostics_to_manifest
from .schemas import (
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    ExecutionMode,
    PrismRunManifest,
)
from .training import train_language_model

LOCAL_SMOKE_TOKEN_BUDGET = 1024
GPU_PROXY_TOKEN_TARGET = 10_000_000_000
FULL_SCALE_PHASE_1_TOKEN_TARGET = 10_000_000_000
FULL_SCALE_PHASE_2_PARAMETER_TARGET = 1_000_000_000
FULL_SCALE_PHASE_2_TOKEN_TARGET = 100_000_000_000


@dataclass(frozen=True)
class LocalSmokeResult:
    metrics: dict[str, float]
    run_manifest: dict[str, Any]
    artifact_output_path: str
    run_manifest_path: str


def execution_mode_from_value(value: str | ExecutionMode | None) -> ExecutionMode:
    if value is None:
        return ExecutionMode.GPU_PROXY_EVAL
    return value if isinstance(value, ExecutionMode) else ExecutionMode(str(value))


def build_evaluation_mode_spec(
    mode: ExecutionMode,
    *,
    settings: PrismSettings,
    gpu_count: int | None = None,
    max_gpu_count: int | None = None,
    gpu_type: str | None = None,
    gpu_server: str | None = None,
    gpu_device_ids: tuple[str, ...] | list[str] | None = None,
    artifact_output_path: str = "/artifacts",
    run_manifest_path: str = f"/artifacts/{RUN_MANIFEST_FILENAME}",
) -> dict[str, Any]:
    dataset = fineweb_edu_manifest_fields_for_mode(mode)
    token_budget = _token_budget(mode)
    parameter_target = _parameter_target(mode, settings)
    spec: dict[str, Any] = {
        "mode": mode.value,
        "official_score_eligible": mode is not ExecutionMode.LOCAL_CPU_SMOKE,
        "image": settings.platform_eval_image,
        "command": list(_gpu_runner_command(gpu_count or settings.platform_eval_gpu_count)),
        "token_budget": token_budget,
        "parameter_target": parameter_target,
        "dataset": {
            **dataset,
            "subset": _subset_for_mode(mode),
            "token_count": _dataset_token_count(mode),
            "network_fallback_allowed": False,
        },
        "resource_profile": _resource_profile(
            mode,
            settings,
            gpu_count=gpu_count,
            max_gpu_count=max_gpu_count,
            gpu_type=gpu_type,
            gpu_server=gpu_server,
            gpu_device_ids=gpu_device_ids,
        ),
        "artifact_output_path": artifact_output_path,
        "run_manifest_path": run_manifest_path,
        "manifest_filename": RUN_MANIFEST_FILENAME,
        "loss_contract": {
            "normalization_scope": "byte_normalized",
            "loss_comparable_required": True,
            "raw_final_loss_cross_architecture_signal": False,
        },
    }
    if mode is ExecutionMode.FULL_SCALE_EVAL:
        spec["phases"] = [
            {
                "name": "full_scale_10b_tokens",
                "token_budget": FULL_SCALE_PHASE_1_TOKEN_TARGET,
                "parameter_target": parameter_target,
                "dataset_subset": "sample-10BT",
            },
            {
                "name": "phase_2_1b_params_100b_tokens",
                "token_budget": FULL_SCALE_PHASE_2_TOKEN_TARGET,
                "parameter_target": FULL_SCALE_PHASE_2_PARAMETER_TARGET,
                "dataset_subset": "sample-100BT",
            },
        ]
    return spec


def run_local_cpu_smoke(
    *,
    submission_id: str,
    code: str,
    code_hash: str,
    arch_hash: str,
    ctx: PrismContext,
    artifact_output_path: Path,
) -> LocalSmokeResult:
    artifact_output_path.mkdir(parents=True, exist_ok=True)
    cfg = BenchConfig(
        train_steps=1,
        eval_steps=1,
        sequence_lengths=(16,),
        batch_size=1,
        max_tokens=64,
    )
    contract = load_fineweb_edu_contract(FineWebEduConfig(mode=ExecutionMode.LOCAL_CPU_SMOKE))
    start = perf_counter()
    train = train_language_model(
        code,
        ctx,
        cfg,
        seed=13,
        texts=contract.texts("train"),
    )
    wall_clock_seconds = max(0.0, perf_counter() - start)
    estimated_flops = float(max(1, train.tokens_seen) * max(1, train.parameter_count) * 6)
    architecture_graph_hash = _write_json(
        artifact_output_path / "architecture_graph.json",
        {
            "schema_version": "architecture_graph.v1",
            "submission_id": submission_id,
            "code_hash": code_hash,
            "arch_hash": arch_hash,
            "mode": ExecutionMode.LOCAL_CPU_SMOKE.value,
        },
    )
    metadata_hash = _write_json(
        artifact_output_path / "architecture_metadata.v1.json",
        {
            "schema_version": "architecture_metadata.v1",
            "submission_id": submission_id,
            "local_cpu_smoke": True,
            "official_score_eligible": False,
        },
    )
    metrics_payload = _metrics_payload(train, estimated_flops)
    metrics_hash = _write_json(artifact_output_path / "metrics.json", metrics_payload)
    run_log_hash = _write_text(
        artifact_output_path / "run.log",
        "local_cpu_smoke completed fixture training and manifest validation\n",
    )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "submission_id": submission_id,
        "architecture_id": f"local-smoke-architecture-{arch_hash[:12]}",
        "architecture_version_id": f"local-smoke-architecture-version-{code_hash[:12]}",
        "training_script_version_id": f"local-smoke-training-{code_hash[:12]}",
        "run_id": f"local-cpu-smoke-{submission_id}",
        "mode": ExecutionMode.LOCAL_CPU_SMOKE.value,
        "dataset": contract.manifest_fields(),
        "model": {
            "parameter_count": train.parameter_count,
            "architecture_graph_hash": architecture_graph_hash,
            "tokenizer_kind": contract.config.tokenizer_kind,
            "vocab_size": ctx.vocab_size,
            "max_sequence_length": cfg.sequence_lengths[0],
        },
        "compute": {
            "gpu_count": 0,
            "gpu_type": None,
            "gpu_server": None,
            "gpu_device_ids": [],
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "distributed_backend": None,
            "effective_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": 1,
            "tokens_seen": train.tokens_seen,
            "estimated_flops": estimated_flops,
            "wall_clock_seconds": wall_clock_seconds,
            "checkpoint_path": None,
            "resume_checkpoint_path": None,
        },
        "metrics": metrics_payload,
        "artifacts": {
            "architecture_graph": _artifact_reference(
                artifact_output_path / "architecture_graph.json",
                architecture_graph_hash,
                "application/json",
            ),
            "architecture_metadata": _artifact_reference(
                artifact_output_path / "architecture_metadata.v1.json",
                metadata_hash,
                "application/json",
            ),
            "run_log": _artifact_reference(
                artifact_output_path / "run.log",
                run_log_hash,
                "text/plain",
            ),
            "checkpoints": [],
            "metrics": _artifact_reference(
                artifact_output_path / "metrics.json",
                metrics_hash,
                "application/json",
            ),
        },
        "validation": {
            "passed": True,
            "score_eligible": False,
            "deterministic_evidence": [],
            "warnings": [
                "local_cpu_smoke validates evaluator wiring only and is not official score eligible"
            ],
            "errors": [],
        },
    }
    PrismRunManifest.model_validate(manifest)
    manifest_path = artifact_output_path / RUN_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return LocalSmokeResult(
        metrics=_local_smoke_summary_metrics(manifest),
        run_manifest=manifest,
        artifact_output_path=str(artifact_output_path),
        run_manifest_path=str(manifest_path),
    )


def _metrics_payload(train: Any, estimated_flops: float) -> dict[str, Any]:
    relative_loss_reduction = max(0.0, train.initial_loss - train.val_loss) / max(
        train.initial_loss, 1e-12
    )
    slope = (train.final_loss - train.initial_loss) / max(float(train.tokens_seen), 1.0)
    return {
        "loss_vs_tokens": [
            {"x": 0.0, "loss": train.initial_loss},
            {"x": float(train.tokens_seen), "loss": train.final_loss},
        ],
        "loss_vs_compute": [
            {"x": 0.0, "loss": train.initial_loss},
            {"x": estimated_flops, "loss": train.final_loss},
        ],
        "loss_vs_params": [{"x": float(train.parameter_count), "loss": train.val_loss}],
        "learning_speed_slope": slope,
        "tokens_seen": train.tokens_seen,
        "estimated_flops": estimated_flops,
        "parameter_count": train.parameter_count,
        "benchmark_scores": {},
        "benchmark_capability_metadata": {
            "status": "not_run",
            "reason": "local_cpu_smoke wiring only",
        },
        "benchmark_noise_metadata": {
            "status": "not_run",
            "reason": "local_cpu_smoke deterministic fixture",
        },
        "benchmark_contamination_metadata": {
            "required": False,
            "reason": "local fixture is not official scoring data",
        },
        "diagnostics": _smoke_diagnostics(),
        "gpu_count": 0,
        "loss": {
            "raw_final_loss": train.final_loss,
            "standardized_eval_loss": train.val_loss,
            "loss_normalization_scope": "byte_normalized",
            "baseline_run_id": "local_cpu_smoke_fixture_baseline_v1",
            "relative_loss_reduction": relative_loss_reduction,
            "architecture_normalized_heldout_improvement": relative_loss_reduction,
            "loss_comparable": True,
            "loss_component_redistribution": {"enabled": False},
        },
        "final_loss": train.final_loss,
    }


def _smoke_diagnostics() -> dict[str, dict[str, Any]]:
    activations = {"layer_0": torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.2, 0.5]])}
    representations = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.4, 0.2]])
    gradients = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.11, 0.19, 0.31])]
    return diagnostics_to_manifest(
        collect_diagnostics(
            activations=activations,
            representations=representations,
            gradient_samples=gradients,
            attention_weights=None,
        )
    )


def _local_smoke_summary_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    metrics = manifest["metrics"]
    final_loss = float(metrics["final_loss"])
    relative = float(metrics["loss"]["relative_loss_reduction"])
    parameter_count = max(float(metrics["parameter_count"]), 1.0)
    q_arch = max(0.0, min(1.0, 0.7 * relative + 0.3 / (1.0 + math.log10(parameter_count))))
    return {
        "q_arch": q_arch,
        "q_recipe": 0.0,
        "final_loss": final_loss,
        "train_loss": final_loss,
        "eval_loss": float(metrics["loss"]["standardized_eval_loss"]),
        "tokens_seen": float(metrics["tokens_seen"]),
        "parameter_count": float(metrics["parameter_count"]),
        "gpu_count": 0.0,
    }


def _resource_profile(
    mode: ExecutionMode,
    settings: PrismSettings,
    *,
    gpu_count: int | None = None,
    max_gpu_count: int | None = None,
    gpu_type: str | None = None,
    gpu_server: str | None = None,
    gpu_device_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return {
            "profile": "local_cpu_smoke",
            "cpus": min(settings.platform_eval_cpus, 1.0),
            "memory": "512m",
            "gpu_count": 0,
            "max_gpu_count": 0,
            "gpu_type": None,
            "official_fixed_profile": False,
        }
    return {
        "profile": "fixed_official_gpu",
        "cpus": settings.platform_eval_cpus,
        "memory": settings.platform_eval_memory,
        "gpu_count": gpu_count or settings.platform_eval_gpu_count,
        "max_gpu_count": max_gpu_count or settings.platform_eval_max_gpu_count,
        "gpu_type": gpu_type if gpu_type is not None else settings.platform_eval_gpu_type,
        "gpu_server": (
            gpu_server if gpu_server is not None else settings.platform_eval_gpu_server
        ),
        "gpu_device_ids": list(
            gpu_device_ids
            if gpu_device_ids is not None
            else settings.platform_eval_gpu_device_ids
        ),
        "official_fixed_profile": True,
    }


def _gpu_runner_command(gpu_count: int) -> tuple[str, ...]:
    if gpu_count < 1:
        raise ValueError("GPU evaluator mode requires at least 1 GPU")
    if gpu_count > 8:
        raise ValueError("GPU evaluator mode supports at most 8 GPUs")
    return (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def _token_budget(mode: ExecutionMode) -> int:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return LOCAL_SMOKE_TOKEN_BUDGET
    if mode is ExecutionMode.GPU_PROXY_EVAL:
        return GPU_PROXY_TOKEN_TARGET
    return FULL_SCALE_PHASE_1_TOKEN_TARGET


def _parameter_target(mode: ExecutionMode, settings: PrismSettings) -> int:
    if mode is ExecutionMode.FULL_SCALE_EVAL:
        return min(settings.max_parameters, FULL_SCALE_PHASE_2_PARAMETER_TARGET)
    return settings.max_parameters


def _subset_for_mode(mode: ExecutionMode) -> str:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return "local_cpu_smoke_fixture"
    if mode is ExecutionMode.GPU_PROXY_EVAL:
        return "sample-10BT"
    return "sample-100BT"


def _dataset_token_count(mode: ExecutionMode) -> int:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return LOCAL_SMOKE_TOKEN_BUDGET
    return int(FINEWEB_EDU_SUBSETS[_subset_for_mode(mode)]["token_count"])


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _artifact_reference(path: Path, digest: str, content_type: str) -> dict[str, Any]:
    return {
        "path": f"artifacts/{path.name}",
        "sha256": digest,
        "content_type": content_type,
        "bytes": path.stat().st_size,
    }
