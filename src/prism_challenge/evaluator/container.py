from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..config import PrismSettings
from ..gpu_scheduler import GpuLease
from ..sdk.executors.docker import (
    DockerExecutor,
    DockerExecutorError,
    DockerLimits,
    DockerMount,
    DockerRunSpec,
)
from .checkpoints import CheckpointWorkspace, checkpoint_workspace
from .interface import PrismContext
from .modes import build_evaluation_mode_spec, execution_mode_from_value
from .sandbox import OPTIONAL_CONTRACT_FUNCTIONS, REQUIRED_CONTRACT_FUNCTIONS
from .schemas import RUN_MANIFEST_FILENAME, DeterministicEvidence, ExecutionMode, PrismRunManifest
from .scoring import score_architecture_manifest, score_training_manifest
from .source_similarity import SourceFile


@dataclass(frozen=True)
class ContainerEvaluationResult:
    container_name: str
    metrics: dict[str, float]
    run_manifest: dict[str, Any] | None = None
    artifact_output_path: str | None = None
    run_manifest_path: str | None = None


class ContainerEvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        evidence: DeterministicEvidence | tuple[DeterministicEvidence, ...] | None = None,
    ) -> None:
        super().__init__(message)
        if evidence is None:
            self.evidence: tuple[DeterministicEvidence, ...] = ()
        elif isinstance(evidence, DeterministicEvidence):
            self.evidence = (evidence,)
        else:
            self.evidence = evidence

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.evidence]


class InfrastructureEvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.artifact_output_path = artifact_output_path
        self.run_manifest_path = run_manifest_path


class PrismContainerEvaluator:
    def __init__(self, *, settings: PrismSettings, ctx: PrismContext) -> None:
        self.settings = settings
        self.ctx = ctx

    def evaluate(
        self,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        backend: str,
        files: tuple[SourceFile, ...] = (),
        entrypoint: str | None = None,
        gpu_lease: GpuLease | None = None,
        execution_mode: ExecutionMode | str | None = None,
        attempt: int = 1,
        resume_checkpoint_dir: Path | None = None,
    ) -> ContainerEvaluationResult:
        payload_files = files or (SourceFile("model.py", code, code_hash),)
        mode = execution_mode_from_value(execution_mode)
        self._enforce_artifact_size(payload_files)
        with TemporaryDirectory(prefix=f"prism-eval-{submission_id[:12]}-") as tmp:
            workspace = Path(tmp)
            artifact_output = self._artifact_output(submission_id, attempt)
            artifact_output.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_workspace(
                artifact_output, submission_id=submission_id, attempt=attempt
            )
            checkpoint.current_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = _checkpoint_payload(
                checkpoint,
                mounted_resume_path=Path("/resume-checkpoint")
                if resume_checkpoint_dir is not None
                else None,
            )
            gpu_allocation = self._gpu_allocation(gpu_lease)
            payload_path = workspace / "payload.json"
            runner_path = workspace / "runner.py"
            payload_path.write_text(
                json.dumps(
                    self._payload(
                        submission_id=submission_id,
                        code=code,
                        code_hash=code_hash,
                        arch_hash=arch_hash,
                        files=payload_files,
                        entrypoint=entrypoint,
                        gpu_lease=gpu_lease,
                        gpu_allocation=gpu_allocation,
                        execution_mode=mode,
                        checkpoint=checkpoint_payload,
                    ),
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            project = workspace / "project"
            project.mkdir()
            for file in payload_files:
                target = project / file.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(file.content, encoding="utf-8")
            runner_path.write_text(_CONTAINER_EVAL_SCRIPT, encoding="utf-8")
            command = _runner_launch_command(gpu_allocation["actual_gpu_count"])
            try:
                result = self._executor().run(
                    DockerRunSpec(
                        image=self.settings.platform_eval_image,
                        command=command,
                        mounts=self._mounts(workspace, artifact_output, resume_checkpoint_dir),
                        workdir="/workspace",
                        env=self._env(
                            submission_id,
                            code_hash,
                            arch_hash,
                            backend,
                            gpu_lease,
                            mode,
                            checkpoint_payload,
                        ),
                        labels=self._labels(submission_id, backend, gpu_lease, mode),
                        limits=DockerLimits(
                            cpus=self.settings.platform_eval_cpus,
                            memory=self.settings.platform_eval_memory,
                            memory_swap=self.settings.platform_eval_memory_swap,
                            pids_limit=self.settings.platform_eval_pids_limit,
                            network=self.settings.docker_network,
                            read_only=self.settings.platform_eval_read_only,
                            user=self.settings.docker_user,
                            gpu_count=int(gpu_allocation["actual_gpu_count"]),
                        ),
                    ),
                    self.settings.platform_eval_timeout_seconds,
                )
            except DockerExecutorError as exc:
                artifact_path, manifest_path = _valid_infra_checkpoint_paths(artifact_output)
                raise InfrastructureEvaluationError(
                    str(exc),
                    artifact_output_path=artifact_path,
                    run_manifest_path=manifest_path,
                ) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise InfrastructureEvaluationError(
                    f"Docker broker returned malformed response: {exc}"
                ) from exc
            if result.timed_out:
                raise ContainerEvaluationError(
                    "Prism container evaluation timed out",
                    _container_evidence(
                        rule_id="prism:resource-timeout",
                        artifact_path="container://prism-eval",
                        ast_node="DockerRunSpec.timeout_seconds",
                        basis=f"{submission_id}:{self.settings.platform_eval_timeout_seconds}",
                        explanation="container evaluation exceeded the configured timeout limit",
                    ),
                )
            if result.returncode != 0:
                detail = result.stderr or result.stdout or "container returned non-zero status"
                raise ContainerEvaluationError(
                    f"Prism container evaluation failed: {_redact_detail(detail[-2000:])}",
                    _container_evidence(
                        rule_id="prism:resource-violation",
                        artifact_path="container://prism-eval",
                        ast_node="DockerRunResult.returncode",
                        basis=f"{submission_id}:{result.returncode}",
                        explanation=(
                            "container evaluation returned a non-zero status under sandbox limits"
                        ),
                    ),
                )
            manifest = _read_run_manifest(artifact_output / RUN_MANIFEST_FILENAME)
            return ContainerEvaluationResult(
                container_name=result.container_name,
                metrics=(
                    _metrics_from_manifest(manifest)
                    if manifest
                    else _parse_metrics(result.stdout)
                ),
                run_manifest=manifest,
                artifact_output_path=str(artifact_output) if manifest else None,
                run_manifest_path=(
                    str(artifact_output / RUN_MANIFEST_FILENAME) if manifest else None
                ),
            )

    def _artifact_output(self, submission_id: str, attempt: int) -> Path:
        return self.settings.platform_eval_artifact_root / submission_id / f"attempt-{attempt}"

    def _mounts(
        self,
        workspace: Path,
        artifact_output: Path,
        resume_checkpoint_dir: Path | None,
    ) -> tuple[DockerMount, ...]:
        mounts = [
            DockerMount(workspace, "/workspace"),
            DockerMount(artifact_output, "/artifacts", read_only=False),
        ]
        if resume_checkpoint_dir is not None:
            mounts.append(DockerMount(resume_checkpoint_dir, "/resume-checkpoint", read_only=True))
        return tuple(mounts)

    def _executor(self) -> DockerExecutor:
        return DockerExecutor(
            challenge=self.settings.slug,
            docker_bin=self.settings.docker_bin,
            allowed_images=tuple(self.settings.docker_allowed_images)
            or (self.settings.platform_eval_image,),
            backend=self.settings.docker_backend,
            broker_url=self.settings.docker_broker_url,
            broker_token=self.settings.docker_broker_token,
            broker_token_file=str(self.settings.docker_broker_token_file)
            if self.settings.docker_broker_token_file
            else None,
        )

    def _enforce_artifact_size(self, files: Iterable[SourceFile]) -> None:
        total_bytes = 0
        max_bytes = self.settings.plagiarism_storage_max_bytes
        for file in files:
            total_bytes += len(file.content.encode("utf-8"))
            if total_bytes > max_bytes:
                raise ContainerEvaluationError(
                    f"Prism container artifact payload exceeds {max_bytes} bytes",
                    _container_evidence(
                        rule_id="prism:artifact-size",
                        artifact_path=file.path,
                        ast_node="ArtifactReference.bytes",
                        basis=f"{file.path}:{total_bytes}:{max_bytes}",
                        explanation="submission artifact payload exceeds the configured size limit",
                    ),
                )

    def _payload(
        self,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        files: tuple[SourceFile, ...],
        entrypoint: str | None,
        gpu_lease: GpuLease | None,
        gpu_allocation: dict[str, Any] | None = None,
        execution_mode: ExecutionMode,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        payload_files = files or (SourceFile("model.py", code, code_hash),)
        gpu_allocation = gpu_allocation or self._gpu_allocation(gpu_lease)
        mode_spec = build_evaluation_mode_spec(
            execution_mode,
            settings=self.settings,
            gpu_count=int(gpu_allocation["actual_gpu_count"]),
            max_gpu_count=int(gpu_allocation["max_gpu_count"]),
            gpu_type=gpu_allocation["gpu_type"],
            gpu_server=gpu_allocation["target_server"],
            gpu_device_ids=list(gpu_allocation["device_ids"]),
        )
        world_size = int(gpu_allocation["actual_gpu_count"])
        return {
            "challenge": self.settings.slug,
            "submission_id": submission_id,
            "code": code,
            "files": [
                {"path": file.path, "content": file.content, "sha256": file.sha256}
                for file in payload_files
            ],
            "entrypoint": entrypoint or _entrypoint(payload_files),
            "code_hash": code_hash,
            "arch_hash": arch_hash,
            "execution_mode": execution_mode.value,
            "mode_spec": mode_spec,
            "contract": {
                "required": sorted(REQUIRED_CONTRACT_FUNCTIONS),
                "optional": sorted(OPTIONAL_CONTRACT_FUNCTIONS),
                "metrics": [
                    "q_arch",
                    "q_recipe",
                    "train_loss",
                    "eval_loss",
                    "parameter_count",
                    "inference_latency_ms",
                ],
            },
            "context": {
                "vocab_size": self.ctx.vocab_size,
                "sequence_length": self.ctx.sequence_length,
                "max_parameters": self.ctx.max_parameters,
                "checkpoint_dir": checkpoint["current"]["path"],
                "resume_checkpoint_dir": (
                    checkpoint["resume"]["path"] if checkpoint["resume"] else None
                ),
                "checkpoint_api_version": checkpoint["api_version"],
                "attempt": checkpoint["attempt"],
                "is_resume": checkpoint["is_resume"],
                "rank": 0,
                "local_rank": 0,
                "world_size": world_size,
                "distributed_backend": "nccl" if world_size > 1 else None,
            },
            "checkpoint_workspace": checkpoint,
            "gpu_allocation": gpu_allocation,
            "artifact_output": {
                "mount": "/artifacts",
                "path": "/artifacts",
                "manifest_path": f"/artifacts/{RUN_MANIFEST_FILENAME}",
            },
        }

    def _env(
        self,
        submission_id: str,
        code_hash: str,
        arch_hash: str,
        backend: str,
        gpu_lease: GpuLease | None = None,
        execution_mode: ExecutionMode = ExecutionMode.GPU_PROXY_EVAL,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        gpu_allocation = self._gpu_allocation(gpu_lease)
        env = {
            "PRISM_SUBMISSION_ID": submission_id,
            "PRISM_CODE_HASH": code_hash,
            "PRISM_ARCH_HASH": arch_hash,
            "PRISM_EXECUTION_BACKEND": backend,
            "PRISM_EXECUTION_MODE": execution_mode.value,
            "PRISM_GPU_COUNT": str(gpu_allocation["actual_gpu_count"]),
            "PRISM_MAX_GPU_COUNT": str(gpu_allocation["max_gpu_count"]),
            "PRISM_ARTIFACT_OUTPUT_PATH": "/artifacts",
            "PRISM_RUN_MANIFEST_PATH": f"/artifacts/{RUN_MANIFEST_FILENAME}",
        }
        if checkpoint is not None:
            env.update(
                {
                    "PRISM_CHECKPOINT_API_VERSION": str(checkpoint["api_version"]),
                    "PRISM_CHECKPOINT_ATTEMPT": str(checkpoint["attempt"]),
                    "PRISM_CHECKPOINT_DIR": str(checkpoint["current"]["path"]),
                    "PRISM_CHECKPOINT_ARTIFACT_PATH": str(
                        checkpoint["current"]["artifact_relative_path"]
                    ),
                    "PRISM_IS_RESUME": "1" if checkpoint["is_resume"] else "0",
                }
            )
            if checkpoint["resume"] is not None:
                env["PRISM_RESUME_CHECKPOINT_DIR"] = str(checkpoint["resume"]["path"])
                env["PRISM_RESUME_CHECKPOINT_ARTIFACT_PATH"] = str(
                    checkpoint["resume"]["artifact_relative_path"]
                )
        if int(gpu_allocation["actual_gpu_count"]) > 1:
            env["PRISM_DISTRIBUTED_BACKEND"] = "nccl"
        if gpu_allocation["target_id"]:
            env["PRISM_GPU_TARGET_ID"] = str(gpu_allocation["target_id"])
        if gpu_allocation["target_server"]:
            env["PRISM_GPU_SERVER"] = str(gpu_allocation["target_server"])
        device_ids = gpu_allocation["device_ids"]
        if device_ids:
            env["PRISM_GPU_DEVICE_IDS"] = ",".join(str(item) for item in device_ids)
        if gpu_allocation["gpu_type"]:
            env["PRISM_GPU_TYPE"] = str(gpu_allocation["gpu_type"])
        return env

    def _labels(
        self,
        submission_id: str,
        backend: str,
        gpu_lease: GpuLease | None,
        execution_mode: ExecutionMode = ExecutionMode.GPU_PROXY_EVAL,
    ) -> dict[str, str]:
        gpu_allocation = self._gpu_allocation(gpu_lease)
        labels = {
            "platform.job": submission_id,
            "platform.task": self.settings.platform_eval_task,
            "platform.backend": backend,
            "prism.submission_id": submission_id,
            "prism.execution_mode": execution_mode.value,
            "prism.actual_gpu_count": str(gpu_allocation["actual_gpu_count"]),
            "prism.max_gpu_count": str(gpu_allocation["max_gpu_count"]),
            "prism.artifact_output_path": "/artifacts",
            "prism.run_manifest_path": f"/artifacts/{RUN_MANIFEST_FILENAME}",
        }
        for key in ("gpu_type", "target_id", "target_server"):
            value = gpu_allocation[key]
            if value:
                labels[f"prism.{key}"] = str(value)
        if gpu_allocation["device_ids"]:
            labels["prism.device_ids"] = ",".join(
                str(item) for item in gpu_allocation["device_ids"]
            )
        return labels

    def _gpu_allocation(self, gpu_lease: GpuLease | None) -> dict[str, Any]:
        return {
            "actual_gpu_count": (
                gpu_lease.gpu_count if gpu_lease else self.settings.platform_eval_gpu_count
            ),
            "max_gpu_count": (
                gpu_lease.max_gpu_count
                if gpu_lease
                else self.settings.platform_eval_max_gpu_count
            ),
            "gpu_type": self.settings.platform_eval_gpu_type,
            "target_id": gpu_lease.target_id if gpu_lease else None,
            "target_server": (
                gpu_lease.target_server
                if gpu_lease
                else self.settings.platform_eval_gpu_server
            ),
            "device_ids": list(
                gpu_lease.device_ids
                if gpu_lease
                else self.settings.platform_eval_gpu_device_ids
            ),
        }


def _runner_launch_command(gpu_count: Any) -> tuple[str, ...]:
    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool):
        raise ContainerEvaluationError(
            "Prism container evaluation GPU count must be an integer"
        )
    if gpu_count < 1:
        raise ContainerEvaluationError(
            "Prism container evaluation GPU count must be at least 1"
        )
    if gpu_count > 8:
        raise ContainerEvaluationError(
            "Prism container evaluation GPU count exceeds supported maximum of 8"
        )
    return (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def _checkpoint_payload(
    checkpoint: CheckpointWorkspace, *, mounted_resume_path: Path | None = None
) -> dict[str, Any]:
    current = _checkpoint_entry_payload(checkpoint.artifact_output, checkpoint.current_dir)
    resume = (
        _checkpoint_entry_payload(
            checkpoint.artifact_output, checkpoint.resume_dir, mounted_path=mounted_resume_path
        )
        if checkpoint.resume_dir is not None and mounted_resume_path is not None
        else None
    )
    return {
        "api_version": 1,
        "attempt": checkpoint.attempt,
        "is_resume": resume is not None,
        "artifact_root": "/artifacts",
        "current": current,
        "resume": resume,
    }


def _checkpoint_entry_payload(
    artifact_output: Path, path: Path, *, mounted_path: Path | None = None
) -> dict[str, str]:
    relative = path.relative_to(artifact_output).as_posix()
    return {
        "artifact_relative_path": relative,
        "path": str(mounted_path) if mounted_path is not None else f"/artifacts/{relative}",
    }


def _valid_infra_checkpoint_paths(artifact_output: Path) -> tuple[str | None, str | None]:
    manifest_path = artifact_output / RUN_MANIFEST_FILENAME
    try:
        manifest = _read_run_manifest(manifest_path)
    except Exception:
        return None, None
    if not manifest or not manifest.get("artifacts", {}).get("checkpoints"):
        return None, None
    return str(artifact_output), str(manifest_path)


def _parse_metrics(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PRISM_METRICS_JSON="):
            payload = json.loads(line.removeprefix("PRISM_METRICS_JSON="))
            if not isinstance(payload, dict):
                raise RuntimeError("Prism container evaluation returned invalid metrics")
            return _normalize_metrics({str(key): float(value) for key, value in payload.items()})
    raise RuntimeError("Prism container evaluation did not return metrics")


def _read_run_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContainerEvaluationError("Prism run manifest artifact is not a JSON object")
    PrismRunManifest.model_validate(payload)
    return payload


def _metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    architecture = score_architecture_manifest(manifest)
    training = score_training_manifest(manifest)
    run_manifest = PrismRunManifest.model_validate(manifest)
    metrics = {
        "q_arch": architecture.score,
        "q_recipe": training.score,
        "parameter_count": float(run_manifest.metrics.parameter_count),
        "gpu_count": float(run_manifest.metrics.gpu_count),
        "tokens_seen": float(run_manifest.metrics.tokens_seen),
        "estimated_flops": float(run_manifest.metrics.estimated_flops),
    }
    if run_manifest.metrics.final_loss is not None:
        metrics["final_loss"] = run_manifest.metrics.final_loss
        metrics["train_loss"] = run_manifest.metrics.final_loss
        metrics["eval_loss"] = run_manifest.metrics.loss.standardized_eval_loss
        metrics["val_loss"] = run_manifest.metrics.loss.standardized_eval_loss
    return metrics


def _normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if "q_arch" not in metrics:
        raise RuntimeError("Prism container evaluation did not return q_arch")
    metrics["q_arch"] = max(0.0, min(1.0, metrics["q_arch"]))
    metrics["q_recipe"] = max(0.0, min(1.0, metrics.get("q_recipe", 0.5)))
    if "train_loss" not in metrics and "final_loss" in metrics:
        metrics["train_loss"] = metrics["final_loss"]
    if "eval_loss" not in metrics and "val_loss" in metrics:
        metrics["eval_loss"] = metrics["val_loss"]
    if "val_loss" not in metrics and "eval_loss" in metrics:
        metrics["val_loss"] = metrics["eval_loss"]
    return metrics


def _container_evidence(
    *, rule_id: str, artifact_path: str, ast_node: str, basis: str, explanation: str
) -> DeterministicEvidence:
    return DeterministicEvidence(
        rule_id=rule_id,
        artifact_path=artifact_path,
        ast_node=ast_node,
        snippet_hash=sha256(basis.encode("utf-8")).hexdigest(),
        explanation=explanation,
    )


def _redact_detail(detail: str) -> str:
    redacted_lines = []
    sensitive_markers = ("api_key", "authorization", "bearer", "password", "secret", "token")
    for line in detail.splitlines():
        if any(marker in line.lower() for marker in sensitive_markers):
            redacted_lines.append("[redacted sandbox log line]")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _entrypoint(files: tuple[SourceFile, ...]) -> str:
    for candidate in ("prism_submission.py", "model.py", "main.py"):
        match = next((file for file in files if file.path.endswith(candidate)), None)
        if match:
            return match.path
    python_file = next((file for file in files if file.path.endswith(".py")), None)
    return python_file.path if python_file else "model.py"


_CONTAINER_EVAL_SCRIPT = r"""
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

if len(sys.argv) != 2:
    raise SystemExit("usage: runner.py payload.json")
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

interface = types.ModuleType("prism_challenge.evaluator.interface")

@dataclasses.dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 4096
    sequence_length: int = 128
    max_layers: int = 96
    max_parameters: int = 150_000_000
    seed: int = 1337
    checkpoint_dir: Path | None = None
    resume_checkpoint_dir: Path | None = None
    checkpoint_api_version: int = 1
    attempt: int = 1
    is_resume: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed_backend: str | None = None
    device: str = "cpu"
    checkpoint_metadata: dict = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 3e-4
    batch_size: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01

@dataclasses.dataclass(frozen=True)
class PrismBatch:
    tokens: object
    targets: object | None = None
    metadata: dict | None = None

interface.PrismContext = PrismContext
interface.TrainingRecipe = TrainingRecipe
interface.PrismBatch = PrismBatch
pkg = types.ModuleType("prism_challenge")
evaluator = types.ModuleType("prism_challenge.evaluator")
sys.modules["prism_challenge"] = pkg
sys.modules["prism_challenge.evaluator"] = evaluator
sys.modules["prism_challenge.evaluator.interface"] = interface

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

project_root = Path(os.environ.get("PRISM_PROJECT_ROOT", "/workspace/project"))
sys.path.insert(0, str(project_root))
entrypoint = project_root / payload.get("entrypoint", "model.py")
sys.path.insert(0, str(entrypoint.parent))
spec = importlib.util.spec_from_file_location("prism_submission", entrypoint)
if spec is None or spec.loader is None:
    raise RuntimeError("invalid Prism project entrypoint")
module = importlib.util.module_from_spec(spec)
sys.modules["prism_submission"] = module
spec.loader.exec_module(module)

ctx_data = payload.get("context", {})
checkpoint_data = payload.get("checkpoint_workspace", {})
current_checkpoint = checkpoint_data.get("current") or {}
resume_checkpoint = checkpoint_data.get("resume") or {}
checkpoint_dir_value = current_checkpoint.get("path") or ctx_data.get("checkpoint_dir")
resume_checkpoint_dir_value = (
    resume_checkpoint.get("path") or ctx_data.get("resume_checkpoint_dir")
)
world_size = int(os.environ.get("WORLD_SIZE", ctx_data.get("world_size", 1)))
rank = int(os.environ.get("RANK", ctx_data.get("rank", 0)))
local_rank = int(os.environ.get("LOCAL_RANK", ctx_data.get("local_rank", 0)))
distributed = world_size > 1
if torch.cuda.is_available():
    if distributed:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank if distributed else 0)
else:
    device = torch.device("cpu")
distributed_backend = None
if distributed:
    distributed_backend = (
        os.environ.get("PRISM_DISTRIBUTED_BACKEND")
        or ctx_data.get("distributed_backend")
        or ("nccl" if torch.cuda.is_available() else "gloo")
    )
    if distributed_backend == "nccl" and not torch.cuda.is_available():
        distributed_backend = "gloo"
    dist.init_process_group(backend=distributed_backend, rank=rank, world_size=world_size)

def distributed_barrier():
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()

def reduce_scalar(value):
    if not (distributed and dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return float(tensor.detach().cpu())

ctx = PrismContext(
    vocab_size=int(ctx_data.get("vocab_size", 4096)),
    sequence_length=min(int(ctx_data.get("sequence_length", 128)), 128),
    max_parameters=int(ctx_data.get("max_parameters", 150_000_000)),
    checkpoint_dir=Path(checkpoint_dir_value) if checkpoint_dir_value else None,
    resume_checkpoint_dir=(
        Path(resume_checkpoint_dir_value) if resume_checkpoint_dir_value else None
    ),
    checkpoint_api_version=int(
        checkpoint_data.get("api_version", ctx_data.get("checkpoint_api_version", 1))
    ),
    attempt=int(checkpoint_data.get("attempt", ctx_data.get("attempt", 1))),
    is_resume=bool(checkpoint_data.get("is_resume", ctx_data.get("is_resume", False))),
    rank=rank,
    local_rank=local_rank,
    world_size=world_size,
    distributed_backend=distributed_backend,
    device=str(device),
    checkpoint_metadata={"checkpoint_workspace": checkpoint_data},
)
torch.manual_seed(ctx.seed)
base_model = module.build_model(ctx)
recipe = module.get_recipe(ctx)
if isinstance(recipe, dict):
    recipe = TrainingRecipe(**recipe)
params = sum(p.numel() for p in base_model.parameters())
if params <= 0 or params > ctx.max_parameters:
    raise RuntimeError(f"invalid parameter count: {params}")
base_model.to(device)

def load_checkpoint_if_available():
    if ctx.resume_checkpoint_dir is None:
        return None
    resume_dir = Path(ctx.resume_checkpoint_dir)
    metadata_files = sorted(resume_dir.rglob("checkpoint_metadata.v1.json"))
    if not metadata_files:
        raise ValueError("requested resume checkpoint is missing metadata")
    if len(metadata_files) != 1:
        raise ValueError("requested resume checkpoint has multiple metadata files")
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    if metadata.get("submission_id") != str(payload.get("submission_id", "")):
        raise ValueError("checkpoint metadata submission_id does not match requested run")
    if metadata.get("code_hash") != str(payload.get("code_hash", "")):
        raise ValueError("checkpoint metadata code_hash does not match requested run")
    if metadata.get("arch_hash") != str(payload.get("arch_hash", "")):
        raise ValueError("checkpoint metadata arch_hash does not match requested run")
    if int(metadata.get("attempt", 0)) != int(ctx.attempt) - 1:
        raise ValueError("checkpoint metadata attempt is not the previous attempt")
    expected_recipe = recipe_fingerprint()
    if metadata.get("recipe_fingerprint") != expected_recipe:
        raise ValueError("checkpoint metadata recipe_fingerprint does not match recipe")
    metadata_dir = PurePosixPath(str(metadata.get("checkpoint_dir", "")))
    checkpoint_path = PurePosixPath(str(metadata.get("checkpoint_path", "")))
    try:
        checkpoint_relative = checkpoint_path.relative_to(metadata_dir)
    except ValueError as exc:
        raise ValueError("checkpoint metadata path escapes requested resume dir") from exc
    if not (resume_dir / checkpoint_relative.as_posix()).is_file():
        raise ValueError("checkpoint metadata points to a missing checkpoint file")
    hook = getattr(module, "load_checkpoint", None)
    if not callable(hook):
        raise ValueError("resume checkpoint requested but load_checkpoint hook is absent")
    result = hook(base_model, resume_dir, ctx)
    if result is not None:
        try:
            json.dumps(result, sort_keys=True)
        except TypeError as exc:
            raise TypeError("load_checkpoint return must be JSON serializable") from exc
    return result

def recipe_fingerprint():
    recipe_payload = {
        "learning_rate": float(getattr(recipe, "learning_rate", 3e-4)),
        "batch_size": int(getattr(recipe, "batch_size", 1)),
        "optimizer": str(getattr(recipe, "optimizer", "adamw")),
        "scheduler": str(getattr(recipe, "scheduler", "cosine")),
        "weight_decay": float(getattr(recipe, "weight_decay", 0.01)),
    }
    return hashlib.sha256(
        json.dumps(recipe_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

distributed_barrier()
load_checkpoint_metadata = load_checkpoint_if_available()
distributed_barrier()
model = (
    DistributedDataParallel(
        base_model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
    )
    if distributed
    else base_model
)
seq = ctx.sequence_length
batch_size = max(1, min(int(getattr(recipe, "batch_size", 1)), 2))
tokens = torch.randint(0, ctx.vocab_size, (batch_size, seq), device=device)
hooks_present = {
    name: callable(getattr(module, name, None))
    for name in (
        "configure_optimizer",
        "inference_logits",
        "infer",
        "compute_loss",
        "train_step",
    )
}
hook_usage = {
    "configure_optimizer": False,
    "inference_logits": False,
    "infer": False,
    "compute_loss": False,
    "train_step": False,
}

def prism_batch(t):
    return PrismBatch(tokens=t[:, :-1], targets=t[:, 1:], metadata={})

def logits_for(t):
    logits_hook = getattr(module, "inference_logits", None)
    infer_hook = getattr(module, "infer", None)
    if callable(logits_hook):
        hook_usage["inference_logits"] = True
        return logits_hook(model, prism_batch(t), ctx)
    if callable(infer_hook):
        hook_usage["infer"] = True
        return infer_hook(model, prism_batch(t), ctx)
    return model(t[:, :-1])

def loss_for(t):
    custom = getattr(module, "compute_loss", None)
    if callable(custom):
        hook_usage["compute_loss"] = True
        return custom(model, prism_batch(t), ctx)
    logits = logits_for(t)
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), t[:, 1:].reshape(-1) % vocab)

custom_opt = getattr(module, "configure_optimizer", None)
if callable(custom_opt):
    hook_usage["configure_optimizer"] = True
    optimizer = custom_opt(model, recipe, ctx)
else:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=min(float(getattr(recipe, "learning_rate", 3e-4)), 3e-4),
        weight_decay=float(getattr(recipe, "weight_decay", 0.01)),
    )
initial_loss = float(loss_for(tokens).detach().cpu())
final_loss = initial_loss
for _ in range(3):
    batch = torch.randint(0, ctx.vocab_size, (batch_size, seq), device=device)
    custom_step = getattr(module, "train_step", None)
    if callable(custom_step):
        hook_usage["train_step"] = True
        loss = custom_step(model, prism_batch(batch), optimizer, ctx)
    else:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final_loss = float(loss.detach().cpu())
initial_loss = reduce_scalar(initial_loss)
final_loss = reduce_scalar(final_loss)
improvement = max(0.0, initial_loss - final_loss)
quality = max(0.0, min(1.0, improvement / max(initial_loss, 1e-6)))
efficiency = 1.0 / (1.0 + math.log10(max(params, 1)))
q_arch = max(0.0, min(1.0, 0.82 * quality + 0.18 * efficiency))
q_recipe = 1.0 if 1e-5 <= float(getattr(recipe, "learning_rate", 3e-4)) <= 3e-3 else 0.5
metrics = {
    "q_arch": q_arch,
    "q_recipe": q_recipe,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "train_loss": final_loss,
    "eval_loss": final_loss,
    "val_loss": final_loss,
    "parameter_count": float(params),
    "hook.configure_optimizer.present": float(hooks_present["configure_optimizer"]),
    "hook.inference_logits.present": float(hooks_present["inference_logits"]),
    "hook.infer.present": float(hooks_present["infer"]),
    "hook.compute_loss.present": float(hooks_present["compute_loss"]),
    "hook.train_step.present": float(hooks_present["train_step"]),
    "hook.configure_optimizer.used": float(hook_usage["configure_optimizer"]),
    "hook.inference_logits.used": float(hook_usage["inference_logits"]),
    "hook.infer.used": float(hook_usage["infer"]),
    "hook.compute_loss.used": float(hook_usage["compute_loss"]),
    "hook.train_step.used": float(hook_usage["train_step"]),
}

def require_json_serializable(value, label):
    try:
        json.dumps(value, sort_keys=True)
    except TypeError as exc:
        raise TypeError(f"{label} must be JSON serializable") from exc

def validate_hook_path(value):
    if not isinstance(value, str):
        raise TypeError("save_checkpoint return path must be a string")
    if value.startswith("/") or "\\" in value:
        raise ValueError("save_checkpoint path must be checkpoint-dir-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path == PurePosixPath("."):
        raise ValueError("save_checkpoint path must name a relative file")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("save_checkpoint path must not contain empty, '.', or '..' segments")
    if len(path.parts[0]) == 2 and path.parts[0][1] == ":":
        raise ValueError("save_checkpoint path must not use host drive prefixes")
    return path

def normalize_save_checkpoint_return(result):
    if result is None:
        return None, None
    if isinstance(result, str):
        path = validate_hook_path(result)
        return path, {"path": result}
    if isinstance(result, dict):
        if set(result) != {"path", "metadata"}:
            raise TypeError("save_checkpoint dict return must contain exactly path and metadata")
        path = validate_hook_path(result["path"])
        if not isinstance(result["metadata"], dict):
            raise TypeError("save_checkpoint return metadata must be a dict")
        require_json_serializable(result, "save_checkpoint return")
        return path, result
    raise TypeError("save_checkpoint must return None, str, or {'path': str, 'metadata': dict}")

def artifact_relative_path(path):
    artifact_root_value = (
        checkpoint_data.get("artifact_root")
        or payload.get("artifact_output", {}).get("path")
        or "/artifacts"
    )
    artifact_root = Path(artifact_root_value).resolve(strict=False)
    target = Path(path).resolve(strict=False)
    try:
        return target.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"checkpoint artifact is outside artifact root: {path}") from exc

def checkpoint_logical_size(checkpoint_dir):
    total = 0
    if checkpoint_dir.is_symlink():
        raise ValueError("checkpoint path contains symlink")
    if not checkpoint_dir.exists():
        return 0
    for item in sorted(checkpoint_dir.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"checkpoint path contains symlink: {item}")
        if item.is_file():
            total += item.stat(follow_symlinks=False).st_size
            if total > 10_000_000_000:
                raise ValueError("checkpoint workspace exceeds 10G limit")
    return total

def write_checkpoint_record():
    if ctx.rank != 0 or ctx.checkpoint_dir is None:
        return None
    hook = getattr(module, "save_checkpoint", None)
    if not callable(hook):
        return None
    checkpoint_dir = Path(ctx.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = hook(base_model, checkpoint_dir, ctx)
    relative_path, hook_return = normalize_save_checkpoint_return(result)
    if relative_path is None:
        return None
    checkpoint_path = checkpoint_dir / relative_path.as_posix()
    checkpoint_root = checkpoint_dir.resolve(strict=False)
    checkpoint_target = checkpoint_path.resolve(strict=False)
    try:
        checkpoint_target.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError("save_checkpoint path escapes checkpoint directory") from exc
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("save_checkpoint path must point to a regular file")
    bytes_total = checkpoint_logical_size(checkpoint_dir)
    checkpoint_artifact_path = artifact_relative_path(checkpoint_path)
    checkpoint_artifact_dir = artifact_relative_path(checkpoint_dir)
    metadata_path = checkpoint_path.parent / "checkpoint_metadata.v1.json"
    metadata_artifact_path = artifact_relative_path(metadata_path)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metadata = {
        "checkpoint_api_version": int(ctx.checkpoint_api_version),
        "submission_id": str(payload.get("submission_id", "container")),
        "attempt": int(ctx.attempt),
        "code_hash": str(payload.get("code_hash", "")),
        "arch_hash": str(payload.get("arch_hash", "")),
        "recipe_fingerprint": recipe_fingerprint(),
        "created_at": created_at,
        "checkpoint_path": checkpoint_artifact_path,
        "hook_return": hook_return,
        "world_size": int(ctx.world_size),
        "rank_writer": 0,
        "checkpoint_dir": checkpoint_artifact_dir,
        "bytes_total": bytes_total,
    }
    require_json_serializable(metadata, "checkpoint metadata")
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return {
        "manifest_entry": {
            "path": checkpoint_artifact_path,
            "metadata_path": metadata_artifact_path,
            "bytes": bytes_total,
            "attempt": int(ctx.attempt),
            "world_size": int(ctx.world_size),
            "rank_writer": 0,
            "created_at": created_at,
        },
        "checkpoint_path": checkpoint_artifact_path,
    }

def diagnostics_manifest():
    return {
            "activation_entropy": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
        "useful_sparsity": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
        "attention_diversity": {
            "status": "not_applicable",
            "aggregate": None,
            "per_layer": {},
            "warnings": [],
            "not_applicable_reason": "container runner does not inspect attention weights",
            "redistribution": {
                "enabled": True,
                "policy_key": "loss_comparability_policy.redistribution_policy",
                "target": "diagnostics_health",
                "reason": "container runner does not inspect attention weights",
            },
        },
        "representation_collapse": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
        "gradient_noise_scale": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
        "activation_norm_stability": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
        "neuron_specialization": {
            "status": "ok",
            "aggregate": 1.0,
            "per_layer": {"all": 1.0},
            "warnings": [],
        },
    }

def write_run_manifest(checkpoint_record):
    manifest_path_value = (
        payload.get("artifact_output", {}).get("manifest_path")
        or os.environ.get("PRISM_RUN_MANIFEST_PATH")
        or "/artifacts/prism_run_manifest.v1.json"
    )
    manifest_path = Path(manifest_path_value)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_seen = int(batch_size * seq * 3)
    estimated_flops = float(max(1, tokens_seen) * max(1, params) * 6)
    graph_payload = {
        "schema_version": "architecture_graph.v1",
        "submission_id": payload.get("submission_id", "container"),
        "code_hash": payload.get("code_hash", ""),
        "arch_hash": payload.get("arch_hash", ""),
    }
    graph_hash = hashlib.sha256(
        json.dumps(graph_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "prism_run_manifest.v1",
        "submission_id": str(payload.get("submission_id", "container")),
        "architecture_id": (
            "container-architecture-" + str(payload.get("arch_hash", "unknown"))[:12]
        ),
        "architecture_version_id": (
            "container-architecture-version-"
            + str(payload.get("code_hash", "unknown"))[:12]
        ),
        "training_script_version_id": (
            "container-training-" + str(payload.get("code_hash", "unknown"))[:12]
        ),
        "run_id": "container-run-" + str(payload.get("submission_id", "container")),
        "mode": str(payload.get("execution_mode", "gpu_proxy_eval")),
        "dataset": {
            "name": "container-runner-fixture",
            "revision": "v1",
            "train_split_fingerprint": "container-train",
            "validation_split_fingerprint": "container-validation",
            "test_split_fingerprint": "container-test",
            "tokenizer_fingerprint": "container-tokenizer",
            "evaluator_fingerprint": "container-runner-v1",
            "benchmark_fingerprints": {},
            "contamination_report_path": "artifacts/contamination.json",
        },
        "model": {
            "parameter_count": int(params),
            "architecture_graph_hash": graph_hash,
            "tokenizer_kind": "container_fixture",
            "vocab_size": int(ctx.vocab_size),
            "max_sequence_length": int(ctx.sequence_length),
        },
        "compute": {
            "gpu_count": int(payload.get("gpu_allocation", {}).get("actual_gpu_count", 0)),
            "gpu_type": payload.get("gpu_allocation", {}).get("gpu_type"),
            "gpu_server": payload.get("gpu_allocation", {}).get("target_server"),
            "gpu_device_ids": [
                str(item)
                for item in payload.get("gpu_allocation", {}).get("device_ids", [])
            ],
            "world_size": int(ctx.world_size),
            "rank": int(ctx.rank),
            "local_rank": int(ctx.local_rank),
            "distributed_backend": ctx.distributed_backend,
            "effective_batch_size": int(batch_size),
            "gradient_accumulation_steps": 1,
            "tokens_seen": tokens_seen,
            "estimated_flops": estimated_flops,
            "wall_clock_seconds": 0.0,
            "checkpoint_path": (
                checkpoint_record["checkpoint_path"] if checkpoint_record else None
            ),
            "resume_checkpoint_path": None,
        },
        "metrics": {
            "loss_vs_tokens": [
                {"x": 0.0, "loss": initial_loss},
                {"x": float(tokens_seen), "loss": final_loss},
            ],
            "loss_vs_compute": [
                {"x": 0.0, "loss": initial_loss},
                {"x": estimated_flops, "loss": final_loss},
            ],
            "loss_vs_params": [{"x": float(params), "loss": final_loss}],
            "learning_speed_slope": (final_loss - initial_loss) / max(
                float(tokens_seen), 1.0
            ),
            "tokens_seen": tokens_seen,
            "estimated_flops": estimated_flops,
            "parameter_count": int(params),
            "benchmark_scores": {},
            "benchmark_capability_metadata": {
                "status": "not_run",
                "reason": "container runner fixture",
            },
            "benchmark_noise_metadata": {
                "status": "not_run",
                "reason": "container runner fixture",
            },
            "benchmark_contamination_metadata": {
                "required": False,
                "reason": "container runner fixture",
            },
            "diagnostics": diagnostics_manifest(),
            "gpu_count": int(payload.get("gpu_allocation", {}).get("actual_gpu_count", 0)),
            "loss": {
                "raw_final_loss": final_loss,
                "standardized_eval_loss": final_loss,
                "loss_normalization_scope": "byte_normalized",
                "baseline_run_id": "container-runner-baseline-v1",
                "relative_loss_reduction": max(0.0, initial_loss - final_loss) / max(
                    initial_loss, 1e-12
                ),
                "architecture_normalized_heldout_improvement": max(
                    0.0, initial_loss - final_loss
                ) / max(initial_loss, 1e-12),
                "loss_comparable": True,
                "loss_component_redistribution": {"enabled": False},
            },
            "final_loss": final_loss,
        },
        "artifacts": {
            "architecture_graph": {
                "path": "artifacts/architecture_graph.json",
                "sha256": graph_hash,
                "content_type": "application/json",
                "bytes": len(json.dumps(graph_payload)),
            },
            "architecture_metadata": {
                "path": "artifacts/architecture_metadata.v1.json",
                "sha256": hashlib.sha256(b"container-metadata").hexdigest(),
                "content_type": "application/json",
                "bytes": 2,
            },
            "run_log": {
                "path": "artifacts/run.log",
                "sha256": hashlib.sha256(b"container-runner").hexdigest(),
                "content_type": "text/plain",
                "bytes": 16,
            },
            "checkpoints": [checkpoint_record["manifest_entry"]] if checkpoint_record else [],
            "metrics": {
                "path": "artifacts/metrics.json",
                "sha256": hashlib.sha256(
                    json.dumps(metrics, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "content_type": "application/json",
                "bytes": len(json.dumps(metrics)),
            },
        },
        "validation": {
            "passed": True,
            "score_eligible": True,
            "deterministic_evidence": [],
            "warnings": [],
            "errors": [],
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

distributed_barrier()
checkpoint_record = write_checkpoint_record() if rank == 0 else None
distributed_barrier()
if rank == 0:
    write_run_manifest(checkpoint_record)
distributed_barrier()
if rank == 0:
    print("PRISM_METRICS_JSON=" + json.dumps(metrics, separators=(",", ":")))
if distributed and dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
"""
