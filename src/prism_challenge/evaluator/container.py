from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
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
from .heldout import HeldoutResult, compute_heldout_metrics
from .interface import PrismContext
from .modes import execution_mode_from_value
from .schemas import RUN_MANIFEST_V2_FILENAME, DeterministicEvidence, ExecutionMode
from .scoring import ScoreValidationError, build_compute_block, score_prequential_bpb
from .source_similarity import SourceFile

DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500
# Name of the trained-weights artifact the in-container runner writes and the HOST scorer loads to
# compute the held-out delta on the SECRET val split (kept in sync with the runner script's
# ``TRAINED_STATE_FILENAME``).
TRAINED_STATE_ARTIFACT = "trained_state.pt"
# Markers the in-container runner watchdog prints on stderr so the host can classify a non-zero
# exit into a precise terminal reason (kept in sync with the runner script).
BUDGET_EXCEEDED_MARKER = "PRISM_RUNNER_BUDGET_EXCEEDED"
ARTIFACTS_QUOTA_MARKER = "PRISM_RUNNER_ARTIFACTS_QUOTA"
# Substrings that indicate a host-RAM / GPU-VRAM / OOM exhaustion of the eval container.
_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "oomkilled",
    "killed process",
    "cannot allocate memory",
    "std::bad_alloc",
)
# SIGKILL-derived exit code (128 + 9): the classic cgroup OOM-kill / hard-cap signal.
_OOM_EXIT_CODE = 137


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
    """Validator re-execution harness (architecture.md section 4.3).

    Writes a challenge-owned ``runner.py`` that FORCES the seed + deterministic flags before any
    miner code runs, imports the miner's two-script bundle (``architecture.py::build_model`` +
    ``training.py::train``), and launches ``torchrun --standalone --nnodes=1 --nproc-per-node=N``
    with a loopback rendezvous. The scored run trains on the LOCKED FineWeb-Edu train split; a
    missing/empty locked data path fails fast (no random-token fallback) and any miner-written
    manifest is ignored.
    """

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
        architecture_entrypoint: str | None = None,
        training_entrypoint: str | None = None,
        build_model_symbol: str = "build_model",
        train_symbol: str = "train",
        gpu_lease: GpuLease | None = None,
        execution_mode: ExecutionMode | str | None = None,
        attempt: int = 1,
    ) -> ContainerEvaluationResult:
        payload_files = files or (SourceFile("architecture.py", code, code_hash),)
        mode = execution_mode_from_value(execution_mode)
        self._enforce_artifact_size(payload_files)
        arch_entry = architecture_entrypoint or _default_entrypoint(payload_files, "architecture")
        train_entry = training_entrypoint or _default_entrypoint(payload_files, "training")
        with TemporaryDirectory(prefix=f"prism-eval-{submission_id[:12]}-") as tmp:
            workspace = Path(tmp)
            artifact_output = self._fresh_artifact_output(submission_id, attempt)
            gpu_allocation = self._gpu_allocation(gpu_lease)
            payload_path = workspace / "payload.json"
            runner_path = workspace / "runner.py"
            payload_path.write_text(
                json.dumps(
                    self._payload(
                        submission_id=submission_id,
                        code_hash=code_hash,
                        arch_hash=arch_hash,
                        files=payload_files,
                        architecture_entrypoint=arch_entry,
                        training_entrypoint=train_entry,
                        build_model_symbol=build_model_symbol,
                        train_symbol=train_symbol,
                        gpu_allocation=gpu_allocation,
                        execution_mode=mode,
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
                        mounts=self._mounts(workspace, artifact_output),
                        workdir="/workspace",
                        env=self._env(
                            submission_id,
                            code_hash,
                            arch_hash,
                            backend,
                            gpu_lease,
                            mode,
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
                    self.settings.platform_eval_hard_timeout_seconds,
                )
            except DockerExecutorError as exc:
                raise InfrastructureEvaluationError(str(exc)) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise InfrastructureEvaluationError(
                    f"Docker broker returned malformed response: {exc}"
                ) from exc
            if result.timed_out:
                # The graceful budget + watchdog should normally fire first; reaching the outer
                # docker/broker cap means the loop hung past every inner budget. Stop it here.
                raise ContainerEvaluationError(
                    "Prism container evaluation exceeded the wall-clock safety cap",
                    _container_evidence(
                        rule_id="prism:budget-exceeded",
                        artifact_path="container://prism-eval",
                        ast_node="DockerRunSpec.timeout_seconds",
                        basis=f"{submission_id}:{self.settings.platform_eval_hard_timeout_seconds}",
                        explanation="container evaluation exceeded the wall-clock budget cap",
                    ),
                )
            if result.returncode != 0:
                detail = result.stderr or result.stdout or "container returned non-zero status"
                rule_id, explanation = _classify_failure(detail, result.returncode)
                raise ContainerEvaluationError(
                    f"Prism container evaluation failed: {_redact_detail(detail[-2000:])}",
                    _container_evidence(
                        rule_id=rule_id,
                        artifact_path="container://prism-eval",
                        ast_node="DockerRunResult.returncode",
                        basis=f"{submission_id}:{result.returncode}",
                        explanation=explanation,
                    ),
                )
            manifest = _read_run_manifest(artifact_output / RUN_MANIFEST_V2_FILENAME)
            if manifest is not None:
                _ensure_compute_block(manifest, gpu_allocation, artifact_output)
                manifest = self._augment_with_heldout(
                    manifest,
                    files=payload_files,
                    architecture_entrypoint=arch_entry,
                    build_model_symbol=build_model_symbol,
                    artifact_output=artifact_output,
                )
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
                    str(artifact_output / RUN_MANIFEST_V2_FILENAME) if manifest else None
                ),
            )

    def _fresh_artifact_output(self, submission_id: str, attempt: int) -> Path:
        """A fresh artifacts dir per run; never reuse a prior run's manifest/artifacts."""
        artifact_output = (
            self.settings.platform_eval_artifact_root / submission_id / f"attempt-{attempt}"
        )
        for stale in _existing_manifests(artifact_output):
            try:
                stale.unlink()
            except OSError:
                pass
        artifact_output.mkdir(parents=True, exist_ok=True)
        return artifact_output

    def _augment_with_heldout(
        self,
        manifest: dict[str, Any],
        *,
        files: tuple[SourceFile, ...],
        architecture_entrypoint: str,
        build_model_symbol: str,
        artifact_output: Path,
    ) -> dict[str, Any]:
        """Augment the v2 manifest with the host-computed held-out delta + anti-memorization gap.

        The SECRET val split is never mounted into the eval container; the host scorer loads the
        trained weights the runner persisted, runs them and a forced-seed random-init twin on val,
        and folds the delta tie-breaker + gap penalty into the challenge-authored score block. A
        missing trained-weights artifact or unavailable val split simply skips the held-out delta
        (the run still scores on prequential bpb), so this never fails the run.

        The trained-state artifact is read ONLY when the CHALLENGE-AUTHORED manifest recorded it for
        THIS run (never a bare ``is_file()`` on the miner-writable artifacts_dir, an RCE sink), and
        the host-side held-out gap is measured on the same tokenizer basis the run trained with.
        """
        recorded = manifest.get("artifacts")
        recorded_name = recorded.get("trained_state") if isinstance(recorded, Mapping) else None
        trained_state_path = _resolve_recorded_trained_state(artifact_output, recorded_name)
        try:
            result = compute_heldout_metrics(
                files={file.path: file.content for file in files},
                entrypoint=architecture_entrypoint,
                build_model_symbol=build_model_symbol,
                ctx=self.ctx,
                trained_state_path=trained_state_path,
                val_data_dir=self.settings.platform_eval_val_data_dir,
                train_bpb=_manifest_train_bpb(manifest),
                train_bpb_basis=_manifest_train_bpb_basis(manifest),
                val_byte_budget=self.settings.platform_eval_heldout_val_byte_budget,
                timeout_seconds=self.settings.platform_eval_heldout_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - held-out is auxiliary; never fail the run on it
            return manifest
        if result is None:
            return manifest
        _merge_heldout_into_manifest(manifest, result)
        try:
            (artifact_output / RUN_MANIFEST_V2_FILENAME).write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return manifest

    def _mounts(self, workspace: Path, artifact_output: Path) -> tuple[DockerMount, ...]:
        # The locked FineWeb-Edu train split + reference tokenizers are bind-mounted READ-ONLY by
        # the broker (per-slug RO data-mount wiring); the runner reads ctx.data_dir from that mount.
        return (
            DockerMount(workspace, "/workspace"),
            DockerMount(artifact_output, "/artifacts", read_only=False),
        )

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

    def reap_job(self, submission_id: str) -> None:
        """Reap the terminal eval replicated-job/container for ``submission_id`` (best-effort).

        The Swarm backend already removes the replicated-job service in its own ``run`` finally;
        this is a defense-in-depth sweep (``docker service rm`` via the broker cleanup) so a dead
        eval service can never accumulate run-over-run even if the broker run path was interrupted
        (architecture.md sections 4.3, 10; VAL-HARNESS-027). Never raises into the caller.
        """
        try:
            self._executor().cleanup_job(submission_id)
        except Exception:  # noqa: BLE001 - reaping is best-effort; never fail a terminal run on it
            pass

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
        code_hash: str,
        arch_hash: str,
        files: tuple[SourceFile, ...],
        architecture_entrypoint: str,
        training_entrypoint: str,
        build_model_symbol: str,
        train_symbol: str,
        gpu_allocation: dict[str, Any],
        execution_mode: ExecutionMode,
    ) -> dict[str, Any]:
        world_size = int(gpu_allocation["actual_gpu_count"])
        return {
            "challenge": self.settings.slug,
            "submission_id": submission_id,
            "files": [
                {"path": file.path, "content": file.content, "sha256": file.sha256}
                for file in files
            ],
            "architecture_entrypoint": architecture_entrypoint,
            "training_entrypoint": training_entrypoint,
            "build_model_symbol": build_model_symbol,
            "train_symbol": train_symbol,
            "code_hash": code_hash,
            "arch_hash": arch_hash,
            "execution_mode": execution_mode.value,
            "master_addr": DEFAULT_MASTER_ADDR,
            "master_port": DEFAULT_MASTER_PORT,
            "context": {
                "vocab_size": self.ctx.vocab_size,
                "sequence_length": self.ctx.sequence_length,
                "max_layers": self.ctx.max_layers,
                "max_parameters": self.ctx.max_parameters,
                "seed": self.ctx.seed,
                "data_dir": self.settings.platform_eval_data_dir,
                "artifacts_dir": "/artifacts",
                "reference_tokenizer_dir": self.settings.platform_eval_reference_tokenizer_dir,
                "token_budget": self.ctx.token_budget,
                "step_budget": self.ctx.step_budget,
                "budget_seconds": self.settings.platform_eval_budget_seconds,
                "watchdog_grace_seconds": self.settings.platform_eval_watchdog_grace_seconds,
                "artifacts_quota_bytes": self.settings.platform_eval_artifacts_quota_bytes,
                "rank": 0,
                "local_rank": 0,
                "world_size": world_size,
                "distributed_backend": "nccl" if world_size > 1 else None,
            },
            "gpu_allocation": gpu_allocation,
            "artifact_output": {
                "mount": "/artifacts",
                "path": "/artifacts",
                "manifest_path": f"/artifacts/{RUN_MANIFEST_V2_FILENAME}",
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
            "PRISM_RUN_MANIFEST_PATH": f"/artifacts/{RUN_MANIFEST_V2_FILENAME}",
            "PRISM_DATA_DIR": self.settings.platform_eval_data_dir,
            "PRISM_REFERENCE_TOKENIZER_DIR": self.settings.platform_eval_reference_tokenizer_dir,
            # Loopback rendezvous so the c10d hostname lookup cannot hang (readiness B2).
            "MASTER_ADDR": DEFAULT_MASTER_ADDR,
            "MASTER_PORT": str(DEFAULT_MASTER_PORT),
        }
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
            "prism.run_manifest_path": f"/artifacts/{RUN_MANIFEST_V2_FILENAME}",
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


def _default_entrypoint(files: tuple[SourceFile, ...], role: str) -> str:
    target = f"{role}.py"
    match = next((file for file in files if file.path.endswith(target)), None)
    if match:
        return match.path
    python_file = next((file for file in files if file.path.endswith(".py")), None)
    return python_file.path if python_file else target


def _existing_manifests(artifact_output: Path) -> list[Path]:
    if not artifact_output.is_dir():
        return []
    return [path for path in artifact_output.glob("prism_run_manifest*.json") if path.is_file()]


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


def _parse_metrics(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PRISM_METRICS_JSON="):
            payload = json.loads(line.removeprefix("PRISM_METRICS_JSON="))
            if not isinstance(payload, dict):
                raise RuntimeError("Prism container evaluation returned invalid metrics")
            return {str(key): float(value) for key, value in payload.items()}
    raise RuntimeError("Prism container evaluation did not return metrics")


def _manifest_train_bpb(manifest: dict[str, Any]) -> float | None:
    """The prequential (online train) bits-per-byte from the v2 manifest, if present."""
    for key in ("metrics", "score"):
        block = manifest.get(key)
        if isinstance(block, dict):
            value = block.get("prequential_bpb", block.get("bits_per_byte"))
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def _manifest_train_bpb_basis(manifest: dict[str, Any]) -> str | None:
    """The tokenizer basis the prequential TRAIN bpb was measured on (``bytes`` / ``tokenizer``)."""
    for key in ("metrics", "score"):
        block = manifest.get(key)
        if isinstance(block, dict):
            value = block.get("train_bpb_basis")
            if isinstance(value, str) and value:
                return value
    return None


def _resolve_recorded_trained_state(artifact_output: Path, recorded: Any) -> Path | None:
    """Resolve the trained-state artifact ONLY if the CHALLENGE manifest recorded it for THIS run.

    A miner-writable artifacts_dir makes a bare ``is_file()`` an unsafe-deserialization (RCE) sink:
    the host reads only the exact challenge-authored artifact name, and only when it resolves to a
    regular file directly under this run's artifact dir (no traversal/symlink escape).
    """
    if not isinstance(recorded, str) or recorded != TRAINED_STATE_ARTIFACT:
        return None
    try:
        base = artifact_output.resolve()
        candidate = (artifact_output / recorded).resolve()
    except OSError:
        return None
    if candidate.parent != base:
        return None
    return candidate if candidate.is_file() else None


def _ensure_compute_block(
    manifest: dict[str, Any], gpu_allocation: Mapping[str, Any], artifact_output: Path
) -> None:
    """Author the authoritative observability-only ``compute`` block on the v2 manifest.

    The GPUs actually LEASED for the run (``gpu_allocation['actual_gpu_count']`` == the DB
    ``eval_jobs.actual_gpu_count``) are the source of truth for ``compute.gpu_count`` (``== 1`` for
    the scored ``nproc=1`` path), so the block is internally consistent with ``run.world_size`` /
    ``run.nproc_per_node`` and the DB for the same eval. The block is RECORDED for observability
    (VAL-GPU-005) and is never an input to ``final_score``. The reconciled manifest is persisted to
    the on-disk artifact where manifest-inspect reads it (best-effort; never fails the run).
    """
    run = manifest.get("run")
    run = run if isinstance(run, Mapping) else {}
    gpu_count = _coerce_int(gpu_allocation.get("actual_gpu_count"), default=1, minimum=0)
    world_size = _coerce_int(run.get("world_size"), default=max(gpu_count, 1), minimum=1)
    nproc_per_node = _coerce_int(run.get("nproc_per_node"), default=world_size, minimum=1)
    device = run.get("device") if isinstance(run.get("device"), str) and run.get("device") else None
    if device is None:
        device = "cuda" if gpu_count > 0 else "cpu"
    max_gpu_count = _coerce_int(gpu_allocation.get("max_gpu_count"), default=0, minimum=0) or None
    manifest["compute"] = build_compute_block(
        gpu_count=gpu_count,
        world_size=world_size,
        nproc_per_node=nproc_per_node,
        device=device,
        max_gpu_count=max_gpu_count,
    )
    try:
        (artifact_output / RUN_MANIFEST_V2_FILENAME).write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _coerce_int(value: Any, *, default: int, minimum: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(minimum, value)
    return max(minimum, default)


def _merge_heldout_into_manifest(manifest: dict[str, Any], result: HeldoutResult) -> None:
    """Fold the host-computed held-out delta + anti-memorization gap into the v2 manifest blocks.

    The metrics/anti_cheat blocks carry the raw held-out values; the ``score`` block is recomputed
    by ``scoring.score_prequential_bpb`` so ``final_score`` reflects the delta tie-break + the
    memorization penalty (single source of truth shared with the queue finalizer).
    """
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        metrics.update(result.as_metrics())
    anti_cheat = manifest.get("anti_cheat")
    if isinstance(anti_cheat, dict):
        anti_cheat["memorization_flag"] = result.memorization_flag
        if result.train_heldout_gap is not None:
            anti_cheat["memorization_gap"] = result.train_heldout_gap
            anti_cheat["train_heldout_gap"] = result.train_heldout_gap
    try:
        recomputed = score_prequential_bpb(manifest)
    except ScoreValidationError:
        return
    manifest["score"] = recomputed.manifest_score_block()


def _read_run_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContainerEvaluationError("Prism run manifest artifact is not a JSON object")
    if payload.get("schema_version") != "prism_run_manifest.v2":
        raise ContainerEvaluationError(
            "Prism run manifest artifact is not schema prism_run_manifest.v2"
        )
    return payload


def _metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    """Derive the metrics surface from the challenge-authored v2 manifest.

    The prequential bits-per-byte scoring fields are computed by the scoring recast; this feature
    surfaces only the re-execution provenance (data coverage / parameter count) so the pipeline can
    finalize the run from the challenge-owned manifest, never miner-reported numbers.
    """
    metrics: dict[str, float] = {}
    run_metrics = manifest.get("metrics")
    if isinstance(run_metrics, dict):
        for key, value in run_metrics.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[str(key)] = float(value)
    data = manifest.get("data")
    if isinstance(data, dict):
        available = data.get("available_bytes")
        if isinstance(available, int | float) and not isinstance(available, bool):
            metrics["available_bytes"] = float(available)
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


def _classify_failure(detail: str, returncode: int) -> tuple[str, str]:
    """Map a non-zero eval exit into a precise (rule_id, explanation) terminal reason.

    Distinguishes the budget watchdog, the artifacts-quota watchdog, and resource/OOM exhaustion
    (host RAM / GPU VRAM) from a generic miner runtime/import crash, so the submission lands failed
    with an accurate reason (architecture.md sections 4.3, 9).
    """
    haystack = detail.lower()
    if BUDGET_EXCEEDED_MARKER.lower() in haystack:
        return (
            "prism:budget-exceeded",
            "training loop exceeded the wall-clock budget safety cap",
        )
    if ARTIFACTS_QUOTA_MARKER.lower() in haystack:
        return (
            "prism:artifacts-quota",
            "artifacts_dir write exceeded the configured disk quota",
        )
    if returncode == _OOM_EXIT_CODE or any(marker in haystack for marker in _OOM_MARKERS):
        return (
            "prism:resource-oom",
            "container evaluation exhausted host RAM / GPU VRAM (OOM) under resource caps",
        )
    return (
        "prism:runtime-error",
        "miner training loop raised at runtime (uncaught exception / import / device error)",
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


_CONTAINER_EVAL_SCRIPT = r'''"""Challenge-owned re-execution runner (architecture.md section 4.3).

The challenge authors this runner. It FORCES global seeds + deterministic flags BEFORE importing
any miner code, resolves the LOCKED FineWeb-Edu train split, imports the miner two-script bundle
(architecture.py::build_model + training.py::train), and invokes the miner-owned train(ctx) loop
under a loopback torchrun rendezvous. A missing/empty locked data path fails fast (NO random-token
fallback); any miner-written manifest is ignored and the challenge authors
prism_run_manifest.v2.json itself.
"""
import dataclasses
import importlib.util
import json
import math
import os
import random
import sys
import threading
import time
import types
from pathlib import Path

MANIFEST_GLOB = "prism_run_manifest*.json"
CHALLENGE_MANIFEST_NAME = "prism_run_manifest.v2.json"
# The challenge persists the trained model's weights (state_dict) so the HOST scorer can compute
# the held-out delta-over-random-init on the SECRET val split, which is NEVER mounted into this
# network=none container (architecture.md sections 5, 6; VAL-HARNESS-015 / VAL-CHEAT-007).
TRAINED_STATE_FILENAME = "trained_state.pt"
# Step-0 (forced random init) loss must sit near the from-scratch baseline (~ln(vocab) nats);
# an impossibly-low initial loss is the smuggled-pretrained-weights anomaly signal.
STEP0_ANOMALY_FRACTION = 0.5
# A NaN/Inf online loss is sanitized to a worst-case (high) per-batch code-length so it can
# never collapse into a finite, advantageous score.
WORST_CASE_LOSS_MULTIPLIER = 2.0
DEFAULT_BATCH_SIZE = 8
# Watchdog stderr markers the HOST classifies a non-zero exit by (kept in sync with container.py).
BUDGET_EXCEEDED_MARKER = "PRISM_RUNNER_BUDGET_EXCEEDED"
ARTIFACTS_QUOTA_MARKER = "PRISM_RUNNER_ARTIFACTS_QUOTA"
WATCHDOG_POLL_SECONDS = 0.5
# Distinct non-zero exit codes for the two watchdog terminations (host reads the marker too).
BUDGET_EXIT_CODE = 7
ARTIFACTS_QUOTA_EXIT_CODE = 8


def _fail(reason):
    sys.stderr.write("PRISM_RUNNER_ERROR: " + reason + "\n")
    raise SystemExit("prism-runner: " + reason)


if len(sys.argv) != 2:
    _fail("usage: runner.py payload.json")
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
context_data = payload.get("context", {})
forced_seed = int(context_data.get("seed", 1337))

# --- FORCE global determinism BEFORE importing any miner code (architecture.md 4.3) ---
os.environ.setdefault("PYTHONHASHSEED", str(forced_seed))
os.environ.setdefault("MASTER_ADDR", str(payload.get("master_addr", "127.0.0.1")))
os.environ.setdefault("MASTER_PORT", str(payload.get("master_port", 29500)))
random.seed(forced_seed)

import torch


def _force_init(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


_force_init(forced_seed)
print(
    "PRISM_RUNNER: forced seed=" + str(forced_seed)
    + " + manual_seed/cuda.manual_seed_all/use_deterministic_algorithms/cudnn applied "
    + "before miner import",
    flush=True,
)

rank = int(os.environ.get("RANK", context_data.get("rank", 0)))
local_rank = int(os.environ.get("LOCAL_RANK", context_data.get("local_rank", 0)))
world_size = int(os.environ.get("WORLD_SIZE", context_data.get("world_size", 1)))
if torch.cuda.is_available():
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

artifacts_dir = (
    context_data.get("artifacts_dir")
    or os.environ.get("PRISM_ARTIFACT_OUTPUT_PATH")
    or "/artifacts"
)
Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

# Fresh per run: discard any stale/planted manifest before the miner loop begins.
if rank == 0:
    for stale in Path(artifacts_dir).glob(MANIFEST_GLOB):
        try:
            stale.unlink()
        except OSError:
            pass
    # Own the held-out artifact path from the start: a miner could have planted a hostile pickle at
    # trained_state.pt before the runner ran. Remove it so only challenge-written weights can exist.
    try:
        (Path(artifacts_dir) / TRAINED_STATE_FILENAME).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _resolve_train_shards(data_dir):
    if not data_dir:
        _fail("locked train data path is not configured (no random-token fallback)")
    base = Path(data_dir)
    if not base.is_dir():
        _fail("locked train data path is missing: " + str(data_dir) + " (no random-token fallback)")
    candidates = []
    for pattern in ("train-*.jsonl", "*.jsonl", "*.bin"):
        candidates = sorted(p for p in base.glob(pattern) if p.is_file())
        if candidates:
            break
    if not candidates and (base / "train").is_dir():
        candidates = sorted(p for p in (base / "train").glob("*.jsonl") if p.is_file())
    nonempty = [p for p in candidates if p.stat().st_size > 0]
    if not nonempty:
        _fail("locked train data is empty: " + str(data_dir) + " (no random-token fallback)")
    return nonempty


data_dir = context_data.get("data_dir") or os.environ.get("PRISM_DATA_DIR")
train_shards = _resolve_train_shards(data_dir)
available_bytes = sum(p.stat().st_size for p in train_shards)
print(
    "PRISM_RUNNER: locked FineWeb-Edu train resolved at " + str(data_dir)
    + " (" + str(len(train_shards)) + " shards, " + str(available_bytes) + " bytes)",
    flush=True,
)


@dataclasses.dataclass
class _PrismBatch:
    tokens: object
    targets: object = None
    metadata: object = None


class _OnlineLossCapture:
    """Challenge-owned predict-then-train online-loss instrument (architecture.md 4.3, 5).

    Yields fresh, single-pass batches from the locked train split in a fixed challenge-controlled
    order (no token/shard repeats within the run) and records the model's loss on each NEW batch
    BEFORE the miner's optimizer updates on it. Because the data is single-pass, this online
    training loss IS the prequential code-length by construction. The challenge computes the loss
    itself (next-token cross-entropy over the model's logits); miner-reported numbers are ignored.
    """

    def __init__(
        self,
        *,
        shards,
        vocab_size,
        seq_len,
        batch_size,
        baseline_nats,
        token_budget,
        step_budget,
        tokenizer,
        device,
        budget_seconds=None,
    ):
        self.shards = list(shards)
        self.vocab_size = max(int(vocab_size), 2)
        self.seq_len = max(int(seq_len or 0), 2)
        self.batch_size = max(int(batch_size or 1), 1)
        self.baseline_nats = baseline_nats
        self.token_budget = token_budget
        self.step_budget = step_budget
        # Graceful wall-clock budget: the loop stops here and the run is scored on the PARTIAL
        # captured stream (the compute-normalized bpb score never depends on wall-clock; this is a
        # safety cap only, architecture.md 4.3, 9).
        self.budget_seconds = budget_seconds
        self.tokenizer = tokenizer
        self.device = device
        self.worst_case_nats = baseline_nats * WORST_CASE_LOSS_MULTIPLIER
        self.online_loss = []
        self.covered_bytes_cumulative = []
        self.covered_bytes = 0.0
        self.consumed_batches = 0
        self.consumed_tokens = 0
        self.consumed_documents = 0
        self.predicted_tokens = 0
        self.shard_offsets = []
        self.nan_inf_batches = 0
        self.sum_nll_nats = 0.0
        self.started = False
        self.start_time = None
        # Which budget bound the single-pass run: token_budget | step_budget | wall_clock |
        # data_exhausted (architecture.md 4.3; VAL-HARNESS-025). ``None`` until the loop ends.
        self.stopped_reason = None
        # Handle to the model the miner trained on the instrumented batches; the challenge persists
        # its weights so the HOST scorer can compute the held-out delta on the SECRET val split
        # (never mounted into this network=none container). See architecture.md sections 5, 6.
        self.model = None

    def _encode(self, text, raw_bytes):
        tokenizer = self.tokenizer
        if tokenizer is None:
            return list(raw_bytes)
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text)
        elif callable(tokenizer):
            ids = tokenizer(text)
        else:
            return list(raw_bytes)
        return [int(token_id) for token_id in ids]

    def _token_stream(self):
        for shard in self.shards:
            shard_name = shard.name
            with shard.open("r", encoding="utf-8") as handle:
                for offset, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record["text"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    if not isinstance(text, str) or not text:
                        continue
                    raw_bytes = text.encode("utf-8")
                    ids = self._encode(text, raw_bytes)
                    if not ids:
                        continue
                    self.consumed_documents += 1
                    self.shard_offsets.append([shard_name, offset])
                    weight = len(raw_bytes) / len(ids)
                    for token_id in ids:
                        yield int(token_id), weight

    def _predict_loss(self, model, tokens):
        import torch
        import torch.nn.functional as functional

        with torch.no_grad():
            logits = model(tokens)
        if not isinstance(logits, torch.Tensor):
            if (
                isinstance(logits, (tuple, list))
                and logits
                and isinstance(logits[0], torch.Tensor)
            ):
                logits = logits[0]
            elif hasattr(logits, "logits"):
                logits = logits.logits
            else:
                raise RuntimeError("miner model forward did not return a logits tensor")
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        vocab = logits.shape[-1]
        predictions = logits[:, :-1, :].reshape(-1, vocab)
        targets = (tokens[:, 1:].reshape(-1)) % vocab
        loss = functional.cross_entropy(predictions, targets)
        return float(loss.detach().item())

    def _emit(self, model, chunk):
        import torch

        ids = [token_id for token_id, _ in chunk]
        nbytes = sum(weight for _, weight in chunk)
        count = len(ids)
        if count == self.seq_len * self.batch_size:
            rows, cols = self.batch_size, self.seq_len
        else:
            rows, cols = 1, count
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)
        tokens = torch.tensor(ids, dtype=torch.long).remainder(self.vocab_size).view(rows, cols)
        tokens = tokens.to(model_device)
        loss_value = self._predict_loss(model, tokens)
        if not math.isfinite(loss_value):
            self.nan_inf_batches += 1
            loss_value = self.worst_case_nats
        # ``loss_value`` is the MEAN next-token nats over the batch's predicted positions; the
        # prequential code-length needs the TOKEN-WEIGHTED total, so multiply by the number of
        # scored next-token predictions (rows * (cols - 1)).
        predicted = rows * max(cols - 1, 0)
        self.online_loss.append(loss_value)
        self.sum_nll_nats += loss_value * predicted
        self.predicted_tokens += predicted
        self.covered_bytes += nbytes
        self.covered_bytes_cumulative.append(self.covered_bytes)
        self.consumed_batches += 1
        return _PrismBatch(tokens=tokens)

    def _budget_reason(self):
        # First-reached budget among token, step, and wall-clock caps. Checked at every token and
        # batch boundary so the single-pass run terminates at whichever budget binds first; the
        # check order is the deterministic tie-break when several bind at the same checkpoint.
        if self.token_budget is not None and self.consumed_tokens >= self.token_budget:
            return "token_budget"
        if self.step_budget is not None and self.consumed_batches >= self.step_budget:
            return "step_budget"
        if (
            self.budget_seconds is not None
            and self.start_time is not None
            and (time.monotonic() - self.start_time) >= self.budget_seconds
        ):
            return "wall_clock"
        return None

    def iterate(self, model):
        if self.started:
            return
        self.started = True
        self.model = model
        self.start_time = time.monotonic()
        needed = self.seq_len * self.batch_size
        buffer = []
        for token_id, weight in self._token_stream():
            reason = self._budget_reason()
            if reason is not None:
                self.stopped_reason = reason
                return
            buffer.append((token_id, weight))
            self.consumed_tokens += 1
            if len(buffer) >= needed:
                yield self._emit(model, buffer[:needed])
                buffer = buffer[needed:]
                reason = self._budget_reason()
                if reason is not None:
                    self.stopped_reason = reason
                    return
        # The single-pass shards were exhausted before any budget bound: emit the trailing partial
        # (challenge-controlled, no wraparound/repeat) and record graceful data exhaustion.
        if len(buffer) >= 2 and self._budget_reason() is None:
            yield self._emit(model, buffer)
        if self.stopped_reason is None:
            self.stopped_reason = "data_exhausted"


# --- challenge-owned interface module: the miner sees the FORCED ctx, not the installed one ---
interface = types.ModuleType("prism_challenge.evaluator.interface")


@dataclasses.dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 50304
    sequence_length: int = 1024
    max_layers: int = 96
    max_parameters: int = 150_000_000
    seed: int = 1337
    data_dir: str | None = None
    artifacts_dir: str | None = None
    reference_tokenizer_dir: str | None = None
    token_budget: int | None = None
    step_budget: int | None = None
    budget_seconds: float | None = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed_backend: str | None = None
    device: str = "cpu"

    @property
    def max_seq_len(self):
        return self.sequence_length

    @property
    def max_params(self):
        return self.max_parameters

    def build_model(self, *args, **kwargs):
        # Re-apply the forced init so the miner cannot override the random initialization.
        _force_init(self.seed)
        builder = interface._MINER_BUILD_MODEL
        if builder is None:
            raise RuntimeError("architecture build_model is not available")
        return builder(self, *args, **kwargs)

    def iter_train_batches(
        self, model, *, batch_size=DEFAULT_BATCH_SIZE, seq_len=None, tokenizer=None
    ):
        # Challenge-controlled, single-pass, predict-then-train online-loss instrument. The miner
        # iterates these batches; the challenge records the per-batch loss on each NEW batch BEFORE
        # the miner's optimizer updates on it (architecture.md 4.3).
        capture = interface._ONLINE_CAPTURE
        if capture is None:
            capture = _OnlineLossCapture(
                shards=interface._TRAIN_SHARDS,
                vocab_size=self.vocab_size,
                seq_len=seq_len if seq_len is not None else self.sequence_length,
                batch_size=batch_size,
                baseline_nats=math.log(max(self.vocab_size, 2)),
                token_budget=self.token_budget,
                step_budget=self.step_budget,
                tokenizer=tokenizer,
                device=self.device,
                budget_seconds=self.budget_seconds,
            )
            interface._ONLINE_CAPTURE = capture
        return capture.iterate(model)

    def reference_tokenizer(self, name):
        from prism_challenge.evaluator.reference_tokenizers import load_reference_tokenizer

        return load_reference_tokenizer(name, self.reference_tokenizer_dir)


@dataclasses.dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 3e-4
    batch_size: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01


interface.PrismContext = PrismContext
interface.PrismBatch = _PrismBatch
interface.TrainingRecipe = TrainingRecipe
interface._MINER_BUILD_MODEL = None
interface._TRAIN_SHARDS = train_shards
interface._ONLINE_CAPTURE = None

# Make the shadow ``prism_challenge.evaluator`` a real PACKAGE pointing at the installed
# evaluator source so the miner's two scripts still see the FORCED interface
# (``prism_challenge.evaluator.interface``) while real sibling submodules — notably
# ``reference_tokenizers`` (and ``dataset``) — remain importable offline. A bare module shadow
# (no ``__path__``) breaks ``ctx.reference_tokenizer('gpt2'|'llama')`` with a ModuleNotFoundError.
_real_evaluator_locations = []
try:
    _evaluator_spec = importlib.util.find_spec("prism_challenge.evaluator")
    if _evaluator_spec is not None and _evaluator_spec.submodule_search_locations:
        _real_evaluator_locations = list(_evaluator_spec.submodule_search_locations)
except Exception:
    _real_evaluator_locations = []

prism_pkg = sys.modules.get("prism_challenge")
if prism_pkg is None:
    prism_pkg = types.ModuleType("prism_challenge")
    prism_pkg.__path__ = []
    sys.modules["prism_challenge"] = prism_pkg
evaluator_pkg = types.ModuleType("prism_challenge.evaluator")
evaluator_pkg.__path__ = _real_evaluator_locations
sys.modules["prism_challenge.evaluator"] = evaluator_pkg
sys.modules["prism_challenge.evaluator.interface"] = interface

# --- import the two miner scripts AFTER forcing init ---
project_root = Path(os.environ.get("PRISM_PROJECT_ROOT", "/workspace/project"))
sys.path.insert(0, str(project_root))


def _import_from_file(path, module_name):
    if not Path(path).is_file():
        _fail("miner module not found: " + str(path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _fail("cannot import miner module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


arch_entry = payload.get("architecture_entrypoint", "architecture.py")
train_entry = payload.get("training_entrypoint", "training.py")
build_model_symbol = payload.get("build_model_symbol", "build_model")
train_symbol = payload.get("train_symbol", "train")
arch_module = _import_from_file(project_root / arch_entry, Path(arch_entry).stem)
miner_build_model = getattr(arch_module, build_model_symbol, None)
if not callable(miner_build_model):
    _fail("architecture entrypoint " + str(arch_entry) + " is missing " + build_model_symbol)
interface._MINER_BUILD_MODEL = miner_build_model
train_module = _import_from_file(project_root / train_entry, Path(train_entry).stem)
miner_train = getattr(train_module, train_symbol, None)
if not callable(miner_train):
    _fail("training entrypoint " + str(train_entry) + " is missing " + train_symbol)

ctx = PrismContext(
    vocab_size=int(context_data.get("vocab_size", 50304)),
    sequence_length=int(context_data.get("sequence_length", 1024)),
    max_layers=int(context_data.get("max_layers", 96) or 96),
    max_parameters=int(context_data.get("max_parameters", 150_000_000)),
    seed=forced_seed,
    data_dir=str(data_dir),
    artifacts_dir=str(artifacts_dir),
    reference_tokenizer_dir=context_data.get("reference_tokenizer_dir")
    or os.environ.get("PRISM_REFERENCE_TOKENIZER_DIR"),
    token_budget=context_data.get("token_budget"),
    step_budget=context_data.get("step_budget"),
    budget_seconds=context_data.get("budget_seconds"),
    rank=rank,
    local_rank=local_rank,
    world_size=world_size,
    distributed_backend=context_data.get("distributed_backend"),
    device=str(device),
)

_WATCHDOG_STOP = threading.Event()


def _artifacts_dir_bytes(root):
    total = 0
    try:
        for path in Path(root).rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _write_watchdog_manifest(reason):
    # Minimal challenge-authored manifest recording WHY the run was force-stopped. The run exits
    # non-zero so this is never scored; it is on-disk evidence for manifest-inspect.
    manifest = {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": str(payload.get("submission_id", "container")),
        "run_id": "prism-reexec-" + str(payload.get("submission_id", "container")),
        "mode": str(payload.get("execution_mode", "gpu_proxy_eval")),
        "run": {
            "seed": forced_seed,
            "forced_init": True,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "stopped_reason": reason,
        },
        "compute": {
            "schema": "prism_compute.v1",
            "gpu_count": world_size,
            "world_size": world_size,
            "nproc_per_node": world_size,
            "device": str(device),
        },
        "data": {
            "data_dir": str(data_dir),
            "source": "locked-fineweb-edu-train",
            "random_token_fallback": False,
            "single_pass": True,
            "stopped_reason": reason,
        },
        "metrics": {"online_loss": [], "covered_bytes": 0},
        "anti_cheat": {
            "budget_exceeded": reason == "wall_clock",
            "artifacts_quota_exceeded": reason == "artifacts_quota",
        },
        "score": {"final_score": None, "primary_metric": "prequential_bpb"},
        "miner_reported_ignored": True,
    }
    try:
        (Path(artifacts_dir) / CHALLENGE_MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _safety_watchdog(hard_budget_seconds, quota_bytes):
    start = time.monotonic()
    while not _WATCHDOG_STOP.wait(WATCHDOG_POLL_SECONDS):
        if (
            hard_budget_seconds is not None
            and (time.monotonic() - start) >= hard_budget_seconds
        ):
            if rank == 0:
                _write_watchdog_manifest("wall_clock")
            sys.stderr.write(
                BUDGET_EXCEEDED_MARKER + ": training loop exceeded the wall-clock budget safety "
                + "cap (" + str(hard_budget_seconds) + "s)\n"
            )
            sys.stderr.flush()
            os._exit(BUDGET_EXIT_CODE)
        if quota_bytes is not None and _artifacts_dir_bytes(artifacts_dir) > quota_bytes:
            if rank == 0:
                _write_watchdog_manifest("artifacts_quota")
            sys.stderr.write(
                ARTIFACTS_QUOTA_MARKER + ": artifacts_dir exceeded the disk quota ("
                + str(quota_bytes) + " bytes)\n"
            )
            sys.stderr.flush()
            os._exit(ARTIFACTS_QUOTA_EXIT_CODE)


def _start_safety_watchdog():
    budget_seconds = context_data.get("budget_seconds")
    grace = context_data.get("watchdog_grace_seconds")
    quota_bytes = context_data.get("artifacts_quota_bytes")
    hard_budget = None
    if budget_seconds is not None:
        hard_budget = float(budget_seconds) + float(grace if grace is not None else 0.0)
    if hard_budget is None and quota_bytes is None:
        return
    threading.Thread(
        target=_safety_watchdog, args=(hard_budget, quota_bytes), daemon=True
    ).start()


# Re-apply the forced init immediately before handing control to the miner loop.
_force_init(forced_seed)
# Start the safety watchdog (hard wall-clock budget + artifacts_dir disk quota) BEFORE handing
# control to the miner. The graceful in-loop budget stops a well-behaved iterating loop and scores
# the partial stream; the watchdog bounds a loop that hangs OUTSIDE the instrumented iterator or
# fills the only writable path, so neither can take down the host (architecture.md 4.3, 9).
_start_safety_watchdog()
print(
    "PRISM_RUNNER: imported architecture (" + str(arch_entry) + ") + training ("
    + str(train_entry) + "); calling train(ctx)",
    flush=True,
)
miner_train(ctx)
_WATCHDOG_STOP.set()
print("PRISM_RUNNER: train(ctx) returned", flush=True)

# --- summarize the challenge-captured online-loss stream (NOT miner-reported numbers) ---
capture = interface._ONLINE_CAPTURE
consumed_batches = capture.consumed_batches if capture is not None else 0
online_loss = list(capture.online_loss) if capture is not None else []
covered_bytes_cumulative = list(capture.covered_bytes_cumulative) if capture is not None else []
covered_bytes = int(round(capture.covered_bytes)) if capture is not None else 0
sum_nll_nats = capture.sum_nll_nats if capture is not None else 0.0
predicted_tokens = capture.predicted_tokens if capture is not None else 0
nan_inf_batches = capture.nan_inf_batches if capture is not None else 0
consumed_documents = capture.consumed_documents if capture is not None else 0
shard_offsets = capture.shard_offsets if capture is not None else []
# Which budget bound the single-pass run (token_budget | step_budget | wall_clock |
# data_exhausted); ``None`` only when the miner never iterated the instrument (zero-forward).
stopped_reason = capture.stopped_reason if capture is not None else None
baseline_nats = (
    capture.baseline_nats if capture is not None else math.log(max(ctx.vocab_size, 2))
)
step0_loss = online_loss[0] if online_loss else None
step0_threshold = STEP0_ANOMALY_FRACTION * baseline_nats
step0_anomaly = step0_loss is not None and step0_loss < step0_threshold
# A zero-batch / no-forward run captured no online loss: flag-fail it instead of fabricating a
# score or dividing by zero (architecture.md 4.3, 6; VAL-HARNESS-020).
zero_forward = consumed_batches == 0 or covered_bytes <= 0
no_learning = zero_forward
distinct_offsets = len({tuple(item) for item in shard_offsets})

# --- challenge-computed prequential bits-per-byte (architecture.md section 5) ---
# bpb = (sum over consumed tokens of -log2 p(token)) / total raw UTF-8 BYTES covered. The
# numerator is the TOKEN-WEIGHTED online negative log-likelihood the challenge captured itself
# (the AREA UNDER the whole single-pass loss curve), so the score is tokenizer-agnostic and
# compute-normalized (tokens/bytes, never wall-clock). final_score is a monotone transform where
# a SMALLER bpb yields a BETTER (larger) score; a step-0 anomaly zeroes the anti-cheat multiplier
# so smuggled-weights runs are flagged, never rewarded. Mirrors scoring.score_prequential_bpb.
sum_neg_log2_likelihood_bits = sum_nll_nats / math.log(2.0)
prequential_bpb = (
    sum_neg_log2_likelihood_bits / covered_bytes
    if covered_bytes > 0 and math.isfinite(sum_neg_log2_likelihood_bits)
    else None
)
score_anti_cheat_multiplier = 0.0 if step0_anomaly else 1.0
score_final = (
    (1.0 / (1.0 + prequential_bpb)) * score_anti_cheat_multiplier
    if prequential_bpb is not None and prequential_bpb > 0.0
    else None
)
trained_model = getattr(capture, "model", None) if capture is not None else None
# Tokenizer basis the prequential TRAIN bpb was measured on: "bytes" when the challenge instrument
# fed raw UTF-8 bytes (no miner tokenizer), else "tokenizer" (the miner passed a tokenizer to
# iter_train_batches). The HOST held-out measures val bpb on the BYTE basis, so the scorer only
# applies the anti-memorization GAP penalty for like-for-like bases (a benign tokenizer learner is
# never false-flagged; the byte-denominator delta tie-break stays valid regardless).
train_bpb_basis = (
    "bytes" if capture is None or getattr(capture, "tokenizer", None) is None else "tokenizer"
)
# The held-out delta + anti-memorization gap are computed HOST-SIDE on the SECRET val split (not
# mounted here); the runner only persists the trained weights so the scorer can run the twin and
# the trained model on val. ``trained_state_file`` is filled in below once the save succeeds.
trained_state_file = None


def _write_challenge_manifest():
    manifest = {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": str(payload.get("submission_id", "container")),
        "run_id": "prism-reexec-" + str(payload.get("submission_id", "container")),
        "mode": str(payload.get("execution_mode", "gpu_proxy_eval")),
        "run": {
            "seed": forced_seed,
            "forced_init": True,
            "deterministic_algorithms": True,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "nproc_per_node": world_size,
            "stopped_reason": stopped_reason,
        },
        "compute": {
            # Observability-only GPUs leased for this run (== world_size == nproc_per_node for the
            # single-node scored path; ==1 for the scored nproc=1 run). The host reconciles
            # gpu_count to the DB actual_gpu_count. Never an input to final_score.
            "schema": "prism_compute.v1",
            "gpu_count": world_size,
            "world_size": world_size,
            "nproc_per_node": world_size,
            "device": str(device),
        },
        "data": {
            "data_dir": str(data_dir),
            "shard_count": len(train_shards),
            "available_bytes": available_bytes,
            "source": "locked-fineweb-edu-train",
            "random_token_fallback": False,
            "single_pass": True,
            "covered_bytes": covered_bytes,
            "covered_bytes_cumulative": covered_bytes_cumulative,
            "consumed_documents": consumed_documents,
            "consumed_batches": consumed_batches,
            "consumed_offsets": len(shard_offsets),
            "distinct_offsets": distinct_offsets,
            "stopped_reason": stopped_reason,
        },
        "metrics": {
            "online_loss": online_loss,
            "step0_loss": step0_loss,
            "random_init_baseline_nats": baseline_nats,
            "consumed_batches": consumed_batches,
            "covered_bytes": covered_bytes,
            "predicted_tokens": predicted_tokens,
            "tokens_seen": predicted_tokens,
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": sum_neg_log2_likelihood_bits,
            "prequential_bpb": prequential_bpb,
            "bits_per_byte": prequential_bpb,
            "total_bytes_covered": covered_bytes,
            "nan_inf_batches": nan_inf_batches,
            "train_bpb_basis": train_bpb_basis,
        },
        "score": {
            "schema": "prism_score.v2",
            "primary_metric": "prequential_bpb",
            "prequential_bpb": prequential_bpb,
            "bits_per_byte": prequential_bpb,
            "final_score": score_final,
            "lower_is_better": True,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "sum_neg_log2_likelihood_bits": sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": sum_neg_log2_likelihood_bits,
            "tokens_consumed": predicted_tokens,
            "compute_normalization": "tokens_bytes",
            "wall_clock_term": False,
            "train_bpb_basis": train_bpb_basis,
            "anti_cheat_multiplier": score_anti_cheat_multiplier,
            "anomaly": step0_anomaly,
            "miner_reported_ignored": True,
        },
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "step0_anomaly_threshold_nats": step0_threshold,
            "nan_inf_detected": nan_inf_batches > 0,
            "nan_inf_batches": nan_inf_batches,
            "no_learning": no_learning,
            "zero_forward": zero_forward,
        },
        "artifacts": {
            # The held-out delta is computed HOST-SIDE from these weights on the SECRET val split.
            "trained_state": trained_state_file,
        },
        "miner_reported_ignored": True,
    }
    out = Path(artifacts_dir) / CHALLENGE_MANIFEST_NAME
    out.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _save_trained_state():
    # Persist the trained weights (CPU state_dict) so the host scorer can run the trained model and
    # a forced-seed random-init twin on the SECRET val split. Best-effort: a save failure only
    # skips the held-out delta; the run still scores on prequential bpb.
    global trained_state_file
    target = Path(artifacts_dir) / TRAINED_STATE_FILENAME
    # Unconditionally OWN this path BEFORE writing: remove any miner-planted file first so a hostile
    # pickle cannot survive by making state_dict() raise (or by a zero-forward run skipping the
    # save). The host reads only the manifest-recorded artifact, so trained_state_file stays None
    # unless THIS challenge save below succeeds.
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if zero_forward or trained_model is None:
        return
    try:
        state = {key: value.detach().to("cpu") for key, value in trained_model.state_dict().items()}
        torch.save(state, target)
        trained_state_file = TRAINED_STATE_FILENAME
    except Exception as exc:  # noqa: BLE001 - never fail the run on a held-out artifact save
        sys.stderr.write("PRISM_RUNNER: trained-state save skipped: " + repr(exc) + "\n")
        try:
            target.unlink()
        except OSError:
            pass
        trained_state_file = None


if rank == 0:
    # The miner may have written its own manifest during train(ctx); discard it and author ours.
    for stale in Path(artifacts_dir).glob(MANIFEST_GLOB):
        if stale.name != CHALLENGE_MANIFEST_NAME:
            try:
                stale.unlink()
            except OSError:
                pass
    _save_trained_state()
    _write_challenge_manifest()
    if no_learning:
        _fail(
            "zero-batch / no-forward run: no online loss captured "
            "(miner never trained on the instrumented single-pass batches)"
        )
    print(
        "PRISM_METRICS_JSON="
        + json.dumps(
            {
                "available_bytes": float(available_bytes),
                "shard_count": float(len(train_shards)),
                "covered_bytes": float(covered_bytes),
                "consumed_batches": float(consumed_batches),
                "online_loss_samples": float(len(online_loss)),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
'''
