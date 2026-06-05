from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from .config import PrismSettings
from .db import dumps, loads
from .evaluator import llm_review, source_similarity
from .evaluator.anti_cheat import evaluate_anti_cheat
from .evaluator.checkpoints import (
    CheckpointWorkspaceError,
    checkpoint_artifact_logical_size,
    load_checkpoint_metadata,
)
from .evaluator.component_agents import (
    ComponentOwnershipDecision,
    SemanticOwnershipAgent,
)
from .evaluator.component_signatures import (
    ComponentSemanticSignature,
    build_semantic_signature,
)
from .evaluator.components import (
    PrismComponentFingerprints,
    PrismProjectComponents,
    component_fingerprints,
    project_components,
)
from .evaluator.container import InfrastructureEvaluationError, PrismContainerEvaluator
from .evaluator.interface import PrismContext, TrainingRecipe
from .evaluator.l1_syntax import validate_l1
from .evaluator.l2_proxy import score_l2
from .evaluator.l3_train import score_l3
from .evaluator.l4_benchmark import score_l4
from .evaluator.lium_client import LiumJob
from .evaluator.modes import execution_mode_from_value, run_local_cpu_smoke
from .evaluator.review_rules import ReviewRule, load_review_rules
from .evaluator.sandbox import (
    SandboxViolation,
    inspect_code,
    load_module,
    load_submission_contract,
)
from .evaluator.schemas import ExecutionMode, PrismRunManifest
from .evaluator.scoring import final_score, score_recipe
from .gpu_scheduler import (
    GpuLease,
    GpuLeaseScheduler,
    lease_request_from_runtime,
    targets_from_settings,
)
from .models import EvaluationAssignmentStatus, SubmissionStatus
from .repository import PrismRepository, now_iso
from .sdk.executors.docker import DockerExecutor, DockerLimits, DockerMount, DockerRunSpec

DEFAULT_REVIEW_RULES = (
    ReviewRule("prism:no-secret-exfiltration", "Do not read, infer, print, or transmit secrets."),
    ReviewRule("prism:no-escape", "Do not use filesystem, process, or network escapes."),
    ReviewRule("prism:model-contract", "Only implement the Prism model and recipe contract."),
)
CONTAINER_EXECUTION_BACKENDS = frozenset(
    {"platform_container", "platform_gpu", "container_gpu", "docker_gpu"}
)
SUPPORTED_EXECUTION_BACKENDS = CONTAINER_EXECUTION_BACKENDS


def _validated_retry_checkpoint_dir(
    job: dict[str, object],
    *,
    submission_id: str,
    code_hash: str,
    arch_hash: str,
    recipe_fingerprint: str,
    previous_attempt: int,
) -> Path:
    artifact_output = Path(str(job["artifact_output_path"]))
    manifest_path = Path(str(job["run_manifest_path"]))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = PrismRunManifest.model_validate(manifest_payload)
    if manifest.submission_id != submission_id:
        raise ValueError("retry checkpoint submission_id mismatch")
    if not manifest.validation.passed or not manifest.validation.score_eligible:
        raise ValueError("retry checkpoint manifest is not validation-passed")
    checkpoints = [
        checkpoint
        for checkpoint in manifest.artifacts.checkpoints
        if checkpoint.attempt == previous_attempt
    ]
    if not checkpoints:
        raise ValueError("retry checkpoint manifest has no prior-attempt checkpoint")
    checkpoint = checkpoints[-1]
    metadata_path = artifact_output / checkpoint.metadata_path
    checkpoint_dir = metadata_path.parent
    checkpoint_artifact_logical_size(checkpoint_dir)
    metadata = load_checkpoint_metadata(metadata_path)
    expected = {
        "submission_id": submission_id,
        "code_hash": code_hash,
        "arch_hash": arch_hash,
        "recipe_fingerprint": recipe_fingerprint,
        "attempt": previous_attempt,
        "checkpoint_dir": checkpoint_dir.relative_to(artifact_output).as_posix(),
    }
    for field, value in expected.items():
        if metadata[field] != value:
            raise ValueError(f"retry checkpoint {field} mismatch")
    checkpoint_path = artifact_output / str(metadata["checkpoint_path"])
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("retry checkpoint file is missing")
    checkpoint_path.resolve(strict=False).relative_to(checkpoint_dir.resolve(strict=False))
    if metadata["checkpoint_path"] != checkpoint.path:
        raise ValueError("retry checkpoint path mismatch")
    return checkpoint_dir


def _recipe_fingerprint(recipe: TrainingRecipe) -> str:
    payload = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _recipe_fingerprint_from_code(code: str, ctx: PrismContext) -> str:
    recipe = load_module(code).get_recipe(ctx)
    if not isinstance(recipe, TrainingRecipe):
        if isinstance(recipe, dict):
            recipe = TrainingRecipe(**recipe)
        else:
            raise SandboxViolation("get_recipe must return TrainingRecipe or dict")
    return _recipe_fingerprint(recipe)


@dataclass(frozen=True)
class StaticReviewOutcome:
    code: str
    rejected: bool
    reason: str | None = None
    violations: tuple[str, ...] = ()
    held: bool = False


@dataclass(frozen=True)
class ComponentReview:
    components: PrismProjectComponents
    fingerprints: PrismComponentFingerprints
    semantic_signature: ComponentSemanticSignature
    ownership_decision: ComponentOwnershipDecision | None = None


class PrismWorker:
    def __init__(
        self,
        repository: PrismRepository,
        ctx: PrismContext,
        *,
        execution_backend: str = "platform_gpu",
        settings: PrismSettings | None = None,
    ) -> None:
        if execution_backend not in SUPPORTED_EXECUTION_BACKENDS:
            raise ValueError(f"Unsupported execution backend: {execution_backend}")
        self.repository = repository
        self.ctx = ctx
        self.lium: Any = None
        self.execution_backend = execution_backend
        self.settings = settings or PrismSettings()

    async def process_next(self) -> str | None:
        submission = await self.repository.claim_next()
        if submission is None:
            return None
        submission_id = str(submission["id"])
        code = str(submission["code"])
        filename = str(submission.get("filename") or "model.py")
        raw_metadata = submission.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        hotkey = str(submission.get("hotkey") or "")
        code_hash = str(submission.get("code_hash") or sha256(code.encode()).hexdigest())
        if self.execution_backend in CONTAINER_EXECUTION_BACKENDS:
            return await self._process_container(
                submission_id, code, filename, metadata, hotkey, code_hash
            )
        raise ValueError(f"Unsupported execution backend: {self.execution_backend}")

    async def assign_next_to_validator(self, validator_hotkey: str):
        existing = await self.repository.active_assignment_for_validator(validator_hotkey)
        if existing is not None:
            return existing
        submission = await self.repository.claim_next()
        if submission is None:
            return None
        submission_id = str(submission["id"])
        code = str(submission["code"])
        filename = str(submission.get("filename") or "model.py")
        raw_metadata = submission.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        hotkey = str(submission.get("hotkey") or "")
        code_hash = str(submission.get("code_hash") or sha256(code.encode()).hexdigest())
        try:
            review = await self._review_static_submission(
                submission_id=submission_id,
                code=code,
                filename=filename,
                metadata=metadata,
                hotkey=hotkey,
                code_hash=code_hash,
            )
            if review.rejected:
                await self._reject_submission(submission_id, review.reason or "review rejected")
                return None
            if review.held:
                return None
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            report = self._inspect_project_snapshot(snapshot, review.code)
        except Exception as exc:
            await self._reject_submission(submission_id, str(exc))
            return None
        arch_hash = component_review.fingerprints.family_hash
        if not arch_hash:
            arch_hash = sha256(":".join(sorted(report.ast_fingerprint)).encode()).hexdigest()
        return await self.repository.create_assignment(
            submission_id=submission_id,
            validator_hotkey=validator_hotkey,
            arch_hash=arch_hash,
            timeout_seconds=self.settings.validator_assignment_timeout_seconds,
        )

    async def reject_assignment(self, assignment_id: str, reason: str | None = None) -> None:
        assignment = await self.repository.get_assignment(assignment_id)
        if assignment is None:
            raise ValueError("assignment not found")
        await self.repository.set_assignment_status(
            assignment_id,
            EvaluationAssignmentStatus.REJECTED,
            error=reason or "validator rejected assignment",
        )
        if assignment.attempt >= self.settings.validator_assignment_max_attempts:
            await self._fail_submission(
                assignment.submission_id, "validator assignment attempts exhausted"
            )
            return
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, updated_at=? WHERE id=?",
                (SubmissionStatus.PENDING.value, now_iso(), assignment.submission_id),
            )

    async def expire_assignments(self) -> list[str]:
        return await self.repository.expire_stale_assignments(
            self.settings.validator_assignment_max_attempts
        )

    async def complete_assignment(self, assignment_id: str, metrics: dict[str, float]) -> None:
        assignment = await self.repository.get_assignment(assignment_id)
        if assignment is None:
            raise ValueError("assignment not found")
        await self.repository.set_assignment_status(
            assignment_id,
            EvaluationAssignmentStatus.COMPLETED,
            metrics=metrics,
        )
        component_review: ComponentReview | None = None
        try:
            snapshot = self._snapshot_from_submission(
                assignment.code,
                assignment.filename,
                assignment.metadata,
            )
            component_review = self._component_review(snapshot)
        except Exception:
            component_review = None
        await self._finalize_remote_result(
            submission_id=assignment.submission_id,
            arch_hash=assignment.arch_hash,
            anti_multiplier=1.0,
            diversity_bonus=0.0,
            metrics=metrics,
            component_review=component_review,
        )

    async def _process_local(
        self,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
    ) -> str:
        review = await self._review_static_submission(
            submission_id=submission_id,
            code=code,
            filename=filename,
            metadata=metadata,
            hotkey=hotkey,
            code_hash=code_hash,
        )
        if review.rejected:
            await self._reject_submission(submission_id, review.reason or "review rejected")
            return submission_id
        if review.held:
            return submission_id
        code = review.code
        async with self.repository.database.connect() as conn:
            try:
                l1 = validate_l1(code, self.ctx)
                if not l1.valid:
                    await conn.execute(
                        "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                        (SubmissionStatus.REJECTED.value, l1.error, now_iso(), submission_id),
                    )
                    return submission_id
                previous = await self.repository.previous_codes(submission_id)
                anti = evaluate_anti_cheat(code, previous)
                l2 = score_l2(code, self.ctx)
                l3 = score_l3(code, self.ctx)
                l4 = await score_l4(code, self.ctx, self.lium)
                _model, recipe, _report = load_submission_contract(code, self.ctx)
                recipe_score = score_recipe(recipe)
                penalty = 1.0 if l3.hard_killed else 0.0
                if l4.scale_collapse:
                    penalty += 0.25
                runtime_config = await self.repository.runtime_config(self.settings, official=True)
                scored = final_score(
                    q_arch=l4.q_arch,
                    q_recipe=recipe_score,
                    anti_cheat_multiplier=anti.multiplier,
                    diversity_bonus=anti.diversity_bonus,
                    penalty=penalty,
                    arch_weight=runtime_config.score_weights.final_architecture_weight,
                    recipe_weight=runtime_config.score_weights.final_recipe_weight,
                )
                for finding in anti.findings:
                    await conn.execute(
                        "INSERT INTO cheat_findings("
                        "id, submission_id, kind, severity, details, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid4()),
                            submission_id,
                            finding.kind,
                            finding.severity,
                            finding.details,
                            now_iso(),
                        ),
                    )
                metrics = {
                    "l2_q_proxy": l2.q_proxy,
                    "l3_loss": l3.loss,
                    "l3_kendall_tau": l3.kendall_tau,
                    **l4.metrics,
                }
                await conn.execute(
                    "INSERT OR REPLACE INTO scores("
                    "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,"
                    "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        submission_id,
                        scored.q_arch,
                        scored.q_recipe,
                        anti.multiplier,
                        anti.diversity_bonus,
                        penalty,
                        scored.final_score,
                        dumps(metrics),
                        now_iso(),
                    ),
                )
                await conn.execute(
                    "UPDATE submissions SET status=?, arch_hash=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.COMPLETED.value, l1.arch_hash, now_iso(), submission_id),
                )
                return submission_id
            except Exception as exc:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.FAILED.value, str(exc), now_iso(), submission_id),
                )
                return submission_id

    async def _submit_remote(
        self,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
    ) -> str:
        try:
            review = await self._review_static_submission(
                submission_id=submission_id,
                code=code,
                filename=filename,
                metadata=metadata,
                hotkey=hotkey,
                code_hash=code_hash,
            )
            if review.rejected:
                await self._reject_submission(submission_id, review.reason or "review rejected")
                return submission_id
            if review.held:
                return submission_id
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code = review.code
            report = self._inspect_project_snapshot(snapshot, code)
            code_hash = sha256(code.encode()).hexdigest()
            arch_hash = component_review.fingerprints.family_hash
            if not arch_hash:
                arch_basis = ":".join(sorted(report.ast_fingerprint))
                arch_hash = sha256(arch_basis.encode()).hexdigest()
            previous = await self.repository.previous_codes(submission_id)
            anti = evaluate_anti_cheat(
                code,
                previous,
                allowed_import_roots=self._local_import_roots(snapshot),
            )
            job = await self.lium.submit_job(
                {
                    "challenge": "prism",
                    "submission_id": submission_id,
                    "code": code,
                    "code_hash": code_hash,
                    "arch_hash": arch_hash,
                    "benchmarks": [
                        "learning_speed",
                        "heldout",
                        "long_context",
                        "reasoning",
                        "generalism",
                        "stability",
                        "efficiency",
                    ],
                    "context": {
                        "vocab_size": self.ctx.vocab_size,
                        "sequence_length": self.ctx.sequence_length,
                        "max_parameters": self.ctx.max_parameters,
                    },
                },
                idempotency_key=submission_id,
            )
            await self._record_remote_job(submission_id, job)
            if job.status == "completed":
                await self._finalize_remote_result(
                    submission_id=submission_id,
                    arch_hash=arch_hash,
                    anti_multiplier=anti.multiplier,
                    diversity_bonus=anti.diversity_bonus,
                    metrics=job.metrics,
                    component_review=component_review,
                )
            return submission_id
        except Exception as exc:
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, str(exc), now_iso(), submission_id),
                )
            return submission_id

    async def _process_container(
        self,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
    ) -> str:
        try:
            review = await self._review_static_submission(
                submission_id=submission_id,
                code=code,
                filename=filename,
                metadata=metadata,
                hotkey=hotkey,
                code_hash=code_hash,
            )
            if review.rejected:
                await self._reject_submission(submission_id, review.reason or "review rejected")
                return submission_id
            if review.held:
                return submission_id
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code = review.code
            report = self._inspect_project_snapshot(snapshot, code)
        except (SandboxViolation, SyntaxError) as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id
        except Exception as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id

        code_hash = sha256(code.encode()).hexdigest()
        arch_hash = component_review.fingerprints.family_hash
        if not arch_hash:
            arch_hash = sha256(":".join(sorted(report.ast_fingerprint)).encode()).hexdigest()
        previous = await self.repository.previous_codes(submission_id)
        anti = evaluate_anti_cheat(
            code,
            previous,
            allowed_import_roots=self._local_import_roots(snapshot),
        )
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        try:
            execution_mode = execution_mode_from_value(metadata.get("execution_mode"))
        except ValueError as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id
        if execution_mode is ExecutionMode.LOCAL_CPU_SMOKE:
            return await self._process_local_cpu_smoke_mode(
                submission_id=submission_id,
                code=code,
                code_hash=code_hash,
                arch_hash=arch_hash,
                anti_multiplier=anti.multiplier,
                diversity_bonus=anti.diversity_bonus,
            )
        score_eligible = metadata.get("score_eligible")
        scheduler = GpuLeaseScheduler(
            self.repository.database, targets_from_settings(self.settings, runtime_config)
        )
        lease = await scheduler.enqueue_or_allocate(
            lease_request_from_runtime(
                submission_id=submission_id,
                job_id=None,
                runtime_policy=runtime_config,
                mode=execution_mode.value,
                score_eligible=bool(score_eligible) if score_eligible is not None else None,
            )
        )
        if not lease.active:
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.PENDING.value, lease.reason, now_iso(), submission_id),
                )
            return submission_id
        effective_settings = self.settings.model_copy(
            update={
                "platform_eval_gpu_count": lease.gpu_count,
                "platform_eval_gpu_type": runtime_config.gpu_policy.gpu_type,
                "platform_eval_gpu_server": lease.target_server,
                "platform_eval_gpu_device_ids": lease.device_ids,
            }
        )
        evaluator = PrismContainerEvaluator(settings=effective_settings, ctx=self.ctx)
        attempt = await self.repository.container_job_attempt_count(
            submission_id, self.execution_backend
        ) + 1
        resume_checkpoint_dir = None
        if "function:load_checkpoint" in report.ast_fingerprint:
            resume_checkpoint_dir = await self._retry_resume_checkpoint_dir(
                submission_id=submission_id,
                code_hash=code_hash,
                arch_hash=arch_hash,
                recipe_fingerprint=_recipe_fingerprint_from_code(code, self.ctx),
                attempt=attempt,
            )
        try:
            result = await asyncio.to_thread(
                evaluator.evaluate,
                submission_id=submission_id,
                code=code,
                code_hash=code_hash,
                arch_hash=arch_hash,
                backend=self.execution_backend,
                files=snapshot.files,
                entrypoint=component_review.components.entrypoint,
                gpu_lease=lease,
                execution_mode=execution_mode,
                attempt=attempt,
                resume_checkpoint_dir=resume_checkpoint_dir,
            )
            await self._record_container_job(
                submission_id=submission_id,
                status="completed",
                container_name=result.container_name,
                metrics=result.metrics,
                lease=lease,
                artifact_output_path=result.artifact_output_path,
                run_manifest_path=result.run_manifest_path,
                attempt=attempt,
            )
            await self._finalize_remote_result(
                submission_id=submission_id,
                arch_hash=arch_hash,
                anti_multiplier=anti.multiplier,
                diversity_bonus=anti.diversity_bonus,
                metrics=result.metrics,
                component_review=component_review,
            )
            return submission_id
        except InfrastructureEvaluationError as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="infra_failed",
                container_name=None,
                metrics={},
                error=str(exc),
                lease=lease,
                infra_retryable=True,
                artifact_output_path=exc.artifact_output_path,
                run_manifest_path=exc.run_manifest_path,
                attempt=attempt,
            )
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.PENDING.value, str(exc), now_iso(), submission_id),
                )
            return submission_id
        except Exception as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="failed",
                container_name=None,
                metrics={},
                error=str(exc),
                lease=lease,
                attempt=attempt,
            )
            await self._fail_submission(submission_id, str(exc))
            return submission_id
        finally:
            await scheduler.release_for_submission(submission_id, "container job finished")

    async def _retry_resume_checkpoint_dir(
        self,
        *,
        submission_id: str,
        code_hash: str,
        arch_hash: str,
        recipe_fingerprint: str,
        attempt: int,
    ) -> Path | None:
        if attempt <= 1:
            return None
        job = await self.repository.latest_retryable_container_job(
            submission_id, self.execution_backend
        )
        if job is None:
            return None
        try:
            return _validated_retry_checkpoint_dir(
                job,
                submission_id=submission_id,
                code_hash=code_hash,
                arch_hash=arch_hash,
                recipe_fingerprint=recipe_fingerprint,
                previous_attempt=attempt - 1,
            )
        except (OSError, ValueError, CheckpointWorkspaceError):
            return None

    async def _process_local_cpu_smoke_mode(
        self,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        anti_multiplier: float,
        diversity_bonus: float,
    ) -> str:
        artifact_output = Path("/tmp/prism-local-cpu-smoke") / submission_id
        try:
            result = await asyncio.to_thread(
                run_local_cpu_smoke,
                submission_id=submission_id,
                code=code,
                code_hash=code_hash,
                arch_hash=arch_hash,
                ctx=self.ctx,
                artifact_output_path=artifact_output,
            )
            metrics = {
                **result.metrics,
                "anti_cheat_multiplier": anti_multiplier,
                "diversity_bonus": diversity_bonus,
            }
            await self._record_container_job(
                submission_id=submission_id,
                status="completed",
                container_name="local_cpu_smoke",
                metrics=metrics,
                artifact_output_path=result.artifact_output_path,
                run_manifest_path=result.run_manifest_path,
            )
            q_arch = max(0.0, min(1.0, float(metrics.get("q_arch", 0.0))))
            q_recipe = max(0.0, min(1.0, float(metrics.get("q_recipe", 0.5))))
            penalty = float(metrics.get("penalty", 0.0))
            runtime_config = await self.repository.runtime_config(self.settings, official=True)
            scored = final_score(
                q_arch=q_arch,
                q_recipe=q_recipe,
                anti_cheat_multiplier=anti_multiplier,
                diversity_bonus=diversity_bonus,
                penalty=penalty,
                arch_weight=runtime_config.score_weights.final_architecture_weight,
                recipe_weight=runtime_config.score_weights.final_recipe_weight,
            )
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO scores("
                    "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,"
                    "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        submission_id,
                        scored.q_arch,
                        scored.q_recipe,
                        anti_multiplier,
                        diversity_bonus,
                        penalty,
                        scored.final_score,
                        dumps(metrics),
                        now_iso(),
                    ),
                )
                await conn.execute(
                    "UPDATE submissions SET status=?, arch_hash=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.COMPLETED.value, arch_hash, now_iso(), submission_id),
                )
            return submission_id
        except Exception as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="failed",
                container_name="local_cpu_smoke",
                metrics={},
                error=str(exc),
                artifact_output_path=str(artifact_output),
                run_manifest_path=str(artifact_output / "prism_run_manifest.v1.json"),
            )
            await self._fail_submission(submission_id, str(exc))
            return submission_id

    def _inspect_project_snapshot(
        self, snapshot: source_similarity.SourceSnapshot, primary_code: str
    ):
        local_imports = self._local_import_roots(snapshot)
        primary_file = next(
            (file for file in snapshot.python_files if file.content == primary_code),
            None,
        )
        report = inspect_code(
            primary_code,
            allowed_import_roots=local_imports,
            artifact_path=primary_file.path if primary_file else "model.py",
        )
        for file in snapshot.python_files:
            if file.content == primary_code:
                continue
            inspect_code(
                file.content,
                require_contract=False,
                allowed_import_roots=local_imports,
                artifact_path=file.path,
            )
        return report

    def _local_import_roots(self, snapshot: source_similarity.SourceSnapshot) -> set[str]:
        return {
            Path(file.path).stem
            for file in snapshot.python_files
            if Path(file.path).stem != "__init__"
        }

    def _snapshot_from_submission(
        self,
        code: str,
        filename: str,
        metadata: dict[str, Any],
    ) -> source_similarity.SourceSnapshot:
        return source_similarity.snapshot_from_submission(
            code,
            filename,
            metadata,
            max_files=self.settings.plagiarism_storage_max_files,
            max_bytes=self.settings.plagiarism_storage_max_bytes,
        )

    def _component_review(self, snapshot: source_similarity.SourceSnapshot) -> ComponentReview:
        components = project_components(snapshot)
        fingerprints = component_fingerprints(components)
        return ComponentReview(
            components=components,
            fingerprints=fingerprints,
            semantic_signature=build_semantic_signature(components, fingerprints),
        )

    async def _component_ownership_review(self, review: ComponentReview) -> ComponentReview:
        if not self.settings.component_agent_enabled:
            decision = ComponentOwnershipDecision(
                architecture_action="existing" if review.components.architecture_id else "new",
                architecture_confidence=1.0,
                training_action="none" if review.components.kind == "architecture_only" else "new",
                training_confidence=1.0,
                matched_architecture_id=review.components.architecture_id,
                reason="component agent disabled",
            )
            return ComponentReview(
                components=review.components,
                fingerprints=review.fingerprints,
                semantic_signature=review.semantic_signature,
                ownership_decision=decision,
            )
        architectures, training = await self.repository.component_candidates(
            family_hash=review.fingerprints.family_hash,
            requested_architecture_id=review.components.architecture_id,
            limit=self.settings.component_agent_candidate_top_k,
        )
        decision = SemanticOwnershipAgent(
            min_confidence=self.settings.component_agent_min_confidence,
            same_threshold=self.settings.component_agent_same_threshold,
            hold_threshold=self.settings.component_agent_hold_threshold,
        ).decide(
            signature=review.semantic_signature,
            architecture_candidates=architectures,
            training_candidates=training,
            requested_architecture_id=review.components.architecture_id,
        )
        if self.settings.component_hold_low_confidence and not decision.held:
            decision = self._hold_low_confidence_decision(decision)
        return ComponentReview(
            components=review.components,
            fingerprints=review.fingerprints,
            semantic_signature=review.semantic_signature,
            ownership_decision=decision,
        )

    def _hold_low_confidence_decision(
        self, decision: ComponentOwnershipDecision
    ) -> ComponentOwnershipDecision:
        arch_action = decision.architecture_action
        train_action = decision.training_action
        reason = decision.reason
        if (
            arch_action in {"existing", "transfer"}
            and decision.architecture_confidence < self.settings.component_agent_min_confidence
        ):
            arch_action = "hold"
            reason = f"{reason}; low architecture confidence"
        if (
            train_action in {"existing", "transfer"}
            and decision.training_confidence < self.settings.component_agent_min_confidence
        ):
            train_action = "hold"
            reason = f"{reason}; low training confidence"
        if arch_action == decision.architecture_action and train_action == decision.training_action:
            return decision
        return ComponentOwnershipDecision(
            architecture_action=arch_action,
            architecture_confidence=decision.architecture_confidence,
            training_action=train_action,
            training_confidence=decision.training_confidence,
            matched_architecture_id=decision.matched_architecture_id,
            matched_training_variant_id=decision.matched_training_variant_id,
            reason=reason,
            raw=decision.raw,
        )

    def _entrypoint_code(self, snapshot: source_similarity.SourceSnapshot, entrypoint: str) -> str:
        match = next((file for file in snapshot.files if file.path == entrypoint), None)
        if match is None:
            raise ValueError(f"Prism project entrypoint not found: {entrypoint}")
        return match.content

    async def _review_static_submission(
        self,
        *,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
    ) -> StaticReviewOutcome:
        try:
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            await self.repository.record_llm_review_event(
                submission_id=submission_id,
                state="received",
                actor="system",
                tool_name="submission_receiver",
                payload={"filename": filename, "code_hash": code_hash},
                reason="submission received for LLM review flow",
                idempotency_key="state:received",
            )
            component_review = self._component_review(snapshot)
            code_for_eval = self._entrypoint_code(snapshot, component_review.components.entrypoint)
        except Exception as exc:
            return StaticReviewOutcome("", True, str(exc))
        if not code_for_eval.strip():
            return StaticReviewOutcome(code_for_eval, True, "submission contains no Python source")
        await self.repository.record_llm_review_event(
            submission_id=submission_id,
            state="static_validation_passed",
            actor="system",
            tool_name="static_validator",
            payload={"entrypoint": component_review.components.entrypoint},
            reason="submission passed static review preconditions",
            idempotency_key="state:static_validation_passed",
        )
        await self.repository.record_llm_review_event(
            submission_id=submission_id,
            state="architecture_analyzed",
            actor="system",
            tool_name="architecture_analyzer",
            payload={
                "family_hash": component_review.fingerprints.family_hash,
                "architecture_graph_hash": (
                    component_review.semantic_signature.architecture_graph_hash
                ),
            },
            reason="architecture graph analyzed for review context",
            idempotency_key="state:architecture_analyzed",
        )
        await self.repository.store_source_snapshot(
            submission_id=submission_id,
            hotkey=hotkey,
            code_hash=code_hash,
            payload=snapshot.to_payload(),
        )
        rules = self._review_rules()
        llm_config = self._llm_config()
        safety = await asyncio.to_thread(
            llm_review.review_code,
            snapshot.combined_python(),
            config=llm_config,
            rules=rules,
            subject="Prism project",
        )
        await self.repository.store_llm_review(
            submission_id=submission_id,
            approved=safety.approved,
            reason=safety.reason,
            violations=safety.violations,
            confidence=safety.confidence,
            raw=safety.raw,
            mermaid=safety.mermaid,
            evidence=safety.evidence,
            held=safety.held,
        )
        if safety.held:
            return StaticReviewOutcome(code_for_eval, False, safety.reason, held=True)
        if not safety.approved:
            return StaticReviewOutcome(code_for_eval, True, safety.reason, tuple(safety.violations))
        if not self.settings.plagiarism_enabled:
            return StaticReviewOutcome(code_for_eval, False)
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        history = await self.repository.source_similarity_candidates(
            exclude_submission_id=submission_id
        )
        duplicate = source_similarity.classify_duplicate(
            submission_id=submission_id,
            code_hash=code_hash,
            snapshot=snapshot,
            architecture_graph=component_review.semantic_signature.architecture_graph,
            rows=history,
            thresholds=runtime_config.duplicate_thresholds.model_dump(),
            top_k=self.settings.plagiarism_top_k,
        )
        if duplicate.candidate is not None:
            violations = ["duplicate_similarity"] if duplicate.rejected else []
            await self.repository.store_plagiarism_review(
                submission_id=submission_id,
                candidate_submission_id=duplicate.candidate.submission_id,
                similarity=float(duplicate.report["source_similarity"]),
                verdict=duplicate.rejected,
                reason=duplicate.reason,
                violations=violations,
                report=duplicate.report,
            )
            if duplicate.rejected:
                return StaticReviewOutcome(
                    code_for_eval,
                    True,
                    reason=duplicate.reason,
                    violations=tuple(violations),
                )
            if duplicate.held:
                await self.repository.hold_submission_for_duplicate_review(
                    submission_id=submission_id,
                    reason=duplicate.reason,
                    report=duplicate.report,
                )
                return StaticReviewOutcome(code_for_eval, False, duplicate.reason, held=True)
            return StaticReviewOutcome(code_for_eval, False)

        return StaticReviewOutcome(code_for_eval, False)

    async def _reject_submission(self, submission_id: str, reason: str) -> None:
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.REJECTED.value, reason, now_iso(), submission_id),
            )

    async def _fail_submission(self, submission_id: str, reason: str) -> None:
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.FAILED.value, reason, now_iso(), submission_id),
            )

    def _review_rules(self) -> tuple[ReviewRule, ...]:
        return load_review_rules(
            defaults=DEFAULT_REVIEW_RULES,
            rules_json=self.settings.subnet_rules_json,
            rules_file=self.settings.subnet_rules_file,
        )

    def _llm_config(self) -> llm_review.LlmReviewConfig:
        return llm_review.LlmReviewConfig(
            enabled=self.settings.llm_review_enabled,
            required=self.settings.llm_review_required,
            base_url=self.settings.chutes_base_url,
            model=self.settings.chutes_model,
            api_key=self.settings.chutes_api_key_value(),
            api_key_file=self.settings.chutes_api_key_file,
            timeout_seconds=self.settings.llm_review_timeout_seconds,
            temperature=self.settings.llm_review_temperature,
            max_tokens=self.settings.llm_review_max_tokens,
            max_retries=self.settings.llm_review_max_retries,
        )

    def _pair_sandbox_runner(self, submission_id: str) -> source_similarity.SandboxRunner:
        executor = DockerExecutor(
            challenge=self.settings.slug,
            docker_bin=self.settings.docker_bin,
            allowed_images=(self.settings.plagiarism_sandbox_image,),
            backend=self.settings.docker_backend,
            broker_url=self.settings.docker_broker_url,
            broker_token=self.settings.docker_broker_token,
            broker_token_file=str(self.settings.docker_broker_token_file)
            if self.settings.docker_broker_token_file
            else None,
        )

        def run(left: Path, right: Path, script: Path) -> str:
            result = executor.run(
                DockerRunSpec(
                    image=self.settings.plagiarism_sandbox_image,
                    command=("python", "/compare.py"),
                    mounts=(
                        DockerMount(left, "/current"),
                        DockerMount(right, "/candidate"),
                        DockerMount(script, "/compare.py"),
                    ),
                    labels={"platform.job": submission_id, "platform.task": "plagiarism"},
                    limits=DockerLimits(
                        cpus=min(self.settings.docker_cpus, 1.0),
                        memory="512m",
                        memory_swap="512m",
                        pids_limit=128,
                        network="none",
                        read_only=True,
                    ),
                ),
                self.settings.plagiarism_sandbox_timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout or "pair sandbox failed")
            return result.stdout

        return run

    async def _record_remote_job(self, submission_id: str, job: LiumJob) -> None:
        status = (
            SubmissionStatus.COMPLETED.value
            if job.status == "completed"
            else SubmissionStatus.RUNNING.value
        )
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, updated_at=? WHERE id=?",
                (status, now_iso(), submission_id),
            )
            await conn.execute(
                "INSERT INTO eval_jobs(id, submission_id, level, status, external_job_id, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    "remote",
                    job.status,
                    job.id,
                    dumps(job.metrics),
                    now_iso(),
                    now_iso(),
                ),
            )

    async def _record_container_job(
        self,
        *,
        submission_id: str,
        status: str,
        container_name: str | None,
        metrics: dict[str, float],
        error: str | None = None,
        lease: GpuLease | None = None,
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
        infra_retryable: bool = False,
        attempt: int = 0,
    ) -> None:
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, external_job_id, metrics, error, "
                "created_at, "
                "updated_at, gpu_lease_id, target_id, target_server, gpu_device_ids, "
                "requested_gpu_count, actual_gpu_count, gpu_mode, gpu_tier, "
                "artifact_output_path, run_manifest_path, infra_retryable) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    self.execution_backend,
                    status,
                    attempt,
                    container_name,
                    dumps(metrics),
                    error,
                    now_iso(),
                    now_iso(),
                    lease.id if lease else None,
                    lease.target_id if lease else None,
                    lease.target_server if lease else None,
                    dumps(list(lease.device_ids)) if lease else dumps([]),
                    lease.requested_gpu_count if lease else 0,
                    lease.gpu_count if lease else 0,
                    lease.mode if lease else "",
                    lease.tier if lease else "",
                    artifact_output_path,
                    run_manifest_path,
                    int(infra_retryable),
                ),
            )

    async def _finalize_remote_result(
        self,
        *,
        submission_id: str,
        arch_hash: str,
        anti_multiplier: float,
        diversity_bonus: float,
        metrics: dict[str, float],
        component_review: ComponentReview | None = None,
    ) -> None:
        q_arch = max(0.0, min(1.0, float(metrics.get("q_arch", 0.0))))
        q_recipe = max(0.0, min(1.0, float(metrics.get("q_recipe", 0.5))))
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        scored = final_score(
            q_arch=q_arch,
            q_recipe=q_recipe,
            anti_cheat_multiplier=anti_multiplier,
            diversity_bonus=diversity_bonus,
            penalty=float(metrics.get("penalty", 0.0)),
            arch_weight=runtime_config.score_weights.final_architecture_weight,
            recipe_weight=runtime_config.score_weights.final_recipe_weight,
        )
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO scores("
                "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,"
                "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    scored.q_arch,
                    scored.q_recipe,
                    anti_multiplier,
                    diversity_bonus,
                    float(metrics.get("penalty", 0.0)),
                    scored.final_score,
                    dumps(metrics),
                    now_iso(),
                ),
            )
            await conn.execute(
                "UPDATE submissions SET status=?, arch_hash=?, updated_at=? WHERE id=?",
                (SubmissionStatus.COMPLETED.value, arch_hash, now_iso(), submission_id),
            )
        if self.settings.component_rewards_enabled and component_review is not None:
            component_review = await self._component_ownership_review(component_review)
            if component_review.ownership_decision is None:
                raise RuntimeError("component ownership decision was not created")
            metric_mean = max(0.0, min(1.0, float(metrics.get("q_recipe", q_recipe))))
            metric_std = max(
                0.0,
                float(
                    metrics.get(
                        "q_recipe_std",
                        metrics.get("metric_std", self.settings.training_metric_default_std),
                    )
                ),
            )
            component_result = await self.repository.record_component_result(
                submission_id=submission_id,
                project_kind=component_review.components.kind,
                family_hash=component_review.fingerprints.family_hash,
                arch_fingerprint=component_review.fingerprints.arch_fingerprint,
                behavior_fingerprint=component_review.fingerprints.behavior_fingerprint,
                training_hash=component_review.fingerprints.training_hash,
                requested_architecture_id=component_review.components.architecture_id,
                q_arch=q_arch,
                q_recipe=q_recipe,
                metric_mean=metric_mean,
                metric_std=metric_std,
                architecture_weight=runtime_config.reward_pools.architecture,
                training_weight=runtime_config.reward_pools.training,
                architecture_delta_abs=self.settings.architecture_improvement_min_delta_abs,
                architecture_delta_rel=self.settings.architecture_improvement_min_delta_rel,
                training_delta_abs=self.settings.training_improvement_min_delta_abs,
                training_delta_rel=self.settings.training_improvement_min_delta_rel,
                training_z_score=self.settings.training_improvement_z_score,
                architecture_transfer_delta_abs=self.settings.architecture_transfer_min_delta_abs,
                architecture_transfer_delta_rel=self.settings.architecture_transfer_min_delta_rel,
                training_transfer_delta_abs=self.settings.training_transfer_min_delta_abs,
                training_transfer_delta_rel=self.settings.training_transfer_min_delta_rel,
                transfer_confidence=self.settings.component_agent_transfer_confidence,
                metrics=metrics,
                semantic_signature=component_review.semantic_signature,
                ownership_decision=component_review.ownership_decision,
            )
            if component_result.get("held"):
                async with self.repository.database.connect() as conn:
                    await conn.execute(
                        "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                        (
                            SubmissionStatus.HELD.value,
                            "component attribution held for review",
                            now_iso(),
                            submission_id,
                        ),
                    )
            elif component_result.get("rejected"):
                async with self.repository.database.connect() as conn:
                    await conn.execute(
                        "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                        (
                            SubmissionStatus.REJECTED.value,
                            "component attribution rejected",
                            now_iso(),
                            submission_id,
                        ),
                    )

    async def poll_remote_jobs(self) -> list[str]:
        if self.lium is None:
            return []
        completed: list[str] = []
        async with self.repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.id as submission_id, s.code as code, s.filename as filename, "
                "s.metadata as metadata, e.external_job_id as external_job_id "
                "FROM submissions s JOIN eval_jobs e ON e.submission_id=s.id "
                "WHERE s.status=? AND e.level='remote' AND e.external_job_id IS NOT NULL",
                (SubmissionStatus.RUNNING.value,),
            )
        for row in rows:
            job = await self.lium.poll_job(str(row["external_job_id"]))
            if job.status != "completed":
                continue
            metadata = loads(str(row["metadata"]))
            snapshot = source_similarity.snapshot_from_submission(
                str(row["code"]),
                str(row["filename"]),
                metadata if isinstance(metadata, dict) else {},
            )
            code = source_similarity.primary_python_code(snapshot)
            report = inspect_code(code)
            arch_hash = sha256(":".join(sorted(report.ast_fingerprint)).encode()).hexdigest()
            previous = await self.repository.previous_codes(str(row["submission_id"]))
            anti = evaluate_anti_cheat(code, previous)
            await self._finalize_remote_result(
                submission_id=str(row["submission_id"]),
                arch_hash=arch_hash,
                anti_multiplier=anti.multiplier,
                diversity_bonus=anti.diversity_bonus,
                metrics=job.metrics,
            )
            completed.append(str(row["submission_id"]))
        return completed
