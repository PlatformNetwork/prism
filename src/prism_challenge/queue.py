from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from .config import PrismSettings
from .db import dumps
from .evaluator import llm_review, source_similarity
from .evaluator.anti_cheat import evaluate_anti_cheat
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
from .evaluator.distributed_contract import (
    check_distributed_contract,
    enforce_single_node_bound,
)
from .evaluator.interface import DEFAULT_TRAINING_ENTRYPOINT, PrismContext
from .evaluator.modes import execution_mode_from_value
from .evaluator.review_rules import ReviewRule, load_review_rules
from .evaluator.sandbox import SandboxViolation, inspect_code
from .evaluator.scoring import ScoreValidationError, score_prequential_bpb
from .evaluator.static_instantiation import check_build_model_static
from .gpu_scheduler import (
    GpuLease,
    GpuLeaseScheduler,
    lease_request_from_runtime,
    targets_from_settings,
)
from .models import SubmissionStatus
from .repository import PrismRepository, now_iso
from .sdk.executors.docker import DockerExecutor, DockerLimits, DockerMount, DockerRunSpec

DEFAULT_REVIEW_RULES = (
    ReviewRule("prism:no-secret-exfiltration", "Do not read, infer, print, or transmit secrets."),
    ReviewRule("prism:no-escape", "Do not use filesystem, process, or network escapes."),
    ReviewRule("prism:model-contract", "Only implement the Prism model and recipe contract."),
)
CONTAINER_EXECUTION_BACKENDS = frozenset(
    {"base_container", "base_gpu", "container_gpu", "docker_gpu"}
)
SUPPORTED_EXECUTION_BACKENDS = CONTAINER_EXECUTION_BACKENDS

logger = logging.getLogger(__name__)


class EvalWallTimeExceeded(RuntimeError):
    """Raised when an eval exceeds the orchestration wall-time backstop and is force-killed.

    The inner docker run has its own ``base_eval_hard_timeout_seconds``; this backstop guards the
    orchestration layer so a thread that never returns (hung CUDA call, wedged docker daemon)
    cannot hold its GPU lease forever. On this error the container is reaped and the lease released.
    """


EvaluatorFactory = Callable[[PrismSettings, PrismContext], PrismContainerEvaluator]


def _default_evaluator_factory(
    settings: PrismSettings, ctx: PrismContext
) -> PrismContainerEvaluator:
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def _is_v2_run_manifest(manifest: Any) -> bool:
    """True when the container returned a challenge-authored prism_run_manifest.v2 with metrics."""
    return (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == "prism_run_manifest.v2"
        and isinstance(manifest.get("metrics"), dict)
    )


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-None submission metadata value among ``keys``."""
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


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


class PrismWorker:
    def __init__(
        self,
        repository: PrismRepository,
        ctx: PrismContext,
        *,
        execution_backend: str = "base_gpu",
        settings: PrismSettings | None = None,
        evaluator_factory: EvaluatorFactory | None = None,
    ) -> None:
        if execution_backend not in SUPPORTED_EXECUTION_BACKENDS:
            raise ValueError(f"Unsupported execution backend: {execution_backend}")
        self.repository = repository
        self.ctx = ctx
        self.execution_backend = execution_backend
        self.settings = settings or PrismSettings()
        self._evaluator_factory = evaluator_factory or _default_evaluator_factory

    async def process_next(self) -> str | None:
        submission = await self.repository.claim_next()
        if submission is None:
            return None
        return await self._process_claimed(submission)

    async def process_submission(self, submission_id: str) -> str | None:
        """Process exactly the submission assigned by the coordination plane.

        Claims the SPECIFIC pending submission (CAS on status) and runs the same re-execution path
        as :meth:`process_next`. A submission that is not pending (already terminal, or in-flight on
        another validator) is a no-op returning ``None``, so a busy validator never starts a second
        run and re-posting a completed assignment never re-dispatches the broker or mutates the
        recorded result.
        """
        submission = await self.repository.claim_submission(submission_id)
        if submission is None:
            return None
        return await self._process_claimed(submission)

    async def _process_claimed(self, submission: dict[str, Any]) -> str:
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

    async def _process_container(
        self,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
    ) -> str:
        # Static gates run FIRST: a sandbox / param-cap / distributed-contract rejection precedes
        # and SKIPS the LLM review entirely -- no llm_reviews/llm_review_events row and no GPU
        # work for a statically-rejected bundle (VAL-LLM-020, VAL-CONTRACT-018).
        try:
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code_for_eval = self._entrypoint_code(snapshot, component_review.components.entrypoint)
            if not code_for_eval.strip():
                await self._reject_submission(submission_id, "submission contains no Python source")
                return submission_id
            report = self._inspect_project_snapshot(snapshot, code_for_eval)
            await self._static_model_instantiation_check(snapshot, component_review)
            self._distributed_contract_check(snapshot, component_review, metadata)
        except (SandboxViolation, SyntaxError) as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id
        except Exception as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id

        # LLM hard gate + plagiarism run only AFTER the static gates have passed.
        try:
            review = await self._review_static_submission(
                submission_id=submission_id,
                snapshot=snapshot,
                component_review=component_review,
                code_for_eval=code_for_eval,
                filename=filename,
                hotkey=hotkey,
                code_hash=code_hash,
            )
            if review.rejected:
                await self._reject_submission(submission_id, review.reason or "review rejected")
                return submission_id
            if review.held:
                return submission_id
            code = review.code
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
                "base_eval_gpu_count": lease.gpu_count,
                "base_eval_gpu_type": runtime_config.gpu_policy.gpu_type,
                "base_eval_gpu_server": lease.target_server,
                "base_eval_gpu_device_ids": lease.device_ids,
            }
        )
        evaluator = self._evaluator_factory(effective_settings, self.ctx)
        attempt = (
            await self.repository.container_job_attempt_count(submission_id, self.execution_backend)
            + 1
        )
        components = component_review.components
        try:
            result = await self._evaluate_within_wall_time(
                evaluator,
                submission_id=submission_id,
                code=code,
                code_hash=code_hash,
                arch_hash=arch_hash,
                files=snapshot.files,
                components=components,
                gpu_lease=lease,
                execution_mode=execution_mode,
                attempt=attempt,
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
            if not _is_v2_run_manifest(result.run_manifest):
                await self._fail_submission(
                    submission_id,
                    "container run produced no challenge-authored prism_run_manifest.v2",
                )
                return submission_id
            await self._finalize_container_score(
                submission_id=submission_id,
                arch_hash=arch_hash,
                anti=anti,
                manifest=cast(dict[str, Any], result.run_manifest),
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
        except EvalWallTimeExceeded as exc:
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
            # Reap (force-kill) the eval container FIRST so an overrunning or wedged job stops
            # consuming the GPU, THEN release the lease. Ordering matters: releasing before the
            # kill could hand the device to the next eval while the old process is still resident
            # (architecture.md 4.3, 10; VAL-HARNESS-027). Both steps are best-effort.
            await asyncio.to_thread(evaluator.reap_job, submission_id)
            await scheduler.release_for_submission(submission_id, "container job finished")

    async def _evaluate_within_wall_time(
        self,
        evaluator: PrismContainerEvaluator,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        files: tuple[source_similarity.SourceFile, ...],
        components: PrismProjectComponents,
        gpu_lease: GpuLease,
        execution_mode: Any,
        attempt: int,
    ) -> Any:
        eval_call = asyncio.to_thread(
            evaluator.evaluate,
            submission_id=submission_id,
            code=code,
            code_hash=code_hash,
            arch_hash=arch_hash,
            backend=self.execution_backend,
            files=files,
            architecture_entrypoint=components.architecture_entrypoint,
            training_entrypoint=components.training_entrypoint,
            build_model_symbol=components.build_model_symbol,
            train_symbol=components.train_symbol,
            gpu_lease=gpu_lease,
            execution_mode=execution_mode,
            attempt=attempt,
        )
        timeout = self.settings.resolved_orchestration_timeout_seconds
        try:
            return await asyncio.wait_for(eval_call, timeout=timeout)
        except TimeoutError as exc:
            raise EvalWallTimeExceeded(
                f"eval exceeded orchestration wall-time of {timeout:g}s; container force-killed"
            ) from exc

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
            require_contract=False,
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

    async def _static_model_instantiation_check(
        self,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
    ) -> None:
        """Instantiate build_model under the forced seed before any GPU work.

        Rejects non-nn.Module returns, surfaces construction errors cleanly, and bounds hostile
        construction (infinite loop / memory balloon) at the static phase.
        """
        components = component_review.components
        entrypoint = components.architecture_entrypoint or components.entrypoint
        files = {file.path: file.content for file in snapshot.python_files}
        await asyncio.to_thread(
            check_build_model_static,
            files,
            entrypoint,
            ctx=self.ctx,
            build_model_symbol=components.build_model_symbol,
            timeout_seconds=self.settings.static_instantiation_timeout_seconds,
            memory_headroom_bytes=self.settings.static_instantiation_memory_headroom_bytes,
        )

    def _distributed_contract_check(
        self,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
        metadata: dict[str, Any],
    ) -> None:
        """Multi-GPU static contract (architecture.md section 8), before any GPU work.

        Statically verifies the training script uses the distributed primitives + a rank-0 write
        guard (per ``distributed_contract_policy``) and enforces the single-node bound (reject a
        ``gpu_count > 8`` / multi-node request). Raises SandboxViolation on a violation, which the
        caller converts into a clean ``rejected`` outcome with no GPU lease/job.
        """
        components = component_review.components
        training_entry = components.training_entrypoint or DEFAULT_TRAINING_ENTRYPOINT
        training_code = next(
            (file.content for file in snapshot.python_files if file.path == training_entry),
            "",
        )
        if training_code:
            check_distributed_contract(
                training_code,
                artifact_path=training_entry,
                policy=self.settings.distributed_contract_policy,
            )
        enforce_single_node_bound(
            _metadata_value(metadata, "gpu_count", "num_gpus", "requested_gpu_count", "gpus"),
            num_nodes=_metadata_value(metadata, "num_nodes", "nnodes", "nodes"),
            max_gpu_count=self.settings.base_eval_max_gpu_count,
        )

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

    def _entrypoint_code(self, snapshot: source_similarity.SourceSnapshot, entrypoint: str) -> str:
        match = next((file for file in snapshot.files if file.path == entrypoint), None)
        if match is None:
            raise ValueError(f"Prism project entrypoint not found: {entrypoint}")
        return match.content

    async def _review_static_submission(
        self,
        *,
        submission_id: str,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
        code_for_eval: str,
        filename: str,
        hotkey: str,
        code_hash: str,
    ) -> StaticReviewOutcome:
        # Invoked ONLY after the static AST sandbox / param-cap / distributed-contract gates have
        # passed, so a static rejection never reaches (or records any event for) the LLM gate.
        await self.repository.record_llm_review_event(
            submission_id=submission_id,
            state="received",
            actor="system",
            tool_name="submission_receiver",
            payload={"filename": filename, "code_hash": code_hash},
            reason="submission received for LLM review flow",
            idempotency_key="state:received",
        )
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
            logger.warning("submission %s held by LLM review: %s", submission_id, safety.reason)
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
            # v2 has no operator hold-resolution surface (the NAS review endpoints were
            # decommissioned), so the borderline-duplicate quarantine band would strand a
            # submission in HELD forever. Fold it into a terminal rejection at ingress;
            # exact-source-hash dedup is the same path.
            rejected = duplicate.rejected or duplicate.held
            violations = ["duplicate_similarity"] if rejected else []
            await self.repository.store_plagiarism_review(
                submission_id=submission_id,
                candidate_submission_id=duplicate.candidate.submission_id,
                similarity=float(duplicate.report["source_similarity"]),
                verdict=rejected,
                reason=duplicate.reason,
                violations=violations,
                report=duplicate.report,
            )
            if rejected:
                return StaticReviewOutcome(
                    code_for_eval,
                    True,
                    reason=duplicate.reason,
                    violations=tuple(violations),
                )
            return StaticReviewOutcome(code_for_eval, False)

        return StaticReviewOutcome(code_for_eval, False)

    async def _reject_submission(self, submission_id: str, reason: str) -> None:
        logger.warning("submission %s rejected: %s", submission_id, reason)
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
            base_url=self.settings.openrouter_base_url,
            model=self.settings.openrouter_model,
            api_key=self.settings.openrouter_api_key_value(),
            api_key_file=self.settings.openrouter_api_key_file,
            timeout_seconds=self.settings.llm_review_timeout_seconds,
            temperature=self.settings.llm_review_temperature,
            max_tokens=self.settings.llm_review_max_tokens,
            max_retries=self.settings.llm_review_max_retries,
            max_source_chars=self.settings.llm_review_max_source_chars,
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
                    labels={"base.job": submission_id, "base.task": "plagiarism"},
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

    async def _finalize_container_score(
        self,
        *,
        submission_id: str,
        arch_hash: str,
        anti: Any,
        manifest: dict[str, Any],
    ) -> None:
        """Finalize a container run using the CHALLENGE-OWNED prequential bits-per-byte score.

        The authoritative score is recomputed by ``scoring.score_prequential_bpb`` from the
        challenge-authored ``prism_run_manifest.v2`` (the legacy NAS q_arch/q_recipe derivation and
        the component-reward branching are NOT on this path, so they no longer affect the score).
        A degenerate run that cannot yield a finite/positive bpb is failed rather than scored.
        """
        try:
            score = score_prequential_bpb(manifest)
        except ScoreValidationError as exc:
            await self._fail_submission(submission_id, f"prequential scoring failed: {exc}")
            return
        anti_multiplier = max(0.0, min(1.0, float(getattr(anti, "multiplier", 1.0))))
        final_score_value = max(0.0, score.final_score * anti_multiplier)
        metrics_payload = score.metrics_payload()
        metrics_payload["arch_hash"] = arch_hash
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO scores("
                "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,"
                "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    final_score_value,
                    0.0,
                    score.anti_cheat_multiplier * anti_multiplier,
                    0.0,
                    0.0,
                    final_score_value,
                    dumps(metrics_payload),
                    now_iso(),
                ),
            )
            await conn.execute(
                "UPDATE submissions SET status=?, arch_hash=?, updated_at=? WHERE id=?",
                (SubmissionStatus.COMPLETED.value, arch_hash, now_iso(), submission_id),
            )
