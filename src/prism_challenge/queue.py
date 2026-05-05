from __future__ import annotations

from hashlib import sha256
from uuid import uuid4

from .db import dumps
from .evaluator.anti_cheat import evaluate_anti_cheat
from .evaluator.interface import PrismContext
from .evaluator.l1_syntax import validate_l1
from .evaluator.l2_proxy import score_l2
from .evaluator.l3_train import score_l3
from .evaluator.l4_benchmark import score_l4
from .evaluator.lium_client import LiumClient, LiumJob
from .evaluator.sandbox import inspect_code, load_submission_contract
from .evaluator.scoring import final_score, score_recipe
from .models import SubmissionStatus
from .repository import PrismRepository, now_iso


class PrismWorker:
    def __init__(
        self,
        repository: PrismRepository,
        ctx: PrismContext,
        lium: LiumClient,
        *,
        execution_backend: str = "local_cpu",
    ) -> None:
        self.repository = repository
        self.ctx = ctx
        self.lium = lium
        self.execution_backend = execution_backend

    async def process_next(self) -> str | None:
        submission = await self.repository.claim_next()
        if submission is None:
            return None
        submission_id = str(submission["id"])
        code = str(submission["code"])
        if self.execution_backend == "remote_provider":
            return await self._submit_remote(submission_id, code)
        return await self._process_local(submission_id, code)

    async def _process_local(self, submission_id: str, code: str) -> str:
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
                scored = final_score(
                    q_arch=l4.q_arch,
                    q_recipe=recipe_score,
                    anti_cheat_multiplier=anti.multiplier,
                    diversity_bonus=anti.diversity_bonus,
                    penalty=penalty,
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

    async def _submit_remote(self, submission_id: str, code: str) -> str:
        try:
            report = inspect_code(code)
            code_hash = sha256(code.encode()).hexdigest()
            arch_basis = ":".join(sorted(report.ast_fingerprint))
            arch_hash = sha256(arch_basis.encode()).hexdigest()
            previous = await self.repository.previous_codes(submission_id)
            anti = evaluate_anti_cheat(code, previous)
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
                )
            return submission_id
        except Exception as exc:
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, str(exc), now_iso(), submission_id),
                )
            return submission_id

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

    async def _finalize_remote_result(
        self,
        *,
        submission_id: str,
        arch_hash: str,
        anti_multiplier: float,
        diversity_bonus: float,
        metrics: dict[str, float],
    ) -> None:
        q_arch = max(0.0, min(1.0, float(metrics.get("q_arch", 0.0))))
        q_recipe = max(0.0, min(1.0, float(metrics.get("q_recipe", 0.5))))
        scored = final_score(
            q_arch=q_arch,
            q_recipe=q_recipe,
            anti_cheat_multiplier=anti_multiplier,
            diversity_bonus=diversity_bonus,
            penalty=float(metrics.get("penalty", 0.0)),
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

    async def poll_remote_jobs(self) -> list[str]:
        completed: list[str] = []
        async with self.repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.id as submission_id, s.code as code, "
                "e.external_job_id as external_job_id "
                "FROM submissions s JOIN eval_jobs e ON e.submission_id=s.id "
                "WHERE s.status=? AND e.level='remote' AND e.external_job_id IS NOT NULL",
                (SubmissionStatus.RUNNING.value,),
            )
        for row in rows:
            job = await self.lium.poll_job(str(row["external_job_id"]))
            if job.status != "completed":
                continue
            report = inspect_code(str(row["code"]))
            arch_hash = sha256(":".join(sorted(report.ast_fingerprint)).encode()).hexdigest()
            previous = await self.repository.previous_codes(str(row["submission_id"]))
            anti = evaluate_anti_cheat(str(row["code"]), previous)
            await self._finalize_remote_result(
                submission_id=str(row["submission_id"]),
                arch_hash=arch_hash,
                anti_multiplier=anti.multiplier,
                diversity_bonus=anti.diversity_bonus,
                metrics=job.metrics,
            )
            completed.append(str(row["submission_id"]))
        return completed
