from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, SupportsFloat, cast
from uuid import uuid4

import aiosqlite

from .db import Database, dumps, loads
from .models import (
    EvaluationAssignmentResponse,
    EvaluationAssignmentStatus,
    SubmissionCreate,
    SubmissionResponse,
    SubmissionStatus,
    SubmissionStatusResponse,
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def epoch_id_for(timestamp: datetime, epoch_seconds: int) -> int:
    return int(timestamp.timestamp()) // epoch_seconds


async def ensure_epoch(conn: aiosqlite.Connection, epoch_id: int, epoch_seconds: int) -> None:
    starts = datetime.fromtimestamp(epoch_id * epoch_seconds, UTC)
    ends = datetime.fromtimestamp((epoch_id + 1) * epoch_seconds, UTC)
    await conn.execute(
        "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
        (epoch_id, starts.isoformat(), ends.isoformat(), "open"),
    )


class PrismRepository:
    def __init__(self, database: Database, epoch_seconds: int) -> None:
        self.database = database
        self.epoch_seconds = epoch_seconds

    async def create_submission(self, hotkey: str, request: SubmissionCreate) -> SubmissionResponse:
        created = datetime.now(UTC)
        epoch_id = epoch_id_for(created, self.epoch_seconds)
        submission_id = str(uuid4())
        code_hash = sha256(request.code.encode()).hexdigest()
        async with self.database.connect() as conn:
            await ensure_epoch(conn, epoch_id, self.epoch_seconds)
            await conn.execute(
                "INSERT OR IGNORE INTO miners(hotkey, first_seen, last_seen) VALUES (?, ?, ?)",
                (hotkey, created.isoformat(), created.isoformat()),
            )
            await conn.execute(
                "UPDATE miners SET last_seen=? WHERE hotkey=?", (created.isoformat(), hotkey)
            )
            await conn.execute(
                "INSERT INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    epoch_id,
                    request.filename,
                    request.code,
                    code_hash,
                    dumps(request.metadata),
                    SubmissionStatus.PENDING.value,
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, metrics, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    "l1",
                    "pending",
                    "{}",
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
        return SubmissionResponse(
            id=submission_id,
            hotkey=hotkey,
            epoch_id=epoch_id,
            status=SubmissionStatus.PENDING,
            code_hash=code_hash,
            created_at=created,
        )

    async def get_submission(self, submission_id: str) -> SubmissionStatusResponse | None:
        async with self.database.connect() as conn:
            row = await conn.execute_fetchall(
                "SELECT s.*, sc.q_arch, sc.q_recipe, sc.final_score FROM submissions s "
                "LEFT JOIN scores sc ON sc.submission_id=s.id WHERE s.id=?",
                (submission_id,),
            )
        if not row:
            return None
        item = list(row)[0]
        return SubmissionStatusResponse(
            id=item["id"],
            hotkey=item["hotkey"],
            epoch_id=item["epoch_id"],
            status=SubmissionStatus(item["status"]),
            code_hash=item["code_hash"],
            created_at=datetime.fromisoformat(item["created_at"]),
            error=item["error"],
            final_score=item["final_score"],
            q_arch=item["q_arch"],
            q_recipe=item["q_recipe"],
        )

    async def previous_codes(self, current_submission_id: str) -> list[str]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT code FROM submissions WHERE id != ? ORDER BY created_at DESC",
                (current_submission_id,),
            )
        return [str(row["code"]) for row in rows]

    async def store_source_snapshot(
        self,
        *,
        submission_id: str,
        hotkey: str,
        code_hash: str,
        payload: dict[str, Any],
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO submission_sources("
                "submission_id, hotkey, code_hash, files, ast_features, token_shingles, "
                "fingerprint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    code_hash,
                    dumps(payload["files"]),
                    dumps(payload["ast_features"]),
                    dumps(payload["token_shingles"]),
                    str(payload["fingerprint"]),
                    now_iso(),
                ),
            )

    async def source_snapshots(
        self, *, exclude_submission_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM submission_sources"
        params: tuple[str, ...] = ()
        if exclude_submission_id:
            query += " WHERE submission_id != ?"
            params = (exclude_submission_id,)
        query += " ORDER BY created_at DESC"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(query, params)
        out = []
        for row in rows:
            item = dict(row)
            item["files"] = loads(item["files"])
            item["ast_features"] = loads(item["ast_features"])
            item["token_shingles"] = loads(item["token_shingles"])
            out.append(item)
        return out

    async def store_plagiarism_review(
        self,
        *,
        submission_id: str,
        candidate_submission_id: str | None,
        similarity: float,
        verdict: bool,
        reason: str,
        violations: list[str],
        report: dict[str, Any],
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO plagiarism_reviews("
                "submission_id, candidate_submission_id, similarity, verdict, reason, violations, "
                "report, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    candidate_submission_id,
                    similarity,
                    int(verdict),
                    reason,
                    dumps(violations),
                    dumps(report),
                    now_iso(),
                ),
            )

    async def store_llm_review(
        self,
        *,
        submission_id: str,
        approved: bool,
        reason: str,
        violations: list[str],
        confidence: float,
        raw: dict[str, Any],
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO llm_reviews("
                "submission_id, approved, reason, violations, confidence, raw, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    int(approved),
                    reason,
                    dumps(violations),
                    confidence,
                    dumps(raw),
                    now_iso(),
                ),
            )

    async def leaderboard(self, epoch_id: int, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id "
                "WHERE s.epoch_id=? ORDER BY sc.final_score DESC LIMIT ?",
                (epoch_id, limit),
            )
        return [dict(row) for row in rows]

    async def score_rows(self, epoch_id: int) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id WHERE s.epoch_id=?",
                (epoch_id,),
            )
        return [dict(row) for row in rows]

    async def component_weight_rows(
        self,
        *,
        architecture_weight: float,
        training_weight: float,
    ) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT owner_hotkey AS hotkey, q_arch_best * ? AS score "
                "FROM architecture_families WHERE q_arch_best > 0 "
                "UNION ALL "
                "SELECT owner_hotkey AS hotkey, q_recipe * ? AS score "
                "FROM training_variants WHERE is_current_best=1 AND q_recipe > 0",
                (architecture_weight, training_weight),
            )
        return [dict(row) for row in rows]

    async def record_component_result(
        self,
        *,
        submission_id: str,
        project_kind: str,
        family_hash: str,
        arch_fingerprint: str,
        behavior_fingerprint: str,
        training_hash: str,
        requested_architecture_id: str | None,
        q_arch: float,
        q_recipe: float,
        metric_mean: float,
        metric_std: float,
        architecture_weight: float,
        training_weight: float,
        architecture_delta_abs: float,
        architecture_delta_rel: float,
        training_delta_abs: float,
        training_delta_rel: float,
        training_z_score: float,
        metrics: dict[str, float],
    ) -> dict[str, object]:
        now = now_iso()
        async with self.database.connect() as conn:
            submission_rows = await conn.execute_fetchall(
                "SELECT hotkey FROM submissions WHERE id=?", (submission_id,)
            )
            submission_list = list(submission_rows)
            if not submission_list:
                raise ValueError("submission not found")
            hotkey = str(submission_list[0]["hotkey"])
            family = await self._component_family(
                conn, requested_architecture_id=requested_architecture_id, family_hash=family_hash
            )
            if project_kind == "training_for_arch" and requested_architecture_id is None:
                raise ValueError("training_for_arch submissions require architecture_id")

            accepted_architecture = False
            arch_points = 0.0
            if family is None:
                if project_kind == "training_for_arch":
                    raise ValueError("requested architecture_id was not found")
                architecture_id = str(uuid4())
                await conn.execute(
                    "INSERT INTO architecture_families("
                    "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey,"
                    "owner_submission_id, canonical_submission_id, q_arch_best, created_at,"
                    "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        architecture_id,
                        family_hash,
                        arch_fingerprint,
                        behavior_fingerprint,
                        hotkey,
                        submission_id,
                        submission_id,
                        q_arch,
                        now,
                        now,
                    ),
                )
                accepted_architecture = True
                arch_points = architecture_weight * q_arch
            else:
                architecture_id = str(family["id"])
                if str(family["family_hash"]) != family_hash:
                    raise ValueError("submission architecture does not match requested family")
                if _meaningful_improvement(
                    q_arch,
                    float(cast(SupportsFloat, family["q_arch_best"])),
                    new_std=0.0,
                    old_std=0.0,
                    min_delta_abs=architecture_delta_abs,
                    min_delta_rel=architecture_delta_rel,
                    z_score=0.0,
                ):
                    await conn.execute(
                        "UPDATE architecture_families SET q_arch_best=?, "
                        "canonical_submission_id=?, updated_at=? WHERE id=?",
                        (q_arch, submission_id, now, architecture_id),
                    )

            training_variant_id: str | None = None
            accepted_training = False
            training_points = 0.0
            if project_kind != "architecture_only":
                current = await self._current_training_variant(conn, architecture_id)
                existing_variant = await self._training_variant(
                    conn, architecture_id, training_hash
                )
                accepted_training = existing_variant is None and (
                    current is None
                    or _meaningful_improvement(
                        metric_mean,
                        float(cast(SupportsFloat, current["metric_mean"])),
                        new_std=metric_std,
                        old_std=float(cast(SupportsFloat, current["metric_std"])),
                        min_delta_abs=training_delta_abs,
                        min_delta_rel=training_delta_rel,
                        z_score=training_z_score,
                    )
                )
                if existing_variant is None:
                    training_variant_id = str(uuid4())
                    if accepted_training:
                        await conn.execute(
                            "UPDATE training_variants SET is_current_best=0 "
                            "WHERE architecture_id=?",
                            (architecture_id,),
                        )
                    await conn.execute(
                        "INSERT INTO training_variants("
                        "id, architecture_id, training_hash, owner_hotkey, submission_id,"
                        "q_recipe, metric_mean, metric_std, is_current_best, created_at,"
                        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            training_variant_id,
                            architecture_id,
                            training_hash,
                            hotkey,
                            submission_id,
                            q_recipe,
                            metric_mean,
                            max(0.0, metric_std),
                            int(accepted_training),
                            now,
                            now,
                        ),
                    )
                else:
                    training_variant_id = str(existing_variant["id"])
                if accepted_training:
                    training_points = training_weight * q_recipe

            await conn.execute(
                "INSERT OR REPLACE INTO component_scores("
                "submission_id, architecture_id, training_variant_id, project_kind, arch_points,"
                "training_points, accepted_architecture, accepted_training, metrics, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    architecture_id,
                    training_variant_id,
                    project_kind,
                    arch_points,
                    training_points,
                    int(accepted_architecture),
                    int(accepted_training),
                    dumps(metrics),
                    now,
                ),
            )
        return {
            "architecture_id": architecture_id,
            "training_variant_id": training_variant_id,
            "accepted_architecture": accepted_architecture,
            "accepted_training": accepted_training,
            "arch_points": arch_points,
            "training_points": training_points,
        }

    async def list_architectures(self, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM architecture_families ORDER BY q_arch_best DESC, created_at LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def list_training_variants(
        self, architecture_id: str | None = None, limit: int = 100
    ) -> list[dict[str, object]]:
        query = "SELECT * FROM training_variants"
        params: tuple[object, ...]
        if architecture_id is not None:
            query += " WHERE architecture_id=?"
            params = (architecture_id, limit)
        else:
            params = (limit,)
        query += " ORDER BY is_current_best DESC, metric_mean DESC, created_at LIMIT ?"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(query, params)
        return [dict(row) for row in rows]

    async def claim_next(self) -> dict[str, object] | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM submissions WHERE status=? ORDER BY created_at LIMIT 1",
                (SubmissionStatus.PENDING.value,),
            )
            if not rows:
                return None
            row = list(rows)[0]
            await conn.execute(
                "UPDATE submissions SET status=?, updated_at=? WHERE id=?",
                (SubmissionStatus.RUNNING.value, now_iso(), row["id"]),
            )
        data = dict(cast(Any, row))
        metadata = loads(str(data.get("metadata", "{}")))
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return data

    async def create_assignment(
        self,
        *,
        submission_id: str,
        validator_hotkey: str,
        arch_hash: str,
        timeout_seconds: int,
    ) -> EvaluationAssignmentResponse:
        now = datetime.now(UTC)
        deadline = now + timedelta(seconds=timeout_seconds)
        async with self.database.connect() as conn:
            attempt_rows = await conn.execute_fetchall(
                "SELECT COALESCE(MAX(attempt), 0) AS attempt FROM evaluation_assignments "
                "WHERE submission_id=?",
                (submission_id,),
            )
            attempt = int(list(attempt_rows)[0]["attempt"]) + 1
            assignment_id = str(uuid4())
            await conn.execute(
                "INSERT INTO evaluation_assignments("
                "id, submission_id, validator_hotkey, status, attempt, deadline_at, arch_hash, "
                "metrics, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    assignment_id,
                    submission_id,
                    validator_hotkey,
                    EvaluationAssignmentStatus.ASSIGNED.value,
                    attempt,
                    deadline.isoformat(),
                    arch_hash,
                    "{}",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            rows = await conn.execute_fetchall(
                "SELECT a.*, s.code, s.filename, s.metadata, s.code_hash "
                "FROM evaluation_assignments a JOIN submissions s ON s.id=a.submission_id "
                "WHERE a.id=?",
                (assignment_id,),
            )
        return _assignment_response(dict(list(rows)[0]))

    async def get_assignment(self, assignment_id: str) -> EvaluationAssignmentResponse | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT a.*, s.code, s.filename, s.metadata, s.code_hash "
                "FROM evaluation_assignments a JOIN submissions s ON s.id=a.submission_id "
                "WHERE a.id=?",
                (assignment_id,),
            )
        return _assignment_response(dict(list(rows)[0])) if rows else None

    async def active_assignment_for_validator(
        self, validator_hotkey: str
    ) -> EvaluationAssignmentResponse | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT a.*, s.code, s.filename, s.metadata, s.code_hash "
                "FROM evaluation_assignments a JOIN submissions s ON s.id=a.submission_id "
                "WHERE a.validator_hotkey=? AND a.status IN (?, ?, ?) "
                "ORDER BY a.created_at LIMIT 1",
                (
                    validator_hotkey,
                    EvaluationAssignmentStatus.ASSIGNED.value,
                    EvaluationAssignmentStatus.ACCEPTED.value,
                    EvaluationAssignmentStatus.RUNNING.value,
                ),
            )
        return _assignment_response(dict(list(rows)[0])) if rows else None

    async def set_assignment_status(
        self,
        assignment_id: str,
        status: EvaluationAssignmentStatus,
        *,
        error: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "UPDATE evaluation_assignments SET status=?, error=COALESCE(?, error), "
                "metrics=COALESCE(?, metrics), updated_at=? WHERE id=?",
                (
                    status.value,
                    error,
                    dumps(metrics) if metrics is not None else None,
                    now_iso(),
                    assignment_id,
                ),
            )

    async def expire_stale_assignments(self, max_attempts: int) -> list[str]:
        now = now_iso()
        expired_submission_ids: list[str] = []
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM evaluation_assignments WHERE status IN (?, ?, ?) "
                "AND deadline_at < ?",
                (
                    EvaluationAssignmentStatus.ASSIGNED.value,
                    EvaluationAssignmentStatus.ACCEPTED.value,
                    EvaluationAssignmentStatus.RUNNING.value,
                    now,
                ),
            )
            for row in rows:
                submission_id = str(row["submission_id"])
                await conn.execute(
                    "UPDATE evaluation_assignments SET status=?, error=?, updated_at=? WHERE id=?",
                    (
                        EvaluationAssignmentStatus.EXPIRED.value,
                        "validator deadline expired",
                        now,
                        row["id"],
                    ),
                )
                attempts = await conn.execute_fetchall(
                    "SELECT COUNT(*) AS count FROM evaluation_assignments WHERE submission_id=?",
                    (submission_id,),
                )
                if int(list(attempts)[0]["count"]) >= max_attempts:
                    await conn.execute(
                        "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                        (
                            SubmissionStatus.FAILED.value,
                            "validator assignment attempts exhausted",
                            now,
                            submission_id,
                        ),
                    )
                else:
                    await conn.execute(
                        "UPDATE submissions SET status=?, updated_at=? WHERE id=?",
                        (SubmissionStatus.PENDING.value, now, submission_id),
                    )
                expired_submission_ids.append(submission_id)
        return expired_submission_ids

    async def _component_family(
        self,
        conn: aiosqlite.Connection,
        *,
        requested_architecture_id: str | None,
        family_hash: str,
    ) -> aiosqlite.Row | None:
        if requested_architecture_id is not None:
            rows = await conn.execute_fetchall(
                "SELECT * FROM architecture_families WHERE id=?", (requested_architecture_id,)
            )
        else:
            rows = await conn.execute_fetchall(
                "SELECT * FROM architecture_families WHERE family_hash=?", (family_hash,)
            )
        rows_list = list(rows)
        return rows_list[0] if rows_list else None

    async def _current_training_variant(
        self, conn: aiosqlite.Connection, architecture_id: str
    ) -> aiosqlite.Row | None:
        rows = await conn.execute_fetchall(
            "SELECT * FROM training_variants WHERE architecture_id=? AND is_current_best=1",
            (architecture_id,),
        )
        rows_list = list(rows)
        return rows_list[0] if rows_list else None

    async def _training_variant(
        self, conn: aiosqlite.Connection, architecture_id: str, training_hash: str
    ) -> aiosqlite.Row | None:
        rows = await conn.execute_fetchall(
            "SELECT * FROM training_variants WHERE architecture_id=? AND training_hash=?",
            (architecture_id, training_hash),
        )
        rows_list = list(rows)
        return rows_list[0] if rows_list else None


def _meaningful_improvement(
    new_value: float,
    old_value: float,
    *,
    new_std: float,
    old_std: float,
    min_delta_abs: float,
    min_delta_rel: float,
    z_score: float,
) -> bool:
    delta = new_value - old_value
    variance_delta = z_score * ((max(0.0, new_std) ** 2 + max(0.0, old_std) ** 2) ** 0.5)
    required = max(min_delta_abs, abs(old_value) * min_delta_rel, variance_delta)
    return delta > required


def _assignment_response(row: dict[str, Any]) -> EvaluationAssignmentResponse:
    metadata = loads(str(row.get("metadata")))
    return EvaluationAssignmentResponse(
        id=str(row["id"]),
        submission_id=str(row["submission_id"]),
        validator_hotkey=str(row["validator_hotkey"]),
        status=EvaluationAssignmentStatus(str(row["status"])),
        attempt=int(row["attempt"]),
        deadline_at=datetime.fromisoformat(str(row["deadline_at"])),
        code=str(row["code"]),
        filename=str(row["filename"]),
        metadata=metadata if isinstance(metadata, dict) else {},
        code_hash=str(row["code_hash"]),
        arch_hash=str(row["arch_hash"]),
    )
