from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, SupportsFloat, cast
from uuid import uuid4

import aiosqlite

from .db import Database, dumps, loads
from .evaluator.schemas import (
    DeterministicEvidence,
)
from .evaluator.scoring import LeaderboardRow, rank_leaderboard
from .models import (
    SubmissionCreate,
    SubmissionResponse,
    SubmissionStatus,
    SubmissionStatusResponse,
)
from .runtime_config import RuntimePolicy, resolve_runtime_policy


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
    def __init__(
        self,
        database: Database,
        epoch_seconds: int,
        worker_claim_timeout_seconds: int = 900,
        held_review_timeout_seconds: int = 86400,
    ) -> None:
        self.database = database
        self.epoch_seconds = epoch_seconds
        self.worker_claim_timeout_seconds = worker_claim_timeout_seconds
        self.held_review_timeout_seconds = held_review_timeout_seconds

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
                "SELECT s.*, sc.q_arch, sc.q_recipe, sc.final_score, "
                "sc.anti_cheat_multiplier, sc.diversity_bonus, sc.penalty FROM submissions s "
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
            anti_cheat_multiplier=item["anti_cheat_multiplier"],
            diversity_bonus=item["diversity_bonus"],
            penalty=item["penalty"],
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

    async def source_similarity_candidates(
        self, *, exclude_submission_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT ss.*, cs.architecture_id, cs.architecture_graph, "
            "af.canonical_graph_hash AS architecture_graph_hash "
            "FROM submission_sources ss "
            "LEFT JOIN component_signatures cs ON cs.submission_id=ss.submission_id "
            "LEFT JOIN architecture_families af ON af.id=cs.architecture_id"
        )
        params: tuple[str, ...] = ()
        if exclude_submission_id:
            query += " WHERE ss.submission_id != ?"
            params = (exclude_submission_id,)
        query += " ORDER BY ss.created_at DESC"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(query, params)
        out = []
        for row in rows:
            item = dict(row)
            item["files"] = loads(item["files"])
            item["ast_features"] = loads(item["ast_features"])
            item["token_shingles"] = loads(item["token_shingles"])
            graph = loads(item.get("architecture_graph")) if item.get("architecture_graph") else {}
            item["architecture_graph"] = graph if isinstance(graph, dict) else {}
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
        mermaid: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        held: bool = False,
    ) -> None:
        raw_mermaid = raw.get("mermaid") if isinstance(raw.get("mermaid"), dict) else None
        mermaid_text = mermaid or (str(raw_mermaid.get("mermaid")) if raw_mermaid else None)
        raw_verdict = raw.get("verdict") if isinstance(raw.get("verdict"), dict) else None
        evidence_payload = _validate_evidence(evidence or raw.get("evidence") or [])
        if raw_verdict is not None:
            if mermaid_text is None:
                raise ValueError(
                    "llm_review_order_error: submit_mermaid required before submit_verdict"
                )
            await self.submit_llm_mermaid(
                submission_id=submission_id,
                mermaid=mermaid_text,
                payload=cast(dict[str, Any], raw_mermaid or {"mermaid": mermaid_text}),
            )
            await self.submit_llm_verdict(
                submission_id=submission_id,
                approved=approved,
                reason=reason,
                violations=violations,
                confidence=confidence,
                raw=cast(dict[str, Any], raw_verdict),
                evidence=evidence_payload or _validate_evidence(raw_verdict.get("evidence") or []),
                mermaid=mermaid_text,
                held=held,
            )
            return
        final_state = "quarantined" if held else "accepted" if approved else "rejected"
        evidence_payload = _validate_evidence(evidence_payload or raw.get("evidence") or [])
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO llm_reviews("
                "submission_id, approved, reason, violations, confidence, raw, mermaid, evidence, "
                "final_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    int(approved),
                    reason,
                    dumps(violations),
                    confidence,
                    dumps(raw),
                    mermaid_text,
                    dumps(evidence_payload),
                    final_state,
                    now_iso(),
                    now_iso(),
                ),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=final_state,
                actor="system",
                tool_name="deterministic_review",
                payload={
                    "approved": approved,
                    "violations": violations,
                    "evidence": evidence_payload,
                },
                reason=reason,
                idempotency_key=f"deterministic:{final_state}",
            )
            if final_state == "quarantined":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.HELD.value, reason, now_iso(), submission_id),
                )

    async def record_llm_review_event(
        self,
        *,
        submission_id: str,
        state: str,
        actor: str,
        tool_name: str,
        payload: dict[str, Any] | None = None,
        reason: str = "",
        idempotency_key: str | None = None,
    ) -> None:
        async with self.database.connect() as conn:
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=state,
                actor=actor,
                tool_name=tool_name,
                payload=payload or {},
                reason=reason,
                idempotency_key=idempotency_key,
            )

    async def submit_llm_mermaid(
        self,
        *,
        submission_id: str,
        mermaid: str,
        actor: str = "llm",
        tool_name: str = "SubmitMermaid",
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        if not mermaid.strip():
            raise ValueError("llm_review_mermaid_empty")
        event_payload = payload or {"mermaid": mermaid}
        stable_key = idempotency_key or _stable_key("SubmitMermaid", event_payload)
        async with self.database.connect() as conn:
            verdict_rows = await conn.execute_fetchall(
                "SELECT 1 FROM llm_review_events WHERE submission_id=? AND state=? LIMIT 1",
                (submission_id, "verdict_submitted"),
            )
            if verdict_rows:
                raise ValueError("llm_review_order_error: mermaid submitted after verdict")
            rows = await conn.execute_fetchall(
                "SELECT payload FROM llm_review_events WHERE submission_id=? AND tool_name=?",
                (submission_id, "SubmitMermaid"),
            )
            for row in rows:
                existing = loads(str(row["payload"]))
                if isinstance(existing, dict) and existing.get("mermaid") == mermaid:
                    return
            if rows:
                raise ValueError("llm_review_mermaid_already_submitted")
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="mermaid_submitted",
                actor=actor,
                tool_name=tool_name,
                payload=event_payload,
                reason="LLM submitted readable Mermaid review metadata",
                idempotency_key=stable_key,
            )

    async def submit_llm_verdict(
        self,
        *,
        submission_id: str,
        approved: bool,
        reason: str,
        violations: list[str],
        confidence: float,
        raw: dict[str, Any],
        evidence: list[dict[str, Any]] | None = None,
        mermaid: str | None = None,
        held: bool = False,
        actor: str = "llm",
        tool_name: str = "SubmitVerdict",
        idempotency_key: str | None = None,
    ) -> None:
        evidence_payload = _validate_evidence(evidence or [])
        # Hard gate: honor the caller's verdict. A safety reject is TERMINAL (rejected) and is
        # NOT downgraded to a hold for lacking deterministic evidence; only an explicit held=True
        # (e.g. a fail-closed LLM error / plagiarism band) quarantines.
        final_state = "quarantined" if held else "accepted" if approved else "rejected"
        async with self.database.connect() as conn:
            mermaid_rows = await conn.execute_fetchall(
                "SELECT payload FROM llm_review_events WHERE submission_id=? AND state=? "
                "ORDER BY sequence LIMIT 1",
                (submission_id, "mermaid_submitted"),
            )
            if not mermaid_rows:
                raise ValueError(
                    "llm_review_order_error: submit_mermaid required before submit_verdict"
                )
            mermaid_payload = loads(str(list(mermaid_rows)[0]["payload"]))
            mermaid_text = mermaid or (
                str(mermaid_payload.get("mermaid")) if isinstance(mermaid_payload, dict) else None
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="verdict_submitted",
                actor=actor,
                tool_name=tool_name,
                payload={**raw, "evidence": evidence_payload},
                reason=reason,
                idempotency_key=idempotency_key or _stable_key("SubmitVerdict", raw),
            )
            now = now_iso()
            await conn.execute(
                "INSERT OR REPLACE INTO llm_reviews("
                "submission_id, approved, reason, violations, confidence, raw, mermaid, evidence, "
                "final_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    int(approved),
                    reason,
                    dumps(violations),
                    confidence,
                    dumps(raw),
                    mermaid_text,
                    dumps(evidence_payload),
                    final_state,
                    now,
                    now,
                ),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=final_state,
                actor="system",
                tool_name="llm_review_state_machine",
                payload={"approved": approved, "held": held, "evidence": evidence_payload},
                reason=reason,
                idempotency_key=f"final:{final_state}:{_stable_key('reason', {'reason': reason})}",
            )
            if final_state == "rejected":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, reason, now, submission_id),
                )
            elif final_state == "quarantined":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.HELD.value, reason, now, submission_id),
                )

    async def quarantine_submission_for_llm_review(
        self, *, submission_id: str, reason: str, payload: dict[str, Any]) -> None:
        now = now_iso()
        async with self.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.HELD.value, reason, now, submission_id),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="quarantined",
                actor="system",
                tool_name="llm_review_quarantine",
                payload=payload,
                reason=reason,
                idempotency_key="submission-status:quarantined",
            )

    async def leaderboard(self, epoch_id: int, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, s.created_at, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id "
                "WHERE s.epoch_id=? AND s.status=? "
                "ORDER BY sc.final_score DESC, s.created_at ASC, s.id ASC LIMIT ?",
                (epoch_id, SubmissionStatus.COMPLETED.value, limit),
            )
        ranked = rank_leaderboard(
            LeaderboardRow(
                submission_id=str(row["id"]),
                hotkey=str(row["hotkey"]),
                final_score=float(cast(SupportsFloat, row["final_score"])),
                accepted_at=str(row["created_at"]),
            )
            for row in rows
        )
        return [
            {"hotkey": entry.hotkey, "id": entry.submission_id, "final_score": entry.final_score}
            for entry in ranked
        ]

    async def score_rows(self, epoch_id: int) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id WHERE s.epoch_id=? AND s.status=?",
                (epoch_id, SubmissionStatus.COMPLETED.value),
            )
        return [dict(row) for row in rows]

    async def list_epochs(self, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, starts_at, ends_at, status FROM epochs "
                "ORDER BY starts_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def list_eval_job_health(self, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, submission_id, level, status, attempts, created_at, updated_at "
                "FROM eval_jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def submission_history(self, days: int = 90) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT date(created_at) AS day, COUNT(*) AS count "
                "FROM submissions "
                "WHERE date(created_at) >= date('now', ?) "
                "GROUP BY day ORDER BY day ASC",
                (f"-{days} days",),
            )
        return [dict(row) for row in rows]

    async def gpu_status_summary(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        async with self.database.connect() as conn:
            status_rows = await conn.execute_fetchall(
                "SELECT status, COUNT(*) AS lease_count, "
                "COALESCE(SUM(gpu_count), 0) AS gpu_total "
                "FROM gpu_leases GROUP BY status",
            )
            tier_rows = await conn.execute_fetchall(
                "SELECT tier, COUNT(*) AS lease_count FROM gpu_leases GROUP BY tier",
            )
        return [dict(row) for row in status_rows], [dict(row) for row in tier_rows]

    async def store_runtime_config(
        self,
        *,
        config_key: str,
        value: dict[str, Any] | list[Any] | str | int | float | bool | None,
        updated_by: str,
        schema_version: int = 1,
        effective_from: str | None = None,
        enabled: bool = True,
    ) -> None:
        updated_at = now_iso()
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT INTO runtime_config("
                "config_key, value_json, schema_version, updated_by, updated_at, "
                "effective_from, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    config_key,
                    dumps(value),
                    schema_version,
                    updated_by,
                    updated_at,
                    effective_from or updated_at,
                    int(enabled),
                ),
            )

    async def active_runtime_config_rows(self, *, at: str | None = None) -> list[dict[str, Any]]:
        effective_at = at or now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT rc.* FROM runtime_config rc "
                "JOIN ("
                "SELECT config_key, "
                "MAX(effective_from || '|' || updated_at || '|' || id) AS marker "
                "FROM runtime_config WHERE enabled=1 AND effective_from <= ? GROUP BY config_key"
                ") active ON active.config_key=rc.config_key "
                "AND active.marker=(rc.effective_from || '|' || rc.updated_at || '|' || rc.id) "
                "ORDER BY rc.config_key",
                (effective_at,),
            )
        return [dict(row) for row in rows]

    async def runtime_config(
        self,
        settings: Any,
        *,
        official: bool = True,
    ) -> RuntimePolicy:
        rows = await self.active_runtime_config_rows()
        return resolve_runtime_policy(settings, rows, allow_sql_fallback=not official)

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

    async def expire_stale_held(self) -> list[str]:
        # The only remaining HELD source is the LLM suspicion-without-evidence quarantine
        # (no resolve surface in v2), so every stale held row is reaped to the terminal
        # rejected state. Cutoff/compare mirror requeue_orphaned_running.
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=self.held_review_timeout_seconds)
        ).isoformat()
        now = now_iso()
        reason = "review hold expired without resolution"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM submissions WHERE status=? AND updated_at < ?",
                (SubmissionStatus.HELD.value, cutoff),
            )
            expired = [str(row["id"]) for row in rows]
            if expired:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? "
                    "WHERE status=? AND updated_at < ?",
                    (
                        SubmissionStatus.REJECTED.value,
                        reason,
                        now,
                        SubmissionStatus.HELD.value,
                        cutoff,
                    ),
                )
        return expired

    async def requeue_orphaned_running(self) -> list[str]:
        # claimed_at and cutoff share one tz-aware ISO formatter, so lexicographic
        # `<` is a valid time test (mirrors expire_stale_assignments' deadline_at).
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=self.worker_claim_timeout_seconds)
        ).isoformat()
        now = now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM submissions "
                "WHERE status=? AND claimed_at IS NOT NULL AND claimed_at < ?",
                (SubmissionStatus.RUNNING.value, cutoff),
            )
            requeued = [str(row["id"]) for row in rows]
            if requeued:
                await conn.execute(
                    "UPDATE submissions SET status=?, claimed_at=NULL, updated_at=? "
                    "WHERE status=? AND claimed_at IS NOT NULL AND claimed_at < ?",
                    (
                        SubmissionStatus.PENDING.value,
                        now,
                        SubmissionStatus.RUNNING.value,
                        cutoff,
                    ),
                )
        return requeued

    async def claim_next(self) -> dict[str, object] | None:
        await self.requeue_orphaned_running()
        await self.expire_stale_held()
        claimed_at = now_iso()
        async with self.database.connect() as conn:
            row: aiosqlite.Row | None = None
            while True:
                # Atomic compare-and-swap: the AND status='pending' guard + RETURNING
                # close the read-then-write gap that let two callers double-claim.
                rows = await conn.execute_fetchall(
                    "UPDATE submissions SET status=?, updated_at=?, claimed_at=? "
                    "WHERE id=("
                    "SELECT id FROM submissions WHERE status=? ORDER BY created_at LIMIT 1"
                    ") AND status=? "
                    "RETURNING *",
                    (
                        SubmissionStatus.RUNNING.value,
                        claimed_at,
                        claimed_at,
                        SubmissionStatus.PENDING.value,
                        SubmissionStatus.PENDING.value,
                    ),
                )
                row_list = list(rows)
                if row_list:
                    row = row_list[0]
                    break
                # 0 rows: lost the race for that row -> retry the NEXT pending row;
                # only return None when no pending row remains.
                pending = await conn.execute_fetchall(
                    "SELECT 1 FROM submissions WHERE status=? LIMIT 1",
                    (SubmissionStatus.PENDING.value,),
                )
                if not list(pending):
                    return None
        data = dict(cast(Any, row))
        metadata = loads(str(data.get("metadata", "{}")))
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return data

    async def container_job_attempt_count(self, submission_id: str, level: str) -> int:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT COUNT(*) AS count FROM eval_jobs WHERE submission_id=? AND level=?",
                (submission_id, level),
            )
        return int(list(rows)[0]["count"])

    async def latest_retryable_container_job(
        self, submission_id: str, level: str
    ) -> dict[str, object] | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM eval_jobs WHERE submission_id=? AND level=? "
                "AND status='infra_failed' AND infra_retryable=1 "
                "AND artifact_output_path IS NOT NULL AND run_manifest_path IS NOT NULL "
                "ORDER BY attempts DESC, created_at DESC LIMIT 1",
                (submission_id, level),
            )
        return dict(list(rows)[0]) if rows else None

    async def _record_llm_review_event(
        self,
        conn: aiosqlite.Connection,
        *,
        submission_id: str,
        state: str,
        actor: str,
        tool_name: str,
        payload: dict[str, Any],
        reason: str,
        idempotency_key: str | None,
    ) -> None:
        stable_key = idempotency_key or _stable_key(tool_name, payload)
        sequence_rows = await conn.execute_fetchall(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM llm_review_events "
            "WHERE submission_id=?",
            (submission_id,),
        )
        sequence = int(list(sequence_rows)[0]["sequence"]) + 1
        await conn.execute(
            "INSERT OR IGNORE INTO llm_review_events("
            "id, submission_id, sequence, state, actor, tool_name, idempotency_key, payload, "
            "reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                submission_id,
                sequence,
                state,
                actor,
                tool_name,
                stable_key,
                dumps(payload),
                reason,
                now_iso(),
            ),
        )



def _validate_evidence(items: Any) -> list[dict[str, Any]]:
    if not items:
        return []
    if not isinstance(items, list):
        items = [items]
    return [DeterministicEvidence.model_validate(item).model_dump(mode="json") for item in items]


def _stable_key(tool_name: str, payload: dict[str, Any]) -> str:
    payload_hash = sha256(dumps(payload).encode("utf-8")).hexdigest()
    return f"{tool_name}:{payload_hash}"
