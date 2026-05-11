from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, SupportsFloat, cast
from uuid import uuid4

import aiosqlite

from .db import Database, dumps, loads
from .evaluator.component_agents import ComponentOwnershipDecision
from .evaluator.component_signatures import ComponentSemanticSignature
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
                "WHERE s.epoch_id=? AND s.status=? ORDER BY sc.final_score DESC LIMIT ?",
                (epoch_id, SubmissionStatus.COMPLETED.value, limit),
            )
        return [dict(row) for row in rows]

    async def score_rows(self, epoch_id: int) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id WHERE s.epoch_id=? AND s.status=?",
                (epoch_id, SubmissionStatus.COMPLETED.value),
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

    async def component_candidates(
        self,
        *,
        family_hash: str,
        requested_architecture_id: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        async with self.database.connect() as conn:
            arch_rows = await conn.execute_fetchall(
                "SELECT a.*, cs.architecture_graph, cs.architecture_summary, cs.mermaid "
                "FROM architecture_families a "
                "LEFT JOIN component_signatures cs ON cs.submission_id=a.canonical_submission_id "
                "ORDER BY CASE WHEN a.family_hash=? THEN 0 WHEN a.id=? THEN 1 ELSE 2 END, "
                "a.updated_at DESC LIMIT ?",
                (family_hash, requested_architecture_id or "", limit),
            )
            architectures = [_decode_candidate_graph(dict(row)) for row in arch_rows]
            architecture_ids = [str(row["id"]) for row in architectures]
            training_rows: list[aiosqlite.Row] = []
            if architecture_ids:
                placeholders = ",".join("?" for _ in architecture_ids)
                training_rows = list(
                    await conn.execute_fetchall(
                        "SELECT tv.*, cs.training_graph, cs.training_summary "
                        "FROM training_variants tv "
                        "LEFT JOIN component_signatures cs ON cs.submission_id=tv.submission_id "
                        f"WHERE tv.architecture_id IN ({placeholders}) "
                        "ORDER BY tv.is_current_best DESC, tv.metric_mean DESC LIMIT ?",
                        (*architecture_ids, limit),
                    )
                )
        return architectures, [_decode_candidate_graph(dict(row)) for row in training_rows]

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
        architecture_transfer_delta_abs: float,
        architecture_transfer_delta_rel: float,
        training_transfer_delta_abs: float,
        training_transfer_delta_rel: float,
        transfer_confidence: float,
        metrics: dict[str, float],
        semantic_signature: ComponentSemanticSignature,
        ownership_decision: ComponentOwnershipDecision,
    ) -> dict[str, object]:
        now = now_iso()
        payload = {
            "submission_id": submission_id,
            "project_kind": project_kind,
            "family_hash": family_hash,
            "arch_fingerprint": arch_fingerprint,
            "behavior_fingerprint": behavior_fingerprint,
            "training_hash": training_hash,
            "requested_architecture_id": requested_architecture_id,
            "q_arch": q_arch,
            "q_recipe": q_recipe,
            "metric_mean": metric_mean,
            "metric_std": metric_std,
            "architecture_weight": architecture_weight,
            "training_weight": training_weight,
            "architecture_delta_abs": architecture_delta_abs,
            "architecture_delta_rel": architecture_delta_rel,
            "training_delta_abs": training_delta_abs,
            "training_delta_rel": training_delta_rel,
            "training_z_score": training_z_score,
            "architecture_transfer_delta_abs": architecture_transfer_delta_abs,
            "architecture_transfer_delta_rel": architecture_transfer_delta_rel,
            "training_transfer_delta_abs": training_transfer_delta_abs,
            "training_transfer_delta_rel": training_transfer_delta_rel,
            "transfer_confidence": transfer_confidence,
            "metrics": metrics,
            "signature": semantic_signature.to_payload(),
        }
        async with self.database.connect() as conn:
            hotkey = await self._submission_hotkey(conn, submission_id)
            await self._store_component_agent_reviews(conn, submission_id, ownership_decision, now)
            if ownership_decision.rejected:
                await self._record_ownership_event(
                    conn,
                    submission_id=submission_id,
                    event="rejected",
                    scope="component",
                    reason=ownership_decision.reason,
                    now=now,
                )
                return {"held": False, "rejected": True}
            if ownership_decision.held:
                hold_id = await self._create_component_hold(
                    conn,
                    submission_id=submission_id,
                    reason=ownership_decision.reason,
                    confidence=min(
                        ownership_decision.architecture_confidence,
                        ownership_decision.training_confidence,
                    ),
                    payload=payload,
                    now=now,
                )
                await self._store_component_signature(
                    conn,
                    submission_id=submission_id,
                    signature=semantic_signature,
                    architecture_id=ownership_decision.matched_architecture_id,
                    training_variant_id=ownership_decision.matched_training_variant_id,
                    now=now,
                )
                return {"held": True, "hold_id": hold_id, "rejected": False}
            result = await self._apply_component_result(
                conn=conn,
                hotkey=hotkey,
                payload=payload,
                decision=ownership_decision,
                now=now,
            )
            await self._store_component_signature(
                conn,
                submission_id=submission_id,
                signature=semantic_signature,
                architecture_id=str(result["architecture_id"]),
                training_variant_id=cast(str | None, result.get("training_variant_id")),
                now=now,
            )
        return result

    async def resolve_component_hold(
        self,
        *,
        hold_id: str,
        architecture_action: str,
        training_action: str,
        architecture_id: str | None,
        training_variant_id: str | None,
        reason: str,
    ) -> dict[str, object]:
        now = now_iso()
        async with self.database.connect() as conn:
            hold_rows = await conn.execute_fetchall(
                "SELECT * FROM component_review_holds WHERE id=? AND status='pending'",
                (hold_id,),
            )
            holds = list(hold_rows)
            if not holds:
                raise ValueError("component review hold not found")
            hold = dict(holds[0])
            payload = loads(str(hold["payload"]))
            if not isinstance(payload, dict):
                raise ValueError("invalid component hold payload")
            submission_id = str(payload["submission_id"])
            hotkey = await self._submission_hotkey(conn, submission_id)
            decision = ComponentOwnershipDecision(
                architecture_action=architecture_action,
                architecture_confidence=1.0,
                training_action=training_action,
                training_confidence=1.0,
                matched_architecture_id=architecture_id,
                matched_training_variant_id=training_variant_id,
                reason=reason,
                raw={"manual_resolution": True, "reason": reason},
            )
            if architecture_action == "reject" or training_action == "reject":
                await conn.execute(
                    "UPDATE component_review_holds SET status=?, resolution=?, updated_at=? "
                    "WHERE id=?",
                    ("rejected", dumps(decision.raw), now, hold_id),
                )
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, reason, now, submission_id),
                )
                return {"rejected": True, "held": False}
            result = await self._apply_component_result(
                conn=conn,
                hotkey=hotkey,
                payload=payload,
                decision=decision,
                now=now,
            )
            await conn.execute(
                "UPDATE component_review_holds SET status=?, resolution=?, updated_at=? WHERE id=?",
                ("resolved", dumps(decision.raw), now, hold_id),
            )
            await conn.execute(
                "UPDATE submissions SET status=?, error=NULL, updated_at=? WHERE id=?",
                (SubmissionStatus.COMPLETED.value, now, submission_id),
            )
            await self._store_component_signature(
                conn,
                submission_id=submission_id,
                signature=ComponentSemanticSignature(**cast(dict[str, Any], payload["signature"])),
                architecture_id=str(result["architecture_id"]),
                training_variant_id=cast(str | None, result.get("training_variant_id")),
                now=now,
            )
            return result

    async def list_component_review_holds(self, limit: int = 100) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, submission_id, status, reason, confidence, created_at, updated_at "
                "FROM component_review_holds ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def _apply_component_result(
        self,
        *,
        conn: aiosqlite.Connection,
        hotkey: str,
        payload: dict[str, Any],
        decision: ComponentOwnershipDecision,
        now: str,
    ) -> dict[str, object]:
        architecture_id, accepted_architecture, arch_points = await self._apply_architecture_result(
            conn=conn,
            hotkey=hotkey,
            payload=payload,
            decision=decision,
            now=now,
        )
        training_variant_id: str | None = None
        accepted_training = False
        training_points = 0.0
        if str(payload["project_kind"]) != "architecture_only":
            (
                training_variant_id,
                accepted_training,
                training_points,
            ) = await self._apply_training_result(
                conn=conn,
                hotkey=hotkey,
                payload=payload,
                decision=decision,
                architecture_id=architecture_id,
                now=now,
            )
        await conn.execute(
            "INSERT OR REPLACE INTO component_scores("
            "submission_id, architecture_id, training_variant_id, project_kind, arch_points,"
            "training_points, accepted_architecture, accepted_training, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(payload["submission_id"]),
                architecture_id,
                training_variant_id,
                str(payload["project_kind"]),
                arch_points,
                training_points,
                int(accepted_architecture),
                int(accepted_training),
                dumps(cast(dict[str, float], payload["metrics"])),
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
            "held": False,
            "rejected": False,
        }

    async def _apply_architecture_result(
        self,
        *,
        conn: aiosqlite.Connection,
        hotkey: str,
        payload: dict[str, Any],
        decision: ComponentOwnershipDecision,
        now: str,
    ) -> tuple[str, bool, float]:
        submission_id = str(payload["submission_id"])
        action = decision.architecture_action
        family = await self._component_family(
            conn,
            requested_architecture_id=decision.matched_architecture_id
            or cast(str | None, payload.get("requested_architecture_id")),
            family_hash=str(payload["family_hash"]),
        )
        accepted = False
        arch_points = 0.0
        if family is None or action == "new":
            architecture_id = str(uuid4())
            await conn.execute(
                "INSERT INTO architecture_families("
                "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey,"
                "owner_submission_id, canonical_submission_id, q_arch_best, created_at,"
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    architecture_id,
                    str(payload["family_hash"]),
                    str(payload["arch_fingerprint"]),
                    str(payload["behavior_fingerprint"]),
                    hotkey,
                    submission_id,
                    submission_id,
                    float(payload["q_arch"]),
                    now,
                    now,
                ),
            )
            accepted = True
            arch_points = float(payload["architecture_weight"]) * float(payload["q_arch"])
            await self._record_ownership_event(
                conn,
                submission_id=submission_id,
                event="created",
                scope="architecture",
                architecture_id=architecture_id,
                to_hotkey=hotkey,
                reason=decision.reason,
                now=now,
            )
            return architecture_id, accepted, arch_points
        architecture_id = str(family["id"])
        old_q = float(cast(SupportsFloat, family["q_arch_best"]))
        q_arch = float(payload["q_arch"])
        transfer = action == "transfer" or (
            decision.architecture_confidence >= float(payload["transfer_confidence"])
            and _meaningful_improvement(
                q_arch,
                old_q,
                new_std=0.0,
                old_std=0.0,
                min_delta_abs=float(payload["architecture_transfer_delta_abs"]),
                min_delta_rel=float(payload["architecture_transfer_delta_rel"]),
                z_score=0.0,
            )
        )
        if transfer:
            await conn.execute(
                "UPDATE architecture_families SET owner_hotkey=?, owner_submission_id=?, "
                "canonical_submission_id=?, q_arch_best=?, updated_at=? WHERE id=?",
                (hotkey, submission_id, submission_id, max(q_arch, old_q), now, architecture_id),
            )
            accepted = True
            arch_points = float(payload["architecture_weight"]) * q_arch
            await self._record_ownership_event(
                conn,
                submission_id=submission_id,
                event="transferred",
                scope="architecture",
                architecture_id=architecture_id,
                from_hotkey=str(family["owner_hotkey"]),
                to_hotkey=hotkey,
                reason=decision.reason,
                now=now,
            )
        elif _meaningful_improvement(
            q_arch,
            old_q,
            new_std=0.0,
            old_std=0.0,
            min_delta_abs=float(payload["architecture_delta_abs"]),
            min_delta_rel=float(payload["architecture_delta_rel"]),
            z_score=0.0,
        ):
            await conn.execute(
                "UPDATE architecture_families SET q_arch_best=?, canonical_submission_id=?, "
                "updated_at=? WHERE id=?",
                (q_arch, submission_id, now, architecture_id),
            )
        return architecture_id, accepted, arch_points

    async def _apply_training_result(
        self,
        *,
        conn: aiosqlite.Connection,
        hotkey: str,
        payload: dict[str, Any],
        decision: ComponentOwnershipDecision,
        architecture_id: str,
        now: str,
    ) -> tuple[str | None, bool, float]:
        submission_id = str(payload["submission_id"])
        if decision.training_action == "none":
            return None, False, 0.0
        existing = None
        if decision.matched_training_variant_id:
            existing = await self._training_variant_by_id(
                conn, decision.matched_training_variant_id
            )
        if existing is None:
            existing = await self._training_variant(
                conn, architecture_id, str(payload["training_hash"])
            )
        current = await self._current_training_variant(conn, architecture_id)
        q_recipe = float(payload["q_recipe"])
        metric_mean = float(payload["metric_mean"])
        metric_std = float(payload["metric_std"])
        accepted = False
        points = 0.0
        if existing is not None and decision.training_action == "transfer":
            old_mean = float(cast(SupportsFloat, existing["metric_mean"]))
            if _meaningful_improvement(
                metric_mean,
                old_mean,
                new_std=metric_std,
                old_std=float(cast(SupportsFloat, existing["metric_std"])),
                min_delta_abs=float(payload["training_transfer_delta_abs"]),
                min_delta_rel=float(payload["training_transfer_delta_rel"]),
                z_score=0.0,
            ):
                await conn.execute(
                    "UPDATE training_variants SET owner_hotkey=?, submission_id=?, q_recipe=?, "
                    "metric_mean=?, metric_std=?, is_current_best=1, updated_at=? WHERE id=?",
                    (
                        hotkey,
                        submission_id,
                        q_recipe,
                        metric_mean,
                        max(0.0, metric_std),
                        now,
                        existing["id"],
                    ),
                )
                await conn.execute(
                    "UPDATE training_variants SET is_current_best=0 "
                    "WHERE architecture_id=? AND id != ?",
                    (architecture_id, existing["id"]),
                )
                accepted = True
                points = float(payload["training_weight"]) * q_recipe
            return str(existing["id"]), accepted, points
        if existing is not None and str(existing["training_hash"]) == str(payload["training_hash"]):
            return str(existing["id"]), False, 0.0
        accepted = current is None or _meaningful_improvement(
            metric_mean,
            float(cast(SupportsFloat, current["metric_mean"])),
            new_std=metric_std,
            old_std=float(cast(SupportsFloat, current["metric_std"])),
            min_delta_abs=float(payload["training_delta_abs"]),
            min_delta_rel=float(payload["training_delta_rel"]),
            z_score=float(payload["training_z_score"]),
        )
        training_variant_id = str(uuid4())
        if accepted:
            await conn.execute(
                "UPDATE training_variants SET is_current_best=0 WHERE architecture_id=?",
                (architecture_id,),
            )
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id,"
            "q_recipe, metric_mean, metric_std, is_current_best, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                training_variant_id,
                architecture_id,
                str(payload["training_hash"]),
                hotkey,
                submission_id,
                q_recipe,
                metric_mean,
                max(0.0, metric_std),
                int(accepted),
                now,
                now,
            ),
        )
        if accepted:
            points = float(payload["training_weight"]) * q_recipe
            await self._record_ownership_event(
                conn,
                submission_id=submission_id,
                event="created",
                scope="training",
                architecture_id=architecture_id,
                training_variant_id=training_variant_id,
                to_hotkey=hotkey,
                reason=decision.reason,
                now=now,
            )
        return training_variant_id, accepted, points

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

    async def _submission_hotkey(self, conn: aiosqlite.Connection, submission_id: str) -> str:
        rows = await conn.execute_fetchall(
            "SELECT hotkey FROM submissions WHERE id=?", (submission_id,)
        )
        row_list = list(rows)
        if not row_list:
            raise ValueError("submission not found")
        return str(row_list[0]["hotkey"])

    async def _store_component_signature(
        self,
        conn: aiosqlite.Connection,
        *,
        submission_id: str,
        signature: ComponentSemanticSignature,
        architecture_id: str | None,
        training_variant_id: str | None,
        now: str,
    ) -> None:
        await conn.execute(
            "INSERT OR REPLACE INTO component_signatures("
            "submission_id, architecture_id, training_variant_id, project_kind, family_hash,"
            "arch_fingerprint, behavior_fingerprint, training_hash, hook_metadata,"
            "architecture_graph, training_graph, mermaid, architecture_summary,"
            "training_summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission_id,
                architecture_id,
                training_variant_id,
                signature.project_kind,
                signature.family_hash,
                signature.arch_fingerprint,
                signature.behavior_fingerprint,
                signature.training_hash,
                dumps(signature.hook_metadata),
                dumps(signature.architecture_graph),
                dumps(signature.training_graph),
                signature.mermaid,
                signature.architecture_summary,
                signature.training_summary,
                now,
            ),
        )

    async def _store_component_agent_reviews(
        self,
        conn: aiosqlite.Connection,
        submission_id: str,
        decision: ComponentOwnershipDecision,
        now: str,
    ) -> None:
        rows = (
            (
                "architecture",
                decision.architecture_action,
                decision.architecture_confidence,
            ),
            ("training", decision.training_action, decision.training_confidence),
        )
        for scope, action, confidence in rows:
            await conn.execute(
                "INSERT INTO component_agent_reviews("
                "id, submission_id, scope, decision, confidence, matched_architecture_id,"
                "matched_training_variant_id, reason, raw, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    scope,
                    action,
                    confidence,
                    decision.matched_architecture_id,
                    decision.matched_training_variant_id,
                    decision.reason,
                    dumps(decision.raw),
                    now,
                ),
            )

    async def _create_component_hold(
        self,
        conn: aiosqlite.Connection,
        *,
        submission_id: str,
        reason: str,
        confidence: float,
        payload: dict[str, Any],
        now: str,
    ) -> str:
        hold_id = str(uuid4())
        await conn.execute(
            "INSERT OR REPLACE INTO component_review_holds("
            "id, submission_id, status, reason, confidence, payload, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hold_id,
                submission_id,
                "pending",
                reason,
                max(0.0, min(1.0, confidence)),
                dumps(payload),
                now,
                now,
            ),
        )
        await self._record_ownership_event(
            conn,
            submission_id=submission_id,
            event="held",
            scope="component",
            reason=reason,
            now=now,
        )
        return hold_id

    async def _record_ownership_event(
        self,
        conn: aiosqlite.Connection,
        *,
        submission_id: str,
        event: str,
        scope: str,
        reason: str,
        now: str,
        architecture_id: str | None = None,
        training_variant_id: str | None = None,
        from_hotkey: str | None = None,
        to_hotkey: str | None = None,
    ) -> None:
        await conn.execute(
            "INSERT INTO ownership_events("
            "id, submission_id, event, scope, architecture_id, training_variant_id,"
            "from_hotkey, to_hotkey, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                submission_id,
                event,
                scope,
                architecture_id,
                training_variant_id,
                from_hotkey,
                to_hotkey,
                reason,
                now,
            ),
        )

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

    async def _training_variant_by_id(
        self, conn: aiosqlite.Connection, training_variant_id: str
    ) -> aiosqlite.Row | None:
        rows = await conn.execute_fetchall(
            "SELECT * FROM training_variants WHERE id=?",
            (training_variant_id,),
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


def _decode_candidate_graph(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("architecture_graph", "training_graph"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                decoded = loads(value)
            except Exception:
                decoded = {}
            row[key] = decoded if isinstance(decoded, dict) else {}
        elif not isinstance(value, dict):
            row[key] = {}
    return row


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
