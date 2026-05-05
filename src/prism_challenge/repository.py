from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

import aiosqlite

from .db import Database, dumps, loads
from .models import SubmissionCreate, SubmissionResponse, SubmissionStatus, SubmissionStatusResponse


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
                "SELECT code FROM submissions WHERE id != ? ORDER BY created_at DESC LIMIT 100",
                (current_submission_id,),
            )
        return [str(row["code"]) for row in rows]

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
        data["metadata"] = loads(str(data.get("metadata", "{}")))
        return data
