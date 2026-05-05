from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = (
    "PRAGMA journal_mode=WAL;"
    "CREATE TABLE IF NOT EXISTS miners ("
    "hotkey TEXT PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS epochs ("
    "id INTEGER PRIMARY KEY, starts_at TEXT NOT NULL, ends_at TEXT NOT NULL, status TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS submissions ("
    "id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, epoch_id INTEGER NOT NULL, filename TEXT NOT NULL,"
    "code TEXT NOT NULL, code_hash TEXT NOT NULL, arch_hash TEXT, metadata TEXT NOT NULL,"
    "status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_submissions_epoch ON submissions(epoch_id, status);"
    "CREATE TABLE IF NOT EXISTS eval_jobs ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, level TEXT NOT NULL, status TEXT NOT NULL,"
    "attempts INTEGER NOT NULL DEFAULT 0, external_job_id TEXT, metrics TEXT NOT NULL,"
    "error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS scores ("
    "submission_id TEXT PRIMARY KEY, q_arch REAL NOT NULL, q_recipe REAL NOT NULL,"
    "anti_cheat_multiplier REAL NOT NULL, diversity_bonus REAL NOT NULL,"
    "penalty REAL NOT NULL, final_score REAL NOT NULL, metrics TEXT NOT NULL,"
    "created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS cheat_findings ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, kind TEXT NOT NULL,"
    "severity REAL NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS nonces ("
    "hotkey TEXT NOT NULL, nonce TEXT NOT NULL, created_at TEXT NOT NULL,"
    "PRIMARY KEY (hotkey, nonce));"
)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as conn:
            await conn.executescript(SCHEMA)
            await conn.commit()

    async def close(self) -> None:
        return None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            await conn.commit()
        finally:
            await conn.close()


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def loads(data: str | None) -> dict[str, Any]:
    if not data:
        return {}
    value = json.loads(data)
    return value if isinstance(value, dict) else {}
