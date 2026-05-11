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
    "CREATE TABLE IF NOT EXISTS evaluation_assignments ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, validator_hotkey TEXT NOT NULL,"
    "status TEXT NOT NULL, attempt INTEGER NOT NULL, deadline_at TEXT NOT NULL,"
    "arch_hash TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}', error TEXT,"
    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_eval_assignments_submission "
    "ON evaluation_assignments(submission_id, attempt);"
    "CREATE INDEX IF NOT EXISTS idx_eval_assignments_validator "
    "ON evaluation_assignments(validator_hotkey, status);"
    "CREATE TABLE IF NOT EXISTS scores ("
    "submission_id TEXT PRIMARY KEY, q_arch REAL NOT NULL, q_recipe REAL NOT NULL,"
    "anti_cheat_multiplier REAL NOT NULL, diversity_bonus REAL NOT NULL,"
    "penalty REAL NOT NULL, final_score REAL NOT NULL, metrics TEXT NOT NULL,"
    "created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS cheat_findings ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, kind TEXT NOT NULL,"
    "severity REAL NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS submission_sources ("
    "submission_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, code_hash TEXT NOT NULL,"
    "files TEXT NOT NULL, ast_features TEXT NOT NULL, token_shingles TEXT NOT NULL,"
    "fingerprint TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_submission_sources_hotkey "
    "ON submission_sources(hotkey, created_at);"
    "CREATE TABLE IF NOT EXISTS plagiarism_reviews ("
    "submission_id TEXT PRIMARY KEY, candidate_submission_id TEXT, similarity REAL NOT NULL,"
    "verdict INTEGER NOT NULL, reason TEXT NOT NULL, violations TEXT NOT NULL,"
    "report TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS llm_reviews ("
    "submission_id TEXT PRIMARY KEY, approved INTEGER NOT NULL, reason TEXT NOT NULL,"
    "violations TEXT NOT NULL, confidence REAL NOT NULL, raw TEXT NOT NULL,"
    "created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS nonces ("
    "hotkey TEXT NOT NULL, nonce TEXT NOT NULL, created_at TEXT NOT NULL,"
    "PRIMARY KEY (hotkey, nonce));"
    "CREATE TABLE IF NOT EXISTS architecture_families ("
    "id TEXT PRIMARY KEY, family_hash TEXT NOT NULL UNIQUE, arch_fingerprint TEXT NOT NULL,"
    "behavior_fingerprint TEXT NOT NULL, owner_hotkey TEXT NOT NULL,"
    "owner_submission_id TEXT NOT NULL, canonical_submission_id TEXT NOT NULL,"
    "q_arch_best REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_architecture_families_owner "
    "ON architecture_families(owner_hotkey);"
    "CREATE TABLE IF NOT EXISTS training_variants ("
    "id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL, training_hash TEXT NOT NULL,"
    "owner_hotkey TEXT NOT NULL, submission_id TEXT NOT NULL, q_recipe REAL NOT NULL,"
    "metric_mean REAL NOT NULL, metric_std REAL NOT NULL,"
    "is_current_best INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,"
    "updated_at TEXT NOT NULL, UNIQUE(architecture_id, training_hash));"
    "CREATE INDEX IF NOT EXISTS idx_training_variants_arch "
    "ON training_variants(architecture_id, is_current_best);"
    "CREATE TABLE IF NOT EXISTS component_scores ("
    "submission_id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL,"
    "training_variant_id TEXT, project_kind TEXT NOT NULL, arch_points REAL NOT NULL,"
    "training_points REAL NOT NULL, accepted_architecture INTEGER NOT NULL,"
    "accepted_training INTEGER NOT NULL, metrics TEXT NOT NULL, created_at TEXT NOT NULL);"
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


def dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def loads(data: str | None) -> Any:
    if not data:
        return {}
    return json.loads(data)
