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
    "code TEXT NOT NULL, code_hash TEXT NOT NULL, arch_hash TEXT, name TEXT, "
    "metadata TEXT NOT NULL,"
    "status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
    "claimed_at TEXT);"
    "CREATE INDEX IF NOT EXISTS idx_submissions_epoch ON submissions(epoch_id, status);"
    "CREATE TABLE IF NOT EXISTS eval_jobs ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, level TEXT NOT NULL, status TEXT NOT NULL,"
    "attempts INTEGER NOT NULL DEFAULT 0, external_job_id TEXT, metrics TEXT NOT NULL,"
    "error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
    "gpu_lease_id TEXT, target_id TEXT, target_server TEXT,"
    "gpu_device_ids TEXT NOT NULL DEFAULT '[]',"
    "requested_gpu_count INTEGER NOT NULL DEFAULT 0, actual_gpu_count INTEGER NOT NULL DEFAULT 0,"
    "gpu_mode TEXT NOT NULL DEFAULT '', gpu_tier TEXT NOT NULL DEFAULT '',"
    "artifact_output_path TEXT, run_manifest_path TEXT,"
    "started_at TEXT, ended_at TEXT,"
    "infra_retryable INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE IF NOT EXISTS gpu_leases ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, job_id TEXT, target_id TEXT,"
    "target_server TEXT, device_ids TEXT NOT NULL DEFAULT '[]',"
    "gpu_count INTEGER NOT NULL DEFAULT 0,"
    "min_gpu_count INTEGER NOT NULL, max_gpu_count INTEGER NOT NULL,"
    "requested_gpu_count INTEGER NOT NULL, mode TEXT NOT NULL, tier TEXT NOT NULL,"
    "score_eligible INTEGER NOT NULL, autosplit_allowed INTEGER NOT NULL DEFAULT 0,"
    "official_fixed_profile INTEGER NOT NULL DEFAULT 1,"
    "status TEXT NOT NULL, created_at TEXT NOT NULL,"
    "updated_at TEXT NOT NULL, released_at TEXT, reason TEXT NOT NULL DEFAULT '',"
    "CHECK(min_gpu_count >= 1), CHECK(max_gpu_count <= 8),"
    "CHECK(gpu_count = 0 OR (gpu_count >= 1 AND gpu_count <= max_gpu_count)));"
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_leases_one_active_submission "
    "ON gpu_leases(submission_id) WHERE status='active';"
    "CREATE INDEX IF NOT EXISTS idx_gpu_leases_fifo ON gpu_leases(status, created_at, id);"
    "CREATE TABLE IF NOT EXISTS evaluation_assignments ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, validator_hotkey TEXT NOT NULL,"
    "status TEXT NOT NULL, attempt INTEGER NOT NULL, deadline_at TEXT NOT NULL,"
    "arch_hash TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}', error TEXT,"
    "checkpoint_ref TEXT,"
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
    "mermaid TEXT, evidence TEXT NOT NULL DEFAULT '[]', final_state TEXT NOT NULL DEFAULT '',"
    "created_at TEXT NOT NULL, updated_at TEXT);"
    "CREATE TABLE IF NOT EXISTS llm_review_events ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, sequence INTEGER NOT NULL,"
    "state TEXT NOT NULL, actor TEXT NOT NULL, tool_name TEXT NOT NULL,"
    "idempotency_key TEXT NOT NULL, payload TEXT NOT NULL, reason TEXT NOT NULL,"
    "created_at TEXT NOT NULL, UNIQUE(submission_id, idempotency_key));"
    "CREATE INDEX IF NOT EXISTS idx_llm_review_events_submission "
    "ON llm_review_events(submission_id, sequence);"
    "CREATE TABLE IF NOT EXISTS nonces ("
    "hotkey TEXT NOT NULL, nonce TEXT NOT NULL, created_at TEXT NOT NULL,"
    "PRIMARY KEY (hotkey, nonce));"
    "CREATE TABLE IF NOT EXISTS architecture_families ("
    "id TEXT PRIMARY KEY, family_hash TEXT NOT NULL UNIQUE, arch_fingerprint TEXT NOT NULL,"
    "behavior_fingerprint TEXT NOT NULL, owner_hotkey TEXT NOT NULL,"
    "owner_submission_id TEXT NOT NULL, canonical_submission_id TEXT NOT NULL,"
    "q_arch_best REAL NOT NULL, display_name TEXT, "
    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
    "canonical_graph_hash TEXT NOT NULL DEFAULT '', canonical_graph_path TEXT,"
    "canonical_metadata_path TEXT, canonical_mermaid_path TEXT, canonical_version_id TEXT);"
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
    "CREATE TABLE IF NOT EXISTS architecture_versions ("
    "id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL, submission_id TEXT NOT NULL,"
    "submitter_hotkey TEXT NOT NULL, owner_hotkey TEXT NOT NULL, version_index INTEGER NOT NULL,"
    "family_hash TEXT NOT NULL, arch_fingerprint TEXT NOT NULL, behavior_fingerprint TEXT NOT NULL,"
    "canonical_graph_hash TEXT NOT NULL, architecture_source_hash TEXT,"
    "canonical_graph_path TEXT NOT NULL, canonical_metadata_path TEXT NOT NULL,"
    "derived_mermaid_path TEXT, q_arch REAL NOT NULL, is_canonical INTEGER NOT NULL DEFAULT 0,"
    "is_owner_version INTEGER NOT NULL DEFAULT 0, official_evaluation_config_id TEXT NOT NULL,"
    "official_run_manifest_path TEXT, created_at TEXT NOT NULL,"
    "UNIQUE(architecture_id, submission_id));"
    "CREATE INDEX IF NOT EXISTS idx_architecture_versions_arch "
    "ON architecture_versions(architecture_id, version_index);"
    "CREATE INDEX IF NOT EXISTS idx_architecture_versions_graph "
    "ON architecture_versions(canonical_graph_hash);"
    "CREATE TABLE IF NOT EXISTS training_script_versions ("
    "id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL, training_variant_id TEXT,"
    "submission_id TEXT NOT NULL, submitter_hotkey TEXT NOT NULL, owner_hotkey TEXT NOT NULL,"
    "version_index INTEGER NOT NULL, training_hash TEXT NOT NULL,"
    "training_graph_hash TEXT NOT NULL,"
    "training_metadata_path TEXT NOT NULL, q_recipe REAL NOT NULL, metric_mean REAL NOT NULL,"
    "metric_std REAL NOT NULL, is_current_best INTEGER NOT NULL DEFAULT 0,"
    "official_evaluation_config_id TEXT NOT NULL, official_run_manifest_path TEXT,"
    "created_at TEXT NOT NULL, UNIQUE(architecture_id, submission_id));"
    "CREATE INDEX IF NOT EXISTS idx_training_script_versions_arch "
    "ON training_script_versions(architecture_id, is_current_best);"
    "CREATE INDEX IF NOT EXISTS idx_training_script_versions_variant "
    "ON training_script_versions(training_variant_id);"
    "CREATE TABLE IF NOT EXISTS official_evaluated_tuples ("
    "submission_id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL,"
    "architecture_version_id TEXT NOT NULL, training_variant_id TEXT,"
    "training_script_version_id TEXT, evaluation_config_id TEXT NOT NULL,"
    "architecture_graph_hash TEXT NOT NULL, training_hash TEXT NOT NULL,"
    "q_arch REAL NOT NULL, q_recipe REAL NOT NULL, metric_mean REAL NOT NULL,"
    "metric_std REAL NOT NULL, metrics TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS component_scores ("
    "submission_id TEXT PRIMARY KEY, architecture_id TEXT NOT NULL,"
    "training_variant_id TEXT, project_kind TEXT NOT NULL, arch_points REAL NOT NULL,"
    "training_points REAL NOT NULL, accepted_architecture INTEGER NOT NULL,"
    "accepted_training INTEGER NOT NULL, metrics TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS component_signatures ("
    "submission_id TEXT PRIMARY KEY, architecture_id TEXT, training_variant_id TEXT,"
    "project_kind TEXT NOT NULL, family_hash TEXT NOT NULL, arch_fingerprint TEXT NOT NULL,"
    "behavior_fingerprint TEXT NOT NULL, training_hash TEXT NOT NULL, hook_metadata TEXT NOT NULL,"
    "architecture_graph TEXT NOT NULL, architecture_graph_hash TEXT NOT NULL DEFAULT '',"
    "architecture_graph_path TEXT, architecture_metadata TEXT NOT NULL DEFAULT '{}',"
    "architecture_metadata_path TEXT, training_graph TEXT NOT NULL, mermaid TEXT NOT NULL,"
    "derived_mermaid_path TEXT, architecture_summary TEXT NOT NULL,"
    "training_summary TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_component_signatures_family "
    "ON component_signatures(family_hash);"
    "CREATE TABLE IF NOT EXISTS component_agent_reviews ("
    "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, scope TEXT NOT NULL,"
    "decision TEXT NOT NULL, confidence REAL NOT NULL, matched_architecture_id TEXT,"
    "matched_training_variant_id TEXT, reason TEXT NOT NULL, raw TEXT NOT NULL,"
    "created_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_component_agent_reviews_submission "
    "ON component_agent_reviews(submission_id);"
    "CREATE TABLE IF NOT EXISTS submission_curves ("
    "submission_id TEXT PRIMARY KEY, online_loss TEXT NOT NULL,"
    "covered_bytes_cumulative TEXT NOT NULL, step0_loss REAL, baseline_nats REAL,"
    "compute TEXT NOT NULL, created_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS architecture_reports ("
    "architecture_id TEXT PRIMARY KEY, content TEXT, model TEXT,"
    "source_submission_id TEXT, generated_at TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS runtime_config ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, config_key TEXT NOT NULL, value_json TEXT NOT NULL,"
    "schema_version INTEGER NOT NULL, updated_by TEXT NOT NULL, updated_at TEXT NOT NULL,"
    "effective_from TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1);"
    "CREATE INDEX IF NOT EXISTS idx_runtime_config_active "
    "ON runtime_config(config_key, enabled, effective_from, updated_at);"
)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as conn:
            await conn.executescript(SCHEMA)
            await _run_migrations(conn)
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


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    await _ensure_columns(
        conn,
        "submissions",
        {"claimed_at": "TEXT", "name": "TEXT"},
    )
    await _ensure_columns(
        conn,
        "evaluation_assignments",
        {"checkpoint_ref": "TEXT"},
    )
    await _ensure_columns(
        conn,
        "architecture_families",
        {
            "display_name": "TEXT",
            "canonical_graph_hash": "TEXT NOT NULL DEFAULT ''",
            "canonical_graph_path": "TEXT",
            "canonical_metadata_path": "TEXT",
            "canonical_mermaid_path": "TEXT",
            "canonical_version_id": "TEXT",
        },
    )
    await _ensure_columns(
        conn,
        "training_variants",
        {
            "current_best_version_id": "TEXT",
            "training_graph_hash": "TEXT NOT NULL DEFAULT ''",
            "training_metadata_path": "TEXT",
            "official_run_manifest_path": "TEXT",
        },
    )

    await _ensure_columns(
        conn,
        "component_signatures",
        {
            "architecture_graph_hash": "TEXT NOT NULL DEFAULT ''",
            "architecture_graph_path": "TEXT",
            "architecture_metadata": "TEXT NOT NULL DEFAULT '{}'",
            "architecture_metadata_path": "TEXT",
            "derived_mermaid_path": "TEXT",
        },
    )

    await _ensure_columns(
        conn,
        "llm_reviews",
        {
            "mermaid": "TEXT",
            "evidence": "TEXT NOT NULL DEFAULT '[]'",
            "final_state": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT",
        },
    )

    await _ensure_columns(
        conn,
        "eval_jobs",
        {
            "gpu_lease_id": "TEXT",
            "target_id": "TEXT",
            "target_server": "TEXT",
            "gpu_device_ids": "TEXT NOT NULL DEFAULT '[]'",
            "requested_gpu_count": "INTEGER NOT NULL DEFAULT 0",
            "actual_gpu_count": "INTEGER NOT NULL DEFAULT 0",
            "gpu_mode": "TEXT NOT NULL DEFAULT ''",
            "gpu_tier": "TEXT NOT NULL DEFAULT ''",
            "artifact_output_path": "TEXT",
            "run_manifest_path": "TEXT",
            "started_at": "TEXT",
            "ended_at": "TEXT",
            "infra_retryable": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    await conn.executescript(
        "CREATE TABLE IF NOT EXISTS submission_curves ("
        "submission_id TEXT PRIMARY KEY, online_loss TEXT NOT NULL,"
        "covered_bytes_cumulative TEXT NOT NULL, step0_loss REAL, baseline_nats REAL,"
        "compute TEXT NOT NULL, created_at TEXT NOT NULL);"
    )
    await conn.executescript(
        "CREATE TABLE IF NOT EXISTS architecture_reports ("
        "architecture_id TEXT PRIMARY KEY, content TEXT, model TEXT,"
        "source_submission_id TEXT, generated_at TEXT NOT NULL);"
    )
    await conn.executescript(
        "CREATE TABLE IF NOT EXISTS gpu_leases ("
        "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, job_id TEXT, target_id TEXT,"
        "target_server TEXT, device_ids TEXT NOT NULL DEFAULT '[]',"
        "gpu_count INTEGER NOT NULL DEFAULT 0,"
        "min_gpu_count INTEGER NOT NULL, max_gpu_count INTEGER NOT NULL,"
        "requested_gpu_count INTEGER NOT NULL, mode TEXT NOT NULL, tier TEXT NOT NULL,"
        "score_eligible INTEGER NOT NULL, autosplit_allowed INTEGER NOT NULL DEFAULT 0,"
        "official_fixed_profile INTEGER NOT NULL DEFAULT 1,"
        "status TEXT NOT NULL, created_at TEXT NOT NULL,"
        "updated_at TEXT NOT NULL, released_at TEXT, reason TEXT NOT NULL DEFAULT '',"
        "CHECK(min_gpu_count >= 1), CHECK(max_gpu_count <= 8),"
        "CHECK(gpu_count = 0 OR (gpu_count >= 1 AND gpu_count <= max_gpu_count)));"
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_leases_one_active_submission "
        "ON gpu_leases(submission_id) WHERE status='active';"
        "CREATE INDEX IF NOT EXISTS idx_gpu_leases_fifo ON gpu_leases(status, created_at, id);"
    )
    await _ensure_columns(
        conn,
        "gpu_leases",
        {
            "autosplit_allowed": "INTEGER NOT NULL DEFAULT 0",
            "official_fixed_profile": "INTEGER NOT NULL DEFAULT 1",
        },
    )


async def _ensure_columns(conn: aiosqlite.Connection, table: str, columns: dict[str, str]) -> None:
    existing_rows = await conn.execute_fetchall(f"PRAGMA table_info({table})")
    existing = {str(row[1]) for row in existing_rows}
    for column, definition in columns.items():
        if column not in existing:
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
