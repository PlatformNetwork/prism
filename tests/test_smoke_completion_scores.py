from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings

TERMINAL_STATES = {"completed", "failed", "rejected", "held"}


def _settings(tmp_path: Path) -> PrismSettings:
    db_path = tmp_path / "smoke-scores.sqlite3"
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        validator_hotkeys=("val-a", "val-b"),
        plagiarism_enabled=False,
        fineweb_sample_count=4,
    )


def _submit_smoke(client: TestClient, *, nonce: str) -> str:
    payload = {
        "code": VALID_CODE,
        "filename": "model.py",
        "metadata": {"execution_mode": "local_cpu_smoke"},
    }
    body = json.dumps(payload).encode()
    headers = {
        **signed_headers("secret", body, hotkey="miner-1", nonce=nonce),
        "Content-Type": "application/json",
    }
    resp = client.post("/v1/submissions", content=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return str(resp.json()["id"])


def _drive_to_terminal(client: TestClient, submission_id: str) -> str:
    for _ in range(25):
        resp = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200, resp.text
        status = client.get(f"/v1/submissions/{submission_id}").json()["status"]
        if status in TERMINAL_STATES:
            return str(status)
    raise AssertionError("submission never reached a terminal state")


def _scores_row_count(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM scores WHERE submission_id=?", (submission_id,)
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def test_local_cpu_smoke_completion_writes_scores_and_appears_on_leaderboard(tmp_path):
    """A completed local_cpu_smoke submission MUST also have a scores row and be
    visible on the public leaderboard (no completed-without-scores leak)."""
    db_path = tmp_path / "smoke-scores.sqlite3"
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        submission_id = _submit_smoke(client, nonce="smoke-n1")
        status = _drive_to_terminal(client, submission_id)

        # Invariant 1: the submission completed.
        assert status == "completed", f"expected completed, got {status}"

        # Invariant 2: a scores row exists for the completed submission.
        assert _scores_row_count(db_path, submission_id) == 1, (
            "completed submission has NO scores row -> completed-without-scores leak"
        )

        # Invariant 3: it is visible on the public leaderboard (INNER JOIN scores).
        board = client.get("/v1/leaderboard")
        assert board.status_code == 200, board.text
        entries = board.json()["entries"]
        assert any(e["submission_id"] == submission_id for e in entries), (
            "completed+scored submission missing from leaderboard"
        )
