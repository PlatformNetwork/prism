from __future__ import annotations

import json

from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import llm_review


def _client(tmp_path, **overrides):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'assignments.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        validator_hotkeys=("val-a", "val-b"),
        plagiarism_enabled=False,
        **overrides,
    )
    return TestClient(create_app(settings))


def _submit(client: TestClient, code: str = VALID_CODE, nonce: str = "assign") -> str:
    body = json.dumps({"code": code, "filename": "model.py"}, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _headers(validator: str = "val-a") -> dict[str, str]:
    return {"Authorization": "Bearer secret", "X-Validator-Hotkey": validator}


def test_master_assigns_validator_and_saves_result(tmp_path):
    with _client(tmp_path) as client:
        submission_id = _submit(client)
        assignment = client.post(
            "/internal/v1/validators/assignments/next", headers=_headers()
        ).json()

        assert assignment["submission_id"] == submission_id
        assert assignment["validator_hotkey"] == "val-a"
        assert assignment["code"] == VALID_CODE
        assert assignment["attempt"] == 1

        accepted = client.post(
            f"/internal/v1/validators/assignments/{assignment['id']}/accept",
            headers={"Authorization": "Bearer secret"},
        )
        assert accepted.status_code == 200
        result = client.post(
            f"/internal/v1/validators/assignments/{assignment['id']}/result",
            json={"metrics": {"q_arch": 0.8, "q_recipe": 0.7, "train_loss": 1.0}},
            headers={"Authorization": "Bearer secret"},
        )
        assert result.status_code == 200, result.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] == 0.8


def test_validator_rejection_requeues_for_replacement(tmp_path):
    with _client(tmp_path) as client:
        submission_id = _submit(client, nonce="reject-requeue")
        first = client.post("/internal/v1/validators/assignments/next", headers=_headers()).json()
        rejected = client.post(
            f"/internal/v1/validators/assignments/{first['id']}/reject",
            json={"reason": "busy"},
            headers={"Authorization": "Bearer secret"},
        )
        assert rejected.status_code == 200

        second = client.post(
            "/internal/v1/validators/assignments/next", headers=_headers("val-b")
        ).json()
        assert second["submission_id"] == submission_id
        assert second["validator_hotkey"] == "val-b"
        assert second["attempt"] == 2


def test_assignment_expiration_requeues_pending_submission(tmp_path):
    with _client(tmp_path, validator_assignment_timeout_seconds=1) as client:
        submission_id = _submit(client, nonce="expire-requeue")
        assignment = client.post(
            "/internal/v1/validators/assignments/next", headers=_headers()
        ).json()
        database = client.app.state.database

        async def expire_deadline():
            async with database.connect() as conn:
                await conn.execute(
                    "UPDATE evaluation_assignments SET deadline_at=? WHERE id=?",
                    ("2000-01-01T00:00:00+00:00", assignment["id"]),
                )

        import anyio

        anyio.run(expire_deadline)
        expired = client.post(
            "/internal/v1/validators/assignments/expire",
            headers={"Authorization": "Bearer secret"},
        ).json()
        assert expired["expired_submission_ids"] == [submission_id]
        second = client.post(
            "/internal/v1/validators/assignments/next", headers=_headers("val-b")
        ).json()
        assert second["submission_id"] == submission_id
        assert second["attempt"] == 2


def test_llm_review_rejects_before_assignment(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        return llm_review.LlmReview(False, "copied miner code", ["plagiarism"], [0, 0])

    monkeypatch.setattr(llm_review, "review_code", reject)
    with _client(tmp_path) as client:
        submission_id = _submit(client, nonce="llm-reject")
        response = client.post("/internal/v1/validators/assignments/next", headers=_headers())
        assert response.status_code == 200
        assert response.json() is None
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "rejected"
        assert status["error"] == "copied miner code"
