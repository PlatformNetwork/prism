from __future__ import annotations

import json

from conftest import VALID_CODE, signed_headers


def test_health_version_and_internal_auth(client):
    assert client.get("/health").json()["slug"] == "prism"
    assert "nas" in client.get("/version").json()["capabilities"]
    assert client.get("/internal/v1/get_weights").status_code == 401
    response = client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
    )
    assert response.status_code == 200
    assert response.json()["weights"] == {}


def test_submit_status_process_and_leaderboard(client):
    payload = {"code": VALID_CODE, "filename": "model.py"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    submission_id = response.json()["id"]

    process = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert process.status_code == 200, process.text
    assert process.json()["submission_id"] == submission_id

    status = client.get(f"/v1/submissions/{submission_id}").json()
    assert status["status"] == "completed"
    assert status["final_score"] >= 0

    leaderboard = client.get("/v1/leaderboard").json()
    assert leaderboard["entries"][0]["submission_id"] == submission_id

    weights = client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
    ).json()["weights"]
    assert weights == {"hk": 1.0}


def test_rejects_bad_signature(client):
    response = client.post(
        "/v1/submissions",
        json={"code": VALID_CODE},
        headers={"X-Hotkey": "hk", "X-Signature": "bad", "X-Nonce": "x", "X-Timestamp": "1"},
    )
    assert response.status_code == 401
