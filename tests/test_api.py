from __future__ import annotations

import base64
import io
import json
import zipfile

from conftest import VALID_CODE, signed_headers

from prism_challenge.sdk.executors.docker import DockerRunResult


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


def test_submit_status_process_and_leaderboard(client, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"q_arch":0.8,"q_recipe":0.7}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
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


def test_internal_bridge_accepts_raw_zip_submission(client):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("model.py", VALID_CODE)
    raw = stream.getvalue()

    response = client.post(
        "/internal/v1/bridge/submissions",
        content=raw,
        headers={
            "Authorization": "Bearer secret",
            "X-Platform-Verified-Hotkey": "hk-bridge",
            "X-Submission-Filename": "project.zip",
            "Content-Type": "application/zip",
        },
    )

    assert response.status_code == 200, response.text
    submission_id = response.json()["id"]
    status = client.get(f"/v1/submissions/{submission_id}").json()
    assert status["hotkey"] == "hk-bridge"
    stored = client.app.state.repository

    async def read_code():
        async with stored.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT code, filename FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    import anyio

    row = anyio.run(read_code)
    assert row["filename"] == "project.zip"
    assert base64.b64decode(row["code"]) == raw
