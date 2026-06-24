from __future__ import annotations

import json
from typing import Any

import anyio
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator import llm_review as llm
from prism_challenge.evaluator.llm_review import LlmReviewConfig, review_code
from prism_challenge.models import SubmissionCreate
from prism_challenge.repository import PrismRepository
from prism_challenge.sdk.executors.docker import DockerRunResult

REJECT_REASON = "training.py never steps the optimizer; dead no-op loop cannot learn the model"
ALLOW_REASON = "architecture and training loop are coherent; real from-scratch learning procedure"


def _fake_chat_class(
    verdict_args: dict[str, Any], mermaid_args: dict[str, Any] | None = None
) -> type:
    mermaid_args = mermaid_args or {
        "mermaid": "flowchart LR\n  A[architecture] --> B[training]",
        "notes": "ok",
    }
    args_by_tool = {"SubmitMermaid": mermaid_args, "SubmitVerdict": verdict_args}

    class _FakeMessage:
        def __init__(self, tool_name: str) -> None:
            self.tool_calls = [{"name": tool_name, "args": args_by_tool[tool_name]}]

    class _FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            self._tool: str | None = None

        def bind_tools(self, tools: list[Any], tool_choice: str, strict: bool) -> _FakeChat:
            self._tool = tool_choice
            return self

        def invoke(self, messages: list[tuple[str, str]]) -> _FakeMessage:
            assert self._tool is not None
            return _FakeMessage(self._tool)

    return _FakeChat


def _verdict(verdict: bool, reason: str, evidence: list[Any] | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "verdict": verdict,
        "violations": [],
        "confidence": 0.95,
        "rule_ids": [],
        "evidence": evidence or [],
    }


def _artifact_dir(spec: Any):
    for mount in spec.mounts:
        if mount.target == "/artifacts":
            return mount.source
    raise AssertionError("container spec has no /artifacts mount")


def _write_v2_manifest(spec: Any) -> None:
    manifest = {
        "schema_version": "prism_run_manifest.v2",
        "metrics": {
            "covered_bytes": 4096,
            "sum_neg_log_likelihood_nats": 2200.0,
            "online_loss": [3.1, 2.9, 2.4],
            "predicted_tokens": 800,
            "tokens_seen": 800,
        },
    }
    (_artifact_dir(spec) / "prism_run_manifest.v2.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _submit(client: TestClient, nonce: str) -> str:
    payload = {"code": two_script_bundle(), "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _hard_gate_settings(tmp_path, name: str) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / name}",
        shared_token="secret",
        allow_insecure_signatures=True,
        llm_review_enabled=True,
        openrouter_api_key="sk-or-test",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        distributed_contract_policy="off",
    )


def test_llm_reject_without_evidence_is_terminal_not_held(monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_load_chat_openai", lambda: _fake_chat_class(_verdict(False, REJECT_REASON))
    )

    review = review_code(
        "# file: architecture.py\n# file: training.py\n",
        config=LlmReviewConfig(api_key="sk-or-test"),
    )

    assert review.approved is False
    assert review.held is False
    assert "dead no-op loop" in review.reason
    assert "suspicion without evidence" not in review.reason


def test_llm_allow_approves_without_hold(monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_load_chat_openai", lambda: _fake_chat_class(_verdict(True, ALLOW_REASON))
    )

    review = review_code("ok", config=LlmReviewConfig(api_key="sk-or-test"))

    assert review.approved is True
    assert review.held is False
    assert review.reason == ALLOW_REASON


def test_safety_review_prompt_flags_indirection_coherence_both_scripts() -> None:
    from prism_challenge.evaluator.llm_review import (
        SAFETY_REVIEW_SYSTEM,
        build_safety_review_prompt,
    )

    system = SAFETY_REVIEW_SYSTEM.lower()
    prompt = build_safety_review_prompt(
        subject="Prism project",
        rules_text="- demo: rule",
        code="# file: architecture.py\n# file: training.py\n",
    ).lower()

    combined = system + "\n" + prompt
    # Inverted gating: a reject is terminal and does NOT require a 64-char evidence hash.
    assert "getattr" in combined
    assert "setattr" in combined
    # Both scripts + architecture<->training coherence.
    assert "architecture" in combined and "training" in combined
    assert "coheren" in combined
    assert "both" in combined
    # The cheat/obfuscation/security mandate.
    assert "obfusc" in combined
    assert "no-op" in combined or "dead" in combined


def test_submit_llm_verdict_reject_without_evidence_is_terminal_rejected(tmp_path) -> None:
    async def run() -> None:
        database = Database(tmp_path / "verdict-reject.sqlite3")
        await database.init()
        repo = PrismRepository(database, epoch_seconds=3600)
        created = await repo.create_submission(
            "miner", SubmissionCreate(code="def build_model(ctx):\n    return None\n")
        )
        submission_id = created.id

        await repo.submit_llm_mermaid(
            submission_id=submission_id, mermaid="flowchart LR\n  A-->B"
        )
        await repo.submit_llm_verdict(
            submission_id=submission_id,
            approved=False,
            reason="smuggled pretrained weights loaded into the model",
            violations=["prism:smuggled-weights"],
            confidence=0.95,
            raw={"reason": "smuggled pretrained weights loaded into the model", "verdict": False},
            evidence=[],
            held=False,
        )

        async with repo.database.connect() as conn:
            review = list(
                await conn.execute_fetchall(
                    "SELECT approved, reason, final_state FROM llm_reviews WHERE submission_id=?",
                    (submission_id,),
                )
            )[0]
            submission = list(
                await conn.execute_fetchall(
                    "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
                )
            )[0]
            events = await conn.execute_fetchall(
                "SELECT state FROM llm_review_events WHERE submission_id=? ORDER BY sequence",
                (submission_id,),
            )

        assert review["final_state"] == "rejected"
        assert int(review["approved"]) == 0
        assert "smuggled pretrained weights" in str(review["reason"])
        assert submission["status"] == "rejected"
        assert "smuggled pretrained weights" in str(submission["error"])
        states = [str(row["state"]) for row in events]
        assert "verdict_submitted" in states
        assert "rejected" in states
        assert "quarantined" not in states

    anyio.run(run)


def test_llm_hard_gate_reject_stops_before_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_load_chat_openai", lambda: _fake_chat_class(_verdict(False, REJECT_REASON))
    )

    def fail_run(self, spec, timeout_seconds):
        raise AssertionError("container must not run after an LLM hard-gate reject")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)

    db_name = "hard-gate-reject.sqlite3"
    settings = _hard_gate_settings(tmp_path, db_name)
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, nonce="hard-gate-reject")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "rejected"
        assert status["q_arch"] is None
        assert "dead no-op loop" in status["error"]

    async def _inspect() -> None:
        database = Database(tmp_path / db_name)
        async with database.connect() as conn:
            review = list(
                await conn.execute_fetchall(
                    "SELECT approved, reason, final_state FROM llm_reviews WHERE submission_id=?",
                    (submission_id,),
                )
            )[0]
            scores = await conn.execute_fetchall(
                "SELECT 1 FROM scores WHERE submission_id=?", (submission_id,)
            )
            gpu_leases = await conn.execute_fetchall(
                "SELECT 1 FROM gpu_leases WHERE submission_id=?", (submission_id,)
            )
            gpu_jobs = await conn.execute_fetchall(
                "SELECT 1 FROM eval_jobs WHERE submission_id=? AND level != 'l1'",
                (submission_id,),
            )
        assert int(review["approved"]) == 0
        assert review["final_state"] == "rejected"
        assert str(review["reason"]).strip()
        assert list(scores) == []
        assert list(gpu_leases) == []
        assert list(gpu_jobs) == []

    anyio.run(_inspect)


def test_llm_hard_gate_allow_proceeds_to_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_load_chat_openai", lambda: _fake_chat_class(_verdict(True, ALLOW_REASON))
    )

    def fake_run(self, spec, timeout_seconds):
        _write_v2_manifest(spec)
        return DockerRunResult(container_name="prism-eval", stdout="", stderr="", returncode=0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)

    settings = _hard_gate_settings(tmp_path, "hard-gate-allow.sqlite3")
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, nonce="hard-gate-allow")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] is not None
        assert float(status["q_arch"]) > 0.0
