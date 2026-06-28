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
from prism_challenge.evaluator.llm_review import (
    SAFETY_REVIEW_INSTRUCTIONS,
    SAFETY_REVIEW_SYSTEM,
    LlmReviewConfig,
    build_safety_review_prompt,
    review_code,
)
from prism_challenge.sdk.executors.docker import DockerRunResult

# Reason strings mirror what the live claude-opus-4.8 gate now returns for the three known
# obfuscation vectors (runtime-decoded hex blob, dynamic setattr rebind, requires_grad
# anti-anti-cheat toggle).
OBFUSCATION_REJECT_REASON = (
    "The training script decodes a runtime hex blob and toggles requires_grad to drive control "
    "flow, which is obfuscation/evasion of the static sandbox and anti-cheat checks."
)
GENUINE_ALLOW_REASON = (
    "Architecture and training loop are coherent; a real from-scratch learning procedure with no "
    "obfuscation or evasion."
)


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


def _verdict(verdict: bool, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "verdict": verdict,
        "violations": ["prism:obfuscation"] if not verdict else [],
        "confidence": 0.95,
        "rule_ids": [],
        "evidence": [],
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


def _settings(tmp_path, name: str) -> PrismSettings:
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


# --- VAL-LLM-007: the safety prompt/criteria enumerate the obfuscation/evasion red flags ---


def test_safety_prompt_enumerates_obfuscation_red_flags() -> None:
    combined = (SAFETY_REVIEW_SYSTEM + "\n" + SAFETY_REVIEW_INSTRUCTIONS).lower()
    prompt = build_safety_review_prompt(
        subject="Prism project",
        rules_text="- demo: rule",
        code="# file: architecture.py\n# file: training.py\n",
    ).lower()
    text = combined + "\n" + prompt

    # Runtime-decoded encoded blobs (the contract's primary obfuscation example).
    assert "fromhex" in text
    assert "base64" in text
    assert (
        "decoded at runtime" in text
        or "decoded during execution" in text
        or ("decode" in text and "runtime" in text)
    )
    # Single-letter / variable-driven indirection.
    assert "single-letter" in text
    assert "indirection" in text
    # Dynamic getattr/setattr attribute rebinding/dispatch to evade the static AST sandbox.
    assert "getattr" in text and "setattr" in text
    assert "artifacts_dir" in text
    assert "evade the static" in text or "evade the ast" in text or "static ast sandbox" in text
    # requires_grad / grad toggling whose only purpose is to evade anti-cheat/anomaly detection.
    assert "requires_grad" in text
    assert "anomaly" in text or "anti-cheat" in text or "step-0" in text
    # Instruct the model to cite obfuscation/evasion when intent is hidden.
    assert "obfuscation" in text and "evasion" in text
    assert "must contain the word 'obfuscation' or 'evasion'" in text


def test_obfuscation_verdict_is_terminal_reject_not_held() -> None:
    # A model obfuscation/evasion verdict (verdict=false) is a TERMINAL reject, never downgraded
    # to a hold for lacking a 64-char evidence hash.
    import prism_challenge.evaluator.llm_review as module

    original = module._load_chat_openai
    module._load_chat_openai = lambda: _fake_chat_class(_verdict(False, OBFUSCATION_REJECT_REASON))
    try:
        review = review_code(
            "# file: architecture.py\n# file: training.py\n",
            config=LlmReviewConfig(api_key="sk-or-test"),
        )
    finally:
        module._load_chat_openai = original

    assert review.approved is False
    assert review.held is False
    assert "obfuscation" in review.reason.lower() or "evasion" in review.reason.lower()
    assert "suspicion without evidence" not in review.reason


def test_obfuscation_verdict_rejects_terminally_pre_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        llm,
        "_load_chat_openai",
        lambda: _fake_chat_class(_verdict(False, OBFUSCATION_REJECT_REASON)),
    )

    def fail_run(self, spec, timeout_seconds):
        raise AssertionError("container must not run after an obfuscation hard-gate reject")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)

    db_name = "obfuscation-reject.sqlite3"
    with TestClient(create_app(_settings(tmp_path, db_name))) as client:
        submission_id = _submit(client, nonce="obf-reject")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "rejected"
        assert status["q_arch"] is None
        assert "obfuscation" in status["error"].lower() or "evasion" in status["error"].lower()

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
        assert (
            "obfuscation" in str(review["reason"]).lower()
            or "evasion" in str(review["reason"]).lower()
        )
        assert list(scores) == []
        assert list(gpu_leases) == []
        assert list(gpu_jobs) == []

    anyio.run(_inspect)


def test_genuine_allow_verdict_still_proceeds_to_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "_load_chat_openai", lambda: _fake_chat_class(_verdict(True, GENUINE_ALLOW_REASON))
    )

    def fake_run(self, spec, timeout_seconds):
        _write_v2_manifest(spec)
        return DockerRunResult(container_name="prism-eval", stdout="", stderr="", returncode=0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)

    with TestClient(create_app(_settings(tmp_path, "obfuscation-allow.sqlite3"))) as client:
        submission_id = _submit(client, nonce="obf-allow")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] is not None
        assert float(status["q_arch"]) > 0.0
