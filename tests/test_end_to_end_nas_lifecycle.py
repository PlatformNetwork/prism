from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import anyio
from conftest import signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import llm_review
from prism_challenge.sdk.executors.docker import DockerRunResult

MODEL_TEMPLATE = "\n".join(
    [
        "import torch",
        "from train import recipe",
        "",
        "class {class_name}(torch.nn.Module):",
        "    def __init__(self, vocab_size):",
        "        super().__init__()",
        "        self.embedding = torch.nn.Embedding(vocab_size, 8)",
        "        self.proj = torch.nn.Linear(8, 8)",
        "        self.output = torch.nn.Linear(8, vocab_size)",
        "",
        "    def forward(self, tokens):",
        "        hidden = torch.relu(self.proj(self.embedding(tokens)))",
        "        return self.output(hidden)",
        "",
        "def build_model(ctx):",
        "    return {class_name}(ctx.vocab_size)",
        "",
        "def get_recipe(ctx):",
        "    return recipe(ctx)",
        "",
    ]
)


def _settings(db_path: Path, *, plagiarism_enabled: bool) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        component_agent_enabled=False,
        plagiarism_enabled=plagiarism_enabled,
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
        training_improvement_min_delta_abs=0.02,
        training_improvement_z_score=1.0,
    )


def _zip_payload(
    *,
    learning_rate: float,
    kind: str = "full",
    class_name: str = "LifecycleModel",
    architecture_id: str | None = None,
) -> str:
    manifest = [
        f"kind: {kind}",
        "architecture:",
        "  entrypoint: src/model.py",
        "training:",
        "  entrypoint: src/train.py",
    ]
    if architecture_id is not None:
        manifest.insert(1, f"architecture_id: {architecture_id}")
    train_code = "\n".join(
        [
            "from prism_challenge.evaluator.interface import TrainingRecipe",
            "",
            "def recipe(ctx):",
            f"    return TrainingRecipe(learning_rate={learning_rate!r}, batch_size=2)",
            "",
            "def train(ctx):",
            "    return None",
            "",
        ]
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("prism.yaml", "\n".join(manifest) + "\n")
        archive.writestr("src/model.py", MODEL_TEMPLATE.format(class_name=class_name))
        archive.writestr("src/train.py", train_code)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _submit_zip(client: TestClient, *, hotkey: str, nonce: str, code: str) -> str:
    payload = {"code": code, "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey=hotkey, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient, expected_submission_id: str | None = None) -> str | None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text
    processed = response.json()["submission_id"]
    if expected_submission_id is not None:
        assert processed == expected_submission_id
    return processed


def _weights(client: TestClient) -> dict[str, float]:
    response = client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
    )
    assert response.status_code == 200, response.text
    return dict(response.json()["weights"])


def _fake_manifest_runner(scores: list[tuple[float, float]]):
    """Simulate the challenge-authored re-execution runner emitting challenge-computed metrics.

    The forced-init runner (container.py) computes the score itself and surfaces it; the bpb
    scoring recast fills in the v2 manifest metrics. Here the (mocked) broker emits the
    challenge metrics on stdout so the worker finalizes the architecture/training lifecycle.
    """
    remaining = iter(scores)

    def fake_run(self, spec, timeout_seconds):
        architecture_quality, training_quality = next(remaining)
        metrics: dict[str, Any] = {
            "q_arch": architecture_quality,
            "q_recipe": training_quality,
            "penalty": 0.0,
        }
        return DockerRunResult(
            "platform-e2e-job",
            "PRISM_METRICS_JSON=" + json.dumps(metrics, separators=(",", ":")) + "\n",
            "",
            0,
        )

    return fake_run


def test_architecture_then_training_lifecycle_updates_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        _fake_manifest_runner([(0.55, 0.15), (0.55, 0.62)]),
    )

    settings = _settings(tmp_path / "e2e.sqlite3", plagiarism_enabled=False)
    with TestClient(create_app(settings)) as client:
        architecture_submission = _submit_zip(
            client,
            hotkey="architect-e2e",
            nonce="architect-e2e-1",
            code=_zip_payload(learning_rate=0.0003),
        )
        _process(client, architecture_submission)
        architecture_status = client.get(f"/v1/submissions/{architecture_submission}").json()
        architectures = client.get("/v1/architectures").json()
        architecture_id = architectures[0]["id"]

        async def architecture_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            repository = client.app.state.repository
            component_rows = await repository.component_weight_rows(
                architecture_weight=0.60,
                training_weight=0.40,
            )
            async with repository.database.connect() as conn:
                rows = await conn.execute_fetchall(
                    "SELECT owner_hotkey, owner_submission_id, canonical_graph_path, "
                    "canonical_metadata_path, canonical_mermaid_path "
                    "FROM architecture_families WHERE id=?",
                    (architecture_id,),
                )
            return [dict(row) for row in rows], component_rows

        architecture_db_rows, component_rows = anyio.run(architecture_rows)
        assert architecture_status["status"] == "completed"
        assert architecture_status["q_arch"] > 0
        assert architectures[0]["owner_hotkey"] == "architect-e2e"
        assert architectures[0]["owner_submission_id"] == architecture_submission
        assert architecture_db_rows == [
            {
                "owner_hotkey": "architect-e2e",
                "owner_submission_id": architecture_submission,
                "canonical_graph_path": (
                    f"artifacts/{architecture_submission}/architecture_graph.json"
                ),
                "canonical_metadata_path": (
                    f"artifacts/{architecture_submission}/architecture_metadata.v1.json"
                ),
                "canonical_mermaid_path": f"artifacts/{architecture_submission}/architecture.mmd",
            }
        ]
        assert any(
            row["component"] == "architecture" and row["hotkey"] == "architect-e2e"
            for row in component_rows
        )

        training_submission = _submit_zip(
            client,
            hotkey="trainer-e2e",
            nonce="trainer-e2e-1",
            code=_zip_payload(
                learning_rate=0.001,
                kind="training_for_arch",
                architecture_id=architecture_id,
            ),
        )
        _process(client, training_submission)
        training_status = client.get(f"/v1/submissions/{training_submission}").json()
        architectures_after = client.get("/v1/architectures").json()
        training = client.get(f"/v1/training-variants?architecture_id={architecture_id}").json()
        current_training = [row for row in training if row["is_current_best"]]
        weights = _weights(client)

    assert training_status["status"] == "completed"
    assert architectures_after[0]["owner_hotkey"] == "architect-e2e"
    assert architectures_after[0]["owner_submission_id"] == architecture_submission
    assert architectures_after[0]["canonical_submission_id"] == architecture_submission
    assert [row["owner_hotkey"] for row in current_training] == ["trainer-e2e"]
    assert [row["submission_id"] for row in current_training] == [training_submission]
    assert weights["architect-e2e"] > 0
    assert weights["trainer-e2e"] > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_quarantined_no_weight_and_duplicate_reject_status(tmp_path, monkeypatch):
    original_review = llm_review.review_code

    def conditional_review(code, **kwargs):
        if "SuspiciousLifecycleModel" in code:
            return llm_review.LlmReview(
                approved=False,
                reason="LLM suspicion without deterministic evidence",
                violations=["suspicious_similarity"],
                scores=[0, 0],
                confidence=0.8,
                raw={"review": "suspicion without evidence"},
                mermaid="graph TD; A[Submission]-->B[Quarantine]",
                evidence=[],
                held=True,
            )
        return original_review(code, **kwargs)

    monkeypatch.setattr("prism_challenge.evaluator.llm_review.review_code", conditional_review)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        _fake_manifest_runner([(0.50, 0.20)]),
    )

    settings = _settings(tmp_path / "quarantine.sqlite3", plagiarism_enabled=True)
    with TestClient(create_app(settings)) as client:
        accepted_payload = _zip_payload(learning_rate=0.0003)
        accepted = _submit_zip(
            client,
            hotkey="accepted-e2e",
            nonce="accepted-e2e-1",
            code=accepted_payload,
        )
        _process(client, accepted)

        duplicate = _submit_zip(
            client,
            hotkey="duplicate-e2e",
            nonce="duplicate-e2e-1",
            code=accepted_payload,
        )
        _process(client, duplicate)

        quarantined = _submit_zip(
            client,
            hotkey="quarantined-e2e",
            nonce="quarantined-e2e-1",
            code=_zip_payload(
                learning_rate=0.0004,
                class_name="SuspiciousLifecycleModel",
            ),
        )
        _process(client, quarantined)

        accepted_status = client.get(f"/v1/submissions/{accepted}").json()
        duplicate_status = client.get(f"/v1/submissions/{duplicate}").json()
        quarantined_status = client.get(f"/v1/submissions/{quarantined}").json()
        weights = _weights(client)

    assert accepted_status["status"] == "completed"
    assert duplicate_status["status"] == "rejected"
    assert "exact source hash duplicate" in duplicate_status["error"]
    assert quarantined_status["status"] == "held"
    assert quarantined_status["error"] == "LLM suspicion without deterministic evidence"
    assert weights == {"accepted-e2e": 1.0}
