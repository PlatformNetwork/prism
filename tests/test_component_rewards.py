from __future__ import annotations

import base64
import io
import json
import zipfile

from conftest import signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.component_agents import ComponentOwnershipDecision
from prism_challenge.sdk.executors.docker import DockerRunResult

MODEL_CODE = """
import torch
from train import recipe

class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.linear = torch.nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.linear(self.embedding(tokens))

def build_model(ctx):
    return TinyModel(ctx.vocab_size)

def get_recipe(ctx):
    return recipe(ctx)
"""


def _zip_payload(*, learning_rate: float, kind: str, architecture_id: str | None = None) -> str:
    manifest = [
        f"kind: {kind}",
        "architecture:",
        "  entrypoint: src/model.py",
        "training:",
        "  entrypoint: src/train.py",
    ]
    if architecture_id is not None:
        manifest.insert(1, f"architecture_id: {architecture_id}")
    train_code = f"""
from prism_challenge.evaluator.interface import TrainingRecipe

def recipe(ctx):
    return TrainingRecipe(learning_rate={learning_rate!r}, batch_size=2)
"""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("prism.yaml", "\n".join(manifest) + "\n")
        archive.writestr("src/model.py", MODEL_CODE)
        archive.writestr("src/train.py", train_code)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _submit_zip(client: TestClient, hotkey: str, nonce: str, code: str) -> str:
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


def _process(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def test_component_rewards_split_architecture_and_training(tmp_path, monkeypatch):
    metrics = iter(
        [
            {"q_arch": 0.9, "q_recipe": 0.5, "q_recipe_std": 0.0},
            {"q_arch": 0.9, "q_recipe": 0.8, "q_recipe_std": 0.0},
            {"q_arch": 0.9, "q_recipe": 0.805, "q_recipe_std": 0.0},
        ]
    )

    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout="PRISM_METRICS_JSON=" + json.dumps(next(metrics), separators=(",", ":")) + "\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'components.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        training_improvement_min_delta_abs=0.02,
        training_improvement_z_score=1.0,
    )

    with TestClient(create_app(settings)) as client:
        architect_submission = _submit_zip(
            client,
            "architect",
            "architect-1",
            _zip_payload(learning_rate=0.0003, kind="full"),
        )
        _process(client)
        architectures = client.get("/v1/architectures").json()
        assert len(architectures) == 1
        assert architectures[0]["owner_hotkey"] == "architect"
        assert architectures[0]["owner_submission_id"] == architect_submission
        architecture_id = architectures[0]["id"]

        trainer_submission = _submit_zip(
            client,
            "trainer",
            "trainer-1",
            _zip_payload(
                learning_rate=0.001,
                kind="training_for_arch",
                architecture_id=architecture_id,
            ),
        )
        _process(client)

        noisy_submission = _submit_zip(
            client,
            "noisy",
            "noisy-1",
            _zip_payload(
                learning_rate=0.0011,
                kind="training_for_arch",
                architecture_id=architecture_id,
            ),
        )
        _process(client)

        training = client.get(f"/v1/training-variants?architecture_id={architecture_id}").json()
        current = [variant for variant in training if variant["is_current_best"]]
        assert [variant["submission_id"] for variant in current] == [trainer_submission]
        assert noisy_submission not in {variant["submission_id"] for variant in current}

        weights = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
        ).json()["weights"]
        assert weights["architect"] > 0
        assert weights["trainer"] > 0
        assert "noisy" not in weights


def test_low_confidence_component_review_is_held_and_resolved(tmp_path, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"q_arch":0.9,"q_recipe":0.7}\n',
            stderr="",
            returncode=0,
        )

    def hold_decision(self, **kwargs):
        return ComponentOwnershipDecision(
            architecture_action="hold",
            architecture_confidence=0.4,
            training_action="hold",
            training_confidence=0.4,
            reason="ambiguous semantic ownership",
            raw={"test": "hold"},
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    monkeypatch.setattr("prism_challenge.queue.SemanticOwnershipAgent.decide", hold_decision)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'holds.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
    )

    with TestClient(create_app(settings)) as client:
        submission_id = _submit_zip(
            client,
            "held-miner",
            "held-1",
            _zip_payload(learning_rate=0.0003, kind="full"),
        )
        _process(client)
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "held"

        holds = client.get(
            "/internal/v1/component-review/holds",
            headers={"Authorization": "Bearer secret"},
        ).json()
        assert holds[0]["submission_id"] == submission_id

        weights = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
        ).json()["weights"]
        assert weights == {}

        resolved = client.post(
            f"/internal/v1/component-review/holds/{holds[0]['id']}/resolve",
            json={
                "architecture_action": "new",
                "training_action": "new",
                "reason": "manual approval",
            },
            headers={"Authorization": "Bearer secret"},
        )
        assert resolved.status_code == 200, resolved.text
        assert client.get(f"/v1/submissions/{submission_id}").json()["status"] == "completed"
        weights = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
        ).json()["weights"]
        assert weights == {"held-miner": 1.0}
