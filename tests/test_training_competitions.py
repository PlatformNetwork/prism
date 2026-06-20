from __future__ import annotations

import base64
import io
import json
import zipfile
from copy import deepcopy

import anyio
import pytest
from conftest import signed_headers
from fastapi.testclient import TestClient
from test_artifact_manifest import _valid_manifest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.component_agents import ComponentOwnershipDecision
from prism_challenge.evaluator.schemas import ExecutionMode
from prism_challenge.evaluator.scoring import score_training_manifest
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

def train(ctx):
    return None
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


def test_training_current_best_updates_without_architecture_owner_transfer(
    tmp_path, monkeypatch
):
    metrics = iter(
        [
            {"q_arch": 0.90, "q_recipe": 0.50, "q_recipe_std": 0.0},
            {"q_arch": 0.90, "q_recipe": 0.60, "q_recipe_std": 0.0},
            {"q_arch": 0.95, "q_recipe": 0.86, "q_recipe_std": 0.0},
        ]
    )

    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout=(
                "PRISM_METRICS_JSON="
                + json.dumps(next(metrics), separators=(",", ":"))
                + "\n"
            ),
            stderr="",
            returncode=0,
        )

    def force_new_training(
        self, *, signature, architecture_candidates, training_candidates, requested_architecture_id
    ):
        if not architecture_candidates:
            return ComponentOwnershipDecision(
                architecture_action="new",
                architecture_confidence=1.0,
                training_action="new",
                training_confidence=1.0,
                reason="test creates initial architecture",
                raw={"test": "initial"},
            )
        return ComponentOwnershipDecision(
            architecture_action="existing",
            architecture_confidence=1.0,
            training_action="new",
            training_confidence=1.0,
            matched_architecture_id=str(architecture_candidates[0]["id"]),
            reason="test adds a new training script for the existing architecture",
            raw={"test": "training"},
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    monkeypatch.setattr("prism_challenge.queue.SemanticOwnershipAgent.decide", force_new_training)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'training.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
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
            "arch-owner-a",
            "arch-owner-a-1",
            _zip_payload(learning_rate=0.0003, kind="full"),
        )
        _process(client)
        architecture = client.get("/v1/architectures").json()[0]
        architecture_id = architecture["id"]

        trainer_a_submission = _submit_zip(
            client,
            "trainer-a",
            "trainer-a-1",
            _zip_payload(
                learning_rate=0.001,
                kind="training_for_arch",
                architecture_id=architecture_id,
            ),
        )
        _process(client)
        trainer_b_submission = _submit_zip(
            client,
            "trainer-b",
            "trainer-b-1",
            _zip_payload(
                learning_rate=0.002,
                kind="training_for_arch",
                architecture_id=architecture_id,
            ),
        )
        _process(client)

        architecture_after = client.get("/v1/architectures").json()[0]
        assert architecture_after["owner_hotkey"] == "arch-owner-a"
        assert architecture_after["owner_submission_id"] == architect_submission
        assert architecture_after["canonical_submission_id"] == architect_submission

        training = client.get(f"/v1/training-variants?architecture_id={architecture_id}").json()
        current = [variant for variant in training if variant["is_current_best"]]
        assert [variant["owner_hotkey"] for variant in current] == ["trainer-b"]
        assert [variant["submission_id"] for variant in current] == [trainer_b_submission]
        assert trainer_a_submission in {variant["submission_id"] for variant in training}

        async def read_rows():
            async with client.app.state.database.connect() as conn:
                versions = await conn.execute_fetchall(
                    "SELECT submitter_hotkey, owner_hotkey, submission_id, is_current_best "
                    "FROM training_script_versions WHERE architecture_id=? ORDER BY version_index",
                    (architecture_id,),
                )
                tuples = await conn.execute_fetchall(
                    "SELECT submission_id, architecture_version_id, training_script_version_id, "
                    "architecture_graph_hash FROM official_evaluated_tuples ORDER BY created_at"
                )
            return [dict(row) for row in versions], [dict(row) for row in tuples]

        versions, tuples = anyio.run(read_rows)
        assert [row["submission_id"] for row in versions] == [
            architect_submission,
            trainer_a_submission,
            trainer_b_submission,
        ]
        assert versions[-1]["submitter_hotkey"] == "trainer-b"
        assert versions[-1]["owner_hotkey"] == "trainer-b"
        assert versions[-1]["is_current_best"] == 1
        assert tuples[-1]["submission_id"] == trainer_b_submission
        assert tuples[-1]["training_script_version_id"] is not None
        assert len(tuples[-1]["architecture_graph_hash"]) == 64


def test_architecture_normalized_loss_component_ignores_global_final_loss() -> None:
    arch_a = _training_manifest(
        architecture_id="arch-a",
        raw_final_loss=0.8,
        architecture_normalized_heldout_improvement=0.10,
    )
    arch_b = _training_manifest(
        architecture_id="arch-b",
        raw_final_loss=1.2,
        architecture_normalized_heldout_improvement=0.40,
    )

    score_a = score_training_manifest(arch_a)
    score_b = score_training_manifest(arch_b)

    assert score_a.component_values["architecture_normalized_heldout_improvement"] == pytest.approx(
        0.10
    )
    assert score_b.component_values["architecture_normalized_heldout_improvement"] == pytest.approx(
        0.40
    )
    assert score_a.details["raw_final_loss_used"] is False
    assert score_b.details["raw_final_loss_used"] is False
    assert score_b.score > score_a.score


def _training_manifest(
    *,
    architecture_id: str,
    raw_final_loss: float,
    architecture_normalized_heldout_improvement: float,
) -> dict:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    payload["architecture_id"] = architecture_id
    payload["metrics"]["benchmark_scores"] = {
        "gsm8k": 0.5,
        "math": 0.5,
        "arc_challenge": 0.5,
        "humaneval": 0.5,
        "mmlu": 0.5,
        "ifeval": 0.5,
        "truthfulqa": 0.5,
        "needle": 0.5,
    }
    payload["metrics"]["final_loss"] = raw_final_loss
    payload["metrics"]["loss"]["raw_final_loss"] = raw_final_loss
    payload["metrics"]["loss"][
        "architecture_normalized_heldout_improvement"
    ] = architecture_normalized_heldout_improvement
    return payload
