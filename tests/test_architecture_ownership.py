from __future__ import annotations

import base64
import io
import json
import zipfile

import anyio
from conftest import signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
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


def _zip_payload(kind: str = "architecture_only") -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "prism.yaml",
            "\n".join(
                [
                    f"kind: {kind}",
                    "architecture:",
                    "  entrypoint: src/model.py",
                    "training:",
                    "  entrypoint: src/train.py",
                ]
            )
            + "\n",
        )
        archive.writestr("src/model.py", MODEL_CODE)
        archive.writestr(
            "src/train.py",
            "from prism_challenge.evaluator.interface import TrainingRecipe\n\n"
            "def recipe(ctx):\n"
            "    return TrainingRecipe(learning_rate=0.0003, batch_size=2)\n\n"
            "def train(ctx):\n"
            "    return None\n",
        )
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _submit_zip(client: TestClient, hotkey: str, nonce: str) -> str:
    payload = {"code": _zip_payload(), "filename": "project.zip"}
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


def test_immutable_owner_survives_duplicate_and_better_architecture_submissions(
    tmp_path, monkeypatch
):
    metrics = iter(
        [
            {"q_arch": 0.70, "q_recipe": 0.50},
            {"q_arch": 0.71, "q_recipe": 0.50},
            {"q_arch": 0.90, "q_recipe": 0.50},
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

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'architecture.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        architecture_transfer_min_delta_abs=0.08,
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )

    with TestClient(create_app(settings)) as client:
        owner_submission = _submit_zip(client, "arch-owner-a", "arch-owner-a-1")
        _process(client)
        duplicate_submission = _submit_zip(client, "arch-duplicate-b", "arch-duplicate-b-1")
        _process(client)
        improved_submission = _submit_zip(client, "improved-arch-c", "improved-arch-c-1")
        _process(client)

        architectures = client.get("/v1/architectures").json()
        assert len(architectures) == 1
        architecture = architectures[0]
        assert architecture["owner_hotkey"] == "arch-owner-a"
        assert architecture["owner_submission_id"] == owner_submission
        assert architecture["canonical_submission_id"] == improved_submission
        assert architecture["q_arch_best"] == 0.90

        async def read_rows():
            async with client.app.state.database.connect() as conn:
                versions = await conn.execute_fetchall(
                    "SELECT submitter_hotkey, owner_hotkey, submission_id, is_canonical, "
                    "is_owner_version, canonical_graph_hash, canonical_metadata_path "
                    "FROM architecture_versions ORDER BY version_index"
                )
                events = await conn.execute_fetchall(
                    "SELECT event FROM ownership_events WHERE scope='architecture' "
                    "ORDER BY created_at"
                )
            return [dict(row) for row in versions], [str(row["event"]) for row in events]

        versions, events = anyio.run(read_rows)
        assert [row["submission_id"] for row in versions] == [
            owner_submission,
            duplicate_submission,
            improved_submission,
        ]
        assert {row["owner_hotkey"] for row in versions} == {"arch-owner-a"}
        assert versions[0]["is_owner_version"] == 1
        assert versions[1]["is_owner_version"] == 0
        assert versions[2]["is_canonical"] == 1
        assert len(versions[2]["canonical_graph_hash"]) == 64
        assert versions[2]["canonical_metadata_path"].endswith("architecture_metadata.v1.json")
        assert "transferred" not in events
