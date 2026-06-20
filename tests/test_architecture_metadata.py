from __future__ import annotations

import base64
import io
import json
import zipfile

import anyio
import pytest
from conftest import signed_headers
from fastapi.testclient import TestClient
from pydantic import ValidationError

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.schemas import (
    ARCHITECTURE_GRAPH_FILENAME,
    ArchitectureGraph,
    ArchitectureMetadata,
    DeterministicEvidence,
    DuplicateReport,
)
from prism_challenge.sdk.executors.docker import DockerRunResult

GRAPH_HASH = "e" * 64
SNIPPET_HASH = "f" * 64


def _architecture_metadata() -> dict:
    return {
        "schema_version": "architecture_metadata.v1",
        "identity": {
            "architecture_id": "architecture-1",
            "architecture_version_id": "architecture-version-1",
            "architecture_graph_hash": GRAPH_HASH,
            "architecture_source_hash": "1" * 64,
            "evaluation_config_id": "eval-config-1",
            "owner_hotkey": "owner-hotkey",
        },
        "graph": {
            "canonical_artifact": ARCHITECTURE_GRAPH_FILENAME,
            "hash": {
                "algorithm": "sha256",
                "value": GRAPH_HASH,
                "canonicalization": "json-sort-keys-no-whitespace",
            },
            "node_count": 4,
            "edge_count": 3,
            "source_free_comparison_keys": ["modules", "classes", "functions", "calls"],
        },
        "derived_mermaid_path": "artifacts/architecture.mmd",
        "architecture_summary": "architecture: classes=tiny; functions=build_model",
        "training_summary": "training: functions=get_recipe",
        "overview": {"project_kind": "full", "primary_classes": ["tiny"]},
        "difficulty": {"tier": "low", "complexity_score": 7},
        "comparison": {"architecture_graph_hash": GRAPH_HASH, "calls": ["nn.Linear"]},
        "comparison_tags": ["embedding", "linear-head"],
        "deterministic_evidence": [
            {
                "rule_id": "architecture.identity.graph_hash",
                "artifact_path": "artifacts/architecture_graph.json",
                "line": 12,
                "snippet_hash": SNIPPET_HASH,
                "explanation": "canonical graph contains the model class and build hook",
            }
        ],
    }


def test_architecture_graph_json_is_canonical_identity() -> None:
    graph = ArchitectureGraph.model_validate(
        {
            "modules": ["model"],
            "classes": ["tiny"],
            "functions": ["build_model"],
            "imports": ["torch"],
            "calls": ["nn.Linear"],
            "parameterized_blocks": [{"name": "embedding", "parameters": 32768}],
            "tokenizer_constraints": {"kind": "fixed_prism_fixture"},
            "dynamic_routing": {"enabled": False},
            "interface": {"required": ["build_model", "get_recipe"]},
        }
    )

    assert "mermaid" not in graph.model_dump()
    assert len(graph.sha256()) == 64


def test_architecture_metadata_validates_without_source_code() -> None:
    metadata = ArchitectureMetadata.model_validate(_architecture_metadata())

    assert metadata.graph.canonical_artifact == "architecture_graph.json"
    assert metadata.derived_mermaid_path == "artifacts/architecture.mmd"
    assert metadata.identity.architecture_graph_hash == metadata.graph.hash.value
    assert metadata.overview["project_kind"] == "full"
    assert metadata.difficulty["tier"] == "low"
    assert metadata.comparison["architecture_graph_hash"] == GRAPH_HASH


def test_architecture_metadata_rejects_mermaid_as_canonical() -> None:
    payload = _architecture_metadata()
    payload["graph"]["canonical_artifact"] = "architecture.mmd"

    with pytest.raises(ValidationError):
        ArchitectureMetadata.model_validate(payload)


def test_architecture_metadata_rejects_hash_mismatch() -> None:
    payload = _architecture_metadata()
    payload["graph"]["hash"]["value"] = "0" * 64

    with pytest.raises(ValidationError, match="architecture_graph_hash"):
        ArchitectureMetadata.model_validate(payload)


def test_deterministic_evidence_requires_file_location() -> None:
    with pytest.raises(ValidationError, match="line or ast_node"):
        DeterministicEvidence.model_validate(
            {
                "rule_id": "safety.no_network",
                "artifact_path": "model.py",
                "snippet_hash": SNIPPET_HASH,
                "explanation": "network import was detected",
            }
        )


def test_duplicate_report_schema_carries_source_free_comparison() -> None:
    report = DuplicateReport.model_validate(
        {
            "submission_id": "submission-2",
            "architecture_graph_hash": GRAPH_HASH,
            "candidate_architecture_id": "architecture-1",
            "candidate_architecture_graph_hash": GRAPH_HASH,
            "source_similarity": 0.93,
            "graph_similarity": 1.0,
            "semantic_similarity": 0.97,
            "outcome": "quarantine",
            "evidence": [
                {
                    "rule_id": "duplicate.graph_similarity",
                    "artifact_path": "artifacts/architecture_graph.json",
                    "ast_node": "ClassDef:TinyModel",
                    "snippet_hash": SNIPPET_HASH,
                    "explanation": "candidate shares canonical graph structure",
                }
            ],
            "reason": "same canonical architecture graph needs deterministic review",
        }
    )

    assert report.outcome == "quarantine"
    assert report.evidence[0].ast_node == "ClassDef:TinyModel"


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


def _zip_payload_with_submitted_mermaid() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "prism.yaml",
            "\n".join(
                [
                    "kind: full",
                    "architecture:",
                    "  entrypoint: src/model.py",
                    "  mermaid: malicious_user_supplied_mermaid",
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
    payload = {
        "code": _zip_payload_with_submitted_mermaid(),
        "filename": "project.zip",
        "metadata": {"mermaid": "malicious_user_supplied_mermaid"},
    }
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


def test_accepted_architecture_emits_source_free_artifacts_and_derived_mermaid(
    tmp_path, monkeypatch
) -> None:
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"q_arch":0.82,"q_recipe":0.61}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'metadata.sqlite3'}",
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
    )

    with TestClient(create_app(settings)) as client:
        submission_id = _submit_zip(client, "metadata-miner", "metadata-1")
        _process(client)

        async def read_artifacts() -> dict[str, object]:
            async with client.app.state.database.connect() as conn:
                rows = await conn.execute_fetchall(
                    "SELECT af.canonical_graph_hash AS family_graph_hash, "
                    "af.canonical_graph_path AS family_graph_path, "
                    "af.canonical_metadata_path AS family_metadata_path, "
                    "af.canonical_mermaid_path AS family_mermaid_path, "
                    "av.id AS version_id, av.architecture_id, av.canonical_graph_hash, "
                    "av.canonical_graph_path, av.canonical_metadata_path, "
                    "av.derived_mermaid_path, cs.architecture_graph, "
                    "cs.architecture_graph_hash AS signature_graph_hash, "
                    "cs.architecture_metadata, cs.mermaid, cs.derived_mermaid_path "
                    "AS signature_mermaid_path "
                    "FROM architecture_versions av "
                    "JOIN architecture_families af ON af.id=av.architecture_id "
                    "JOIN component_signatures cs ON cs.submission_id=av.submission_id "
                    "WHERE av.submission_id=?",
                    (submission_id,),
                )
            assert len(rows) == 1
            return dict(rows[0])

        row = anyio.run(read_artifacts)

    graph = ArchitectureGraph.model_validate(json.loads(str(row["architecture_graph"])))
    metadata = ArchitectureMetadata.model_validate(json.loads(str(row["architecture_metadata"])))
    source_free_fixture = {
        "metadata": metadata.model_dump(mode="json"),
        "mermaid": str(row["mermaid"]),
    }

    assert graph.sha256() == row["canonical_graph_hash"]
    assert row["signature_graph_hash"] == row["canonical_graph_hash"]
    assert row["family_graph_hash"] == row["canonical_graph_hash"]
    assert metadata.identity.architecture_id == row["architecture_id"]
    assert metadata.identity.architecture_version_id == row["version_id"]
    assert metadata.identity.architecture_graph_hash == row["canonical_graph_hash"]
    assert metadata.graph.hash.value == row["canonical_graph_hash"]
    assert metadata.overview["project_kind"] == "full"
    assert metadata.difficulty["parameterized_block_count"] >= 2
    assert metadata.comparison["architecture_graph_hash"] == row["canonical_graph_hash"]
    assert row["canonical_graph_path"].endswith("architecture_graph.json")
    assert row["canonical_metadata_path"].endswith("architecture_metadata.v1.json")
    assert row["derived_mermaid_path"].endswith("architecture.mmd")
    assert row["family_graph_path"] == row["canonical_graph_path"]
    assert row["family_metadata_path"] == row["canonical_metadata_path"]
    assert row["family_mermaid_path"] == row["derived_mermaid_path"]
    assert row["signature_mermaid_path"] == row["derived_mermaid_path"]
    assert source_free_fixture["mermaid"].startswith("flowchart LR")
    assert "malicious_user_supplied_mermaid" not in source_free_fixture["mermaid"]
