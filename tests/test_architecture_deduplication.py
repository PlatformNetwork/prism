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
from prism_challenge.db import loads
from prism_challenge.sdk.executors.docker import DockerRunResult

TINY_MODEL = "\n".join(
    [
        "import torch",
        "from train import recipe",
        "",
        "class TinyModel(torch.nn.Module):",
        "    def __init__(self, vocab_size):",
        "        super().__init__()",
        "        self.embedding = torch.nn.Embedding(vocab_size, 8)",
        "        self.linear = torch.nn.Linear(8, vocab_size)",
        "",
        "    def forward(self, tokens):",
        "        return self.linear(self.embedding(tokens))",
        "",
        "def build_model(ctx):",
        "    return TinyModel(ctx.vocab_size)",
        "",
        "def get_recipe(ctx):",
        "    return recipe(ctx)",
        "",
    ]
)

DEEP_MODEL = "\n".join(
    [
        "import torch",
        "from train import recipe",
        "",
        "class DeepResidual(torch.nn.Module):",
        "    def __init__(self, vocab_size):",
        "        super().__init__()",
        "        self.embedding = torch.nn.Embedding(vocab_size, 8)",
        "        self.hidden = torch.nn.Linear(8, 8)",
        "        self.activation = torch.nn.ReLU()",
        "        self.output = torch.nn.Linear(8, vocab_size)",
        "",
        "    def forward(self, tokens):",
        "        hidden = self.activation(self.hidden(self.embedding(tokens)))",
        "        return self.output(hidden)",
        "",
        "def build_model(ctx):",
        "    return DeepResidual(ctx.vocab_size)",
        "",
        "def get_recipe(ctx):",
        "    return recipe(ctx)",
        "",
    ]
)


def _zip_payload(model_code: str, *, suffix: str = "") -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "prism.yaml",
            "\n".join(
                [
                    "kind: architecture_only",
                    "architecture:",
                    "  entrypoint: src/model.py",
                    "training:",
                    "  entrypoint: src/train.py",
                ]
            )
            + "\n",
        )
        archive.writestr("src/model.py", model_code + suffix)
        archive.writestr(
            "src/train.py",
            "from prism_challenge.evaluator.interface import TrainingRecipe\n\n"
            "def recipe(ctx):\n"
            "    return TrainingRecipe(learning_rate=0.0003, batch_size=2)\n\n"
            "def train(ctx):\n"
            "    return None\n",
        )
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


def _settings(tmp_path, **overrides) -> PrismSettings:
    values = {
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'dedup.sqlite3'}",
        "shared_token": "secret",
        "allow_insecure_signatures": True,
        "docker_enabled": True,
        "docker_backend": "broker",
        "docker_broker_url": "http://platform-docker-broker:8082",
        "docker_broker_token": "secret",
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        "distributed_contract_policy": "off",
    }
    values.update(overrides)
    return PrismSettings(**values)


def _fake_run(self, spec, timeout_seconds):
    return DockerRunResult(
        container_name="prism-eval",
        stdout='PRISM_METRICS_JSON={"q_arch":0.8,"q_recipe":0.5}\n',
        stderr="",
        returncode=0,
    )


def _duplicate_report(client: TestClient, submission_id: str) -> dict[str, object]:
    async def read_report() -> dict[str, object]:
        async with client.app.state.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT report FROM plagiarism_reviews WHERE submission_id=?",
                (submission_id,),
            )
        assert rows
        report = loads(str(rows[0]["report"]))
        assert isinstance(report, dict)
        return report

    return anyio.run(read_report)


def _set_duplicate_thresholds(client: TestClient, value: dict[str, float]) -> None:
    async def store() -> None:
        await client.app.state.repository.store_runtime_config(
            config_key="duplicate_thresholds",
            value=value,
            updated_by="test",
        )

    anyio.run(store)


def test_exact_duplicate_no_owner_and_persists_report(tmp_path, monkeypatch):
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)

    with TestClient(create_app(_settings(tmp_path))) as client:
        payload = _zip_payload(TINY_MODEL)
        original = _submit_zip(client, "owner", "owner-1", payload)
        _process(client)

        duplicate = _submit_zip(client, "copycat", "copycat-1", payload)
        _process(client)

        status = client.get(f"/v1/submissions/{duplicate}").json()
        architectures = client.get("/v1/architectures").json()
        report = _duplicate_report(client, duplicate)

    assert status["status"] == "rejected"
    assert len(architectures) == 1
    assert architectures[0]["owner_submission_id"] == original
    assert architectures[0]["owner_hotkey"] == "owner"
    assert report["outcome"] == "reject"
    assert report["exact_source_hash"] is True
    assert report["thresholds"]["exact_source_similarity"] == 0.98
    assert report["evidence"][0]["snippet_hash"]


def test_identical_graph_changed_source_attaches_without_new_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)

    with TestClient(create_app(_settings(tmp_path))) as client:
        original = _submit_zip(client, "owner", "owner-1", _zip_payload(TINY_MODEL))
        _process(client)
        variant = _submit_zip(
            client,
            "variant",
            "variant-1",
            _zip_payload(TINY_MODEL, suffix="\n\n"),
        )
        _process(client)

        status = client.get(f"/v1/submissions/{variant}").json()
        architectures = client.get("/v1/architectures").json()
        report = _duplicate_report(client, variant)

    assert status["status"] == "completed"
    assert len(architectures) == 1
    assert architectures[0]["owner_submission_id"] == original
    assert architectures[0]["owner_hotkey"] == "owner"
    assert report["outcome"] == "attach"
    assert report["source_similarity"] >= 0.98


def test_borderline_similarity_quarantines_with_sql_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)

    with TestClient(create_app(_settings(tmp_path))) as client:
        _set_duplicate_thresholds(
            client,
            {
                "exact_source_similarity": 0.98,
                "quarantine_source_similarity": 0.10,
                "same_architecture_similarity": 0.10,
                "static_reject_similarity": 0.96,
            },
        )
        _submit_zip(client, "owner", "owner-1", _zip_payload(TINY_MODEL))
        _process(client)
        borderline = _submit_zip(client, "borderline", "borderline-1", _zip_payload(DEEP_MODEL))
        _process(client)

        status = client.get(f"/v1/submissions/{borderline}").json()
        holds = client.get(
            "/internal/v1/component-review/holds",
            headers={"Authorization": "Bearer secret"},
        ).json()
        report = _duplicate_report(client, borderline)

    assert status["status"] == "held"
    assert holds[0]["submission_id"] == borderline
    assert report["outcome"] == "quarantine"
    assert report["thresholds"]["quarantine_source_similarity"] == 0.10


def test_distinct_architecture_below_thresholds_creates_new_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)
    settings = _settings(
        tmp_path,
        component_agent_same_threshold=0.99,
        component_agent_hold_threshold=0.99,
    )

    with TestClient(create_app(settings)) as client:
        _set_duplicate_thresholds(
            client,
            {
                "exact_source_similarity": 0.999,
                "quarantine_source_similarity": 0.999,
                "same_architecture_similarity": 0.999,
                "static_reject_similarity": 0.999,
            },
        )
        first = _submit_zip(client, "owner-a", "owner-a-1", _zip_payload(TINY_MODEL))
        _process(client)
        second = _submit_zip(client, "owner-b", "owner-b-1", _zip_payload(DEEP_MODEL))
        _process(client)

        architectures = client.get("/v1/architectures").json()
        report = _duplicate_report(client, second)

    assert {item["owner_submission_id"] for item in architectures} == {first, second}
    assert {item["owner_hotkey"] for item in architectures} == {"owner-a", "owner-b"}
    assert report["outcome"] == "allow"
