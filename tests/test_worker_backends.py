from __future__ import annotations

import json

from conftest import signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.sdk.executors.docker import DockerRunResult

REMOTE_ONLY_CODE = """
def build_model(ctx):
    raise RuntimeError('container must not execute submitted model code in remote mode')

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""


def _submit(client: TestClient, code: str, nonce: str = "remote1") -> str:
    payload = {"code": code, "filename": "model.py"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_removed_legacy_remote_provider_backend_is_rejected(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'remote.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="remote_provider",
    )
    try:
        create_app(settings)
    except ValueError as exc:
        assert "Unsupported execution backend" in str(exc)
    else:
        raise AssertionError("remote_provider must not be supported")


def test_platform_gpu_worker_runs_submission_in_container(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["spec"] = spec
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return DockerRunResult(
            container_name="prism-eval",
            stdout=(
                'PRISM_METRICS_JSON={"q_arch":0.88,"q_recipe":0.66,"penalty":0.0,'
                '"train_loss":1.2,"eval_loss":1.4,"inference_latency_ms":7.0}\n'
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'platform-gpu.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        platform_eval_cpus=3.0,
        platform_eval_memory="12g",
        platform_eval_gpu_count=2,
        platform_eval_gpu_type="l4",
        plagiarism_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="platform-gpu")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] == 0.88

    spec = captured["spec"]
    assert spec.image == "ghcr.io/platformnetwork/prism-evaluator:latest"
    assert spec.labels["platform.job"] == submission_id
    assert spec.labels["platform.task"] == "architecture"
    assert spec.env["PRISM_EXECUTION_BACKEND"] == "platform_gpu"
    assert spec.env["PRISM_GPU_COUNT"] == "2"
    assert spec.env["PRISM_GPU_TYPE"] == "l4"
    assert spec.limits.cpus == 3.0
    assert spec.limits.memory == "12g"
    assert captured["payload"]["code"] == REMOTE_ONLY_CODE
    assert captured["timeout_seconds"] == 900


def test_platform_gpu_rejects_sandbox_violations_before_container(tmp_path, monkeypatch):
    def fail_run(self, spec, timeout_seconds):
        raise AssertionError("container should not run")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)
    code = """
import os

def build_model(ctx):
    return None

def get_recipe(ctx):
    return {}
"""
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'platform-gpu-reject.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, code, nonce="platform-gpu-reject")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "rejected"
        assert "forbidden imports: os" in status["error"]


def test_platform_gpu_version_advertises_docker_executor(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'platform-gpu-version.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
    )
    with TestClient(create_app(settings)) as client:
        version = client.get("/version").json()
    assert "docker_executor" in version["capabilities"]
    assert "lium" not in version["capabilities"]


def test_removed_legacy_local_cpu_backend_is_rejected(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'local.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="local_cpu",
        sequence_length=16,
    )
    try:
        create_app(settings)
    except ValueError as exc:
        assert "Unsupported execution backend" in str(exc)
    else:
        raise AssertionError("local_cpu must not be supported")


CUSTOM_HOOK_CODE = """
import torch
import torch.nn.functional as F
from prism_challenge.evaluator.interface import TrainingRecipe

class HookModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 6)
        self.proj = torch.nn.Linear(6, vocab_size)

    def forward(self, tokens):
        return self.proj(self.emb(tokens))

def build_model(ctx):
    return HookModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=0.001, batch_size=1)

def configure_optimizer(model, recipe, ctx):
    return torch.optim.SGD(model.parameters(), lr=recipe.learning_rate)

def inference_logits(model, batch, ctx):
    return model(batch.tokens)

def compute_loss(model, batch, ctx):
    logits = inference_logits(model, batch, ctx)
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), batch.targets.reshape(-1) % vocab)

def train_step(model, batch, optimizer, ctx):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch, ctx)
    loss.backward()
    optimizer.step()
    return loss
"""


def test_platform_gpu_accepts_custom_training_and_inference_hooks(tmp_path, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"q_arch":0.7,"q_recipe":0.8}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'hooks.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, CUSTOM_HOOK_CODE, nonce="custom-hooks")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] is not None
