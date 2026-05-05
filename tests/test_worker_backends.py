from __future__ import annotations

import json

from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings

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


def test_remote_provider_worker_does_not_execute_submission_code(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'remote.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="remote_provider",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE)
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] == 0.42
        weights = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Platform-Challenge-Slug": "prism"},
        ).json()["weights"]
        assert weights == {"hk": 1.0}


def test_local_cpu_worker_executes_trainable_model(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'local.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="local_cpu",
        sequence_length=16,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, VALID_CODE, nonce="local1")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] >= 0


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


def test_local_cpu_accepts_custom_training_and_inference_hooks(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'hooks.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="local_cpu",
        sequence_length=16,
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
