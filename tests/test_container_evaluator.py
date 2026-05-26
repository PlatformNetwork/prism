from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    ContainerEvaluationError,
    PrismContainerEvaluator,
    _parse_metrics,
    _runner_launch_command,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.sdk.executors.docker import DockerRunResult


def _evaluator() -> PrismContainerEvaluator:
    return PrismContainerEvaluator(
        settings=PrismSettings(
            shared_token="secret",
            docker_backend="broker",
            docker_broker_url="http://broker",
            docker_broker_token="token",
        ),
        ctx=PrismContext(sequence_length=16),
    )


def test_runner_launch_command_uses_torchrun_for_single_gpu() -> None:
    assert _runner_launch_command(1) == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def test_runner_launch_command_uses_torchrun_for_two_gpus() -> None:
    assert _runner_launch_command(2) == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def test_runner_launch_command_allows_eight_gpus() -> None:
    assert _runner_launch_command(8) == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def test_runner_launch_command_rejects_more_than_eight_gpus() -> None:
    with pytest.raises(ContainerEvaluationError, match="maximum of 8"):
        _runner_launch_command(9)


def test_runner_launch_command_rejects_less_than_one_gpu() -> None:
    with pytest.raises(ContainerEvaluationError, match="at least 1"):
        _runner_launch_command(0)


@pytest.mark.parametrize("gpu_count", [None, "2", 2.0, True])
def test_runner_launch_command_rejects_invalid_values(gpu_count) -> None:
    with pytest.raises(ContainerEvaluationError, match="must be an integer"):
        _runner_launch_command(gpu_count)


def test_container_evaluator_uses_torchrun_for_multi_gpu(monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        captured["command"] = spec.command
        captured["env"] = spec.env
        return DockerRunResult(
            "container",
            'PRISM_METRICS_JSON={"q_arch":1.0,"q_recipe":0.5}\n',
            "",
            0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    evaluator = PrismContainerEvaluator(
        settings=PrismSettings(
            shared_token="secret",
            docker_backend="broker",
            docker_broker_url="http://broker",
            docker_broker_token="token",
            platform_eval_gpu_count=2,
        ),
        ctx=PrismContext(sequence_length=16),
    )

    evaluator.evaluate(
        submission_id="sub",
        code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
        code_hash="code",
        arch_hash="arch",
        backend="platform_gpu",
    )

    assert captured["command"] == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )
    assert captured["env"]["PRISM_GPU_COUNT"] == "2"
    assert captured["env"]["PRISM_DISTRIBUTED_BACKEND"] == "nccl"



def test_container_evaluator_payload_and_env_include_checkpoint_workspace(tmp_path, monkeypatch):
    captured = {}
    resume_dir = tmp_path / "resume-checkpoint"
    resume_dir.mkdir()

    def fake_run(self, spec, timeout_seconds):
        captured["env"] = spec.env
        captured["artifact_mount"] = spec.mounts[1]
        captured["mounts"] = spec.mounts
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        return DockerRunResult(
            "container",
            'PRISM_METRICS_JSON={"q_arch":1.0,"q_recipe":0.5}\n',
            "",
            0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)

    _evaluator().evaluate(
        submission_id="sub",
        code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
        code_hash="code",
        arch_hash="arch",
        backend="platform_gpu",
        attempt=2,
        resume_checkpoint_dir=resume_dir,
    )

    checkpoint = captured["payload"]["checkpoint_workspace"]
    assert captured["artifact_mount"].target == "/artifacts"
    assert captured["artifact_mount"].read_only is False
    resume_mount = next(
        mount for mount in captured["mounts"] if mount.target == "/resume-checkpoint"
    )
    assert resume_mount.source == resume_dir
    assert resume_mount.read_only is True
    assert checkpoint == {
        "api_version": 1,
        "attempt": 2,
        "is_resume": True,
        "artifact_root": "/artifacts",
        "current": {
            "artifact_relative_path": "checkpoints/sub/attempt-2/current",
            "path": "/artifacts/checkpoints/sub/attempt-2/current",
        },
        "resume": {
            "artifact_relative_path": "checkpoints/sub/attempt-1/current",
            "path": "/resume-checkpoint",
        },
    }
    assert captured["payload"]["context"]["checkpoint_dir"] == (
        "/artifacts/checkpoints/sub/attempt-2/current"
    )
    assert captured["payload"]["context"]["resume_checkpoint_dir"] == (
        "/resume-checkpoint"
    )
    assert captured["env"]["PRISM_CHECKPOINT_DIR"] == (
        "/artifacts/checkpoints/sub/attempt-2/current"
    )
    assert captured["env"]["PRISM_CHECKPOINT_ARTIFACT_PATH"] == (
        "checkpoints/sub/attempt-2/current"
    )
    assert captured["env"]["PRISM_RESUME_CHECKPOINT_DIR"] == (
        "/resume-checkpoint"
    )
    assert captured["env"]["PRISM_IS_RESUME"] == "1"


def test_embedded_runner_prism_context_matches_checkpoint_contract() -> None:
    assert "checkpoint_dir: Path | None = None" in _CONTAINER_EVAL_SCRIPT
    assert "resume_checkpoint_dir: Path | None = None" in _CONTAINER_EVAL_SCRIPT
    assert "checkpoint_api_version: int = 1" in _CONTAINER_EVAL_SCRIPT
    assert "world_size: int = 1" in _CONTAINER_EVAL_SCRIPT
    assert "checkpoint_metadata: dict = dataclasses.field(default_factory=dict)" in (
        _CONTAINER_EVAL_SCRIPT
    )


RUNNER_MODEL_WITH_SAVE = """
import torch


def build_model(ctx):
    return torch.nn.Sequential(
        torch.nn.Embedding(ctx.vocab_size, 4),
        torch.nn.Linear(4, ctx.vocab_size),
    )


def get_recipe(ctx):
    return {"batch_size": 1, "learning_rate": 0.0003}


def save_checkpoint(model, checkpoint_dir, ctx):
    path = checkpoint_dir / "rank0" / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"attempt": ctx.attempt, "world_size": ctx.world_size}, path)
    return {"path": "rank0/model.pt", "metadata": {"marker": "runner-save"}}
"""

RUNNER_MODEL_WITHOUT_SAVE = """
import torch


def build_model(ctx):
    return torch.nn.Sequential(
        torch.nn.Embedding(ctx.vocab_size, 4),
        torch.nn.Linear(4, ctx.vocab_size),
    )


def get_recipe(ctx):
    return {"batch_size": 1, "learning_rate": 0.0003}
"""


RUNNER_MODEL_DISTRIBUTED_WITH_SAVE = """
import torch


def build_model(ctx):
    return torch.nn.Sequential(
        torch.nn.Embedding(ctx.vocab_size, 4),
        torch.nn.Linear(4, ctx.vocab_size),
    )


def get_recipe(ctx):
    return {"batch_size": 1, "learning_rate": 0.0003}


def compute_loss(model, batch, ctx):
    return torch.tensor(float(ctx.rank + 1), device=ctx.device)


def train_step(model, batch, optimizer, ctx):
    return torch.tensor(float(ctx.rank + 1), device=ctx.device)


def save_checkpoint(model, checkpoint_dir, ctx):
    path = checkpoint_dir / "rank0" / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": type(model).__name__, "world_size": ctx.world_size}, path)
    return {
        "path": "rank0/model.pt",
        "metadata": {"model_type": type(model).__name__, "world_size": ctx.world_size},
    }
"""


def _runner_payload(artifacts, checkpoint_dir):
    return {
        "submission_id": "sub",
        "code_hash": "code-hash",
        "arch_hash": "arch-hash",
        "execution_mode": "gpu_proxy_eval",
        "entrypoint": "model.py",
        "context": {
            "vocab_size": 32,
            "sequence_length": 8,
            "max_parameters": 10000,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_api_version": 1,
            "attempt": 1,
            "is_resume": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "distributed_backend": None,
        },
        "checkpoint_workspace": {
            "api_version": 1,
            "attempt": 1,
            "is_resume": False,
            "artifact_root": str(artifacts),
            "current": {
                "artifact_relative_path": "checkpoints/sub/attempt-1/current",
                "path": str(checkpoint_dir),
            },
            "resume": None,
        },
        "gpu_allocation": {
            "actual_gpu_count": 0,
            "gpu_type": None,
            "target_server": None,
            "device_ids": [],
        },
        "artifact_output": {
            "path": str(artifacts),
            "manifest_path": str(artifacts / "prism_run_manifest.v1.json"),
        },
    }


def _run_embedded_runner(
    tmp_path, model_code, *, env_overrides=None, command=None, payload_mutator=None
):
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    artifacts = workspace / "artifacts"
    checkpoint_dir = artifacts / "checkpoints" / "sub" / "attempt-1" / "current"
    project.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    runner_path = workspace / "runner.py"
    payload_path = workspace / "payload.json"
    runner_path.write_text(_CONTAINER_EVAL_SCRIPT, encoding="utf-8")
    (project / "model.py").write_text(model_code, encoding="utf-8")
    payload = _runner_payload(artifacts, checkpoint_dir)
    if payload_mutator is not None:
        payload_mutator(payload, artifacts, checkpoint_dir)
    payload_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project)
    env["PRISM_PROJECT_ROOT"] = str(project)
    if env_overrides:
        env.update(env_overrides)
    if command is None:
        command = [sys.executable, str(runner_path), str(payload_path)]
    result = subprocess.run(
        command,
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result, artifacts, checkpoint_dir


def test_embedded_runner_writes_checkpoint_manifest_from_save_hook(tmp_path) -> None:
    result, artifacts, checkpoint_dir = _run_embedded_runner(
        tmp_path,
        RUNNER_MODEL_WITH_SAVE,
    )

    assert result.returncode == 0, result.stderr
    assert "PRISM_METRICS_JSON=" in result.stdout
    metadata_path = checkpoint_dir / "rank0" / "checkpoint_metadata.v1.json"
    manifest_path = artifacts / "prism_run_manifest.v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = manifest["artifacts"]["checkpoints"][0]
    assert metadata["checkpoint_path"] == "checkpoints/sub/attempt-1/current/rank0/model.pt"
    assert metadata["hook_return"] == {
        "path": "rank0/model.pt",
        "metadata": {"marker": "runner-save"},
    }
    assert manifest["compute"]["checkpoint_path"] == metadata["checkpoint_path"]
    assert checkpoint == {
        "path": "checkpoints/sub/attempt-1/current/rank0/model.pt",
        "metadata_path": "checkpoints/sub/attempt-1/current/rank0/checkpoint_metadata.v1.json",
        "bytes": metadata["bytes_total"],
        "attempt": 1,
        "world_size": 1,
        "rank_writer": 0,
        "created_at": metadata["created_at"],
    }


def test_embedded_runner_without_save_hook_writes_manifest_without_checkpoint(tmp_path) -> None:
    result, artifacts, _checkpoint_dir = _run_embedded_runner(
        tmp_path,
        RUNNER_MODEL_WITHOUT_SAVE,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((artifacts / "prism_run_manifest.v1.json").read_text())
    assert manifest["compute"]["checkpoint_path"] is None
    assert manifest["artifacts"]["checkpoints"] == []

def test_embedded_runner_resume_requires_load_checkpoint_hook(tmp_path) -> None:
    def request_resume(payload, artifacts, resume_dir):
        current_dir = artifacts / "checkpoints" / "sub" / "attempt-2" / "current"
        current_dir.mkdir(parents=True)
        checkpoint_path = resume_dir / "model.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        recipe_payload = {
            "learning_rate": 0.0003,
            "batch_size": 1,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 0.01,
        }
        recipe_fingerprint = sha256(
            json.dumps(recipe_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        metadata = {
            "checkpoint_api_version": 1,
            "submission_id": "sub",
            "attempt": 1,
            "code_hash": "code-hash",
            "arch_hash": "arch-hash",
            "recipe_fingerprint": recipe_fingerprint,
            "created_at": "2026-05-25T00:00:00Z",
            "checkpoint_path": "checkpoints/sub/attempt-1/current/model.pt",
            "hook_return": {"path": "model.pt"},
            "world_size": 1,
            "rank_writer": 0,
            "checkpoint_dir": "checkpoints/sub/attempt-1/current",
            "bytes_total": checkpoint_path.stat().st_size,
        }
        (resume_dir / "checkpoint_metadata.v1.json").write_text(
            json.dumps(metadata, separators=(",", ":")),
            encoding="utf-8",
        )
        payload["context"].update(
            {
                "checkpoint_dir": str(current_dir),
                "resume_checkpoint_dir": str(resume_dir),
                "attempt": 2,
                "is_resume": True,
            }
        )
        payload["checkpoint_workspace"].update(
            {
                "attempt": 2,
                "is_resume": True,
                "current": {
                    "artifact_relative_path": "checkpoints/sub/attempt-2/current",
                    "path": str(current_dir),
                },
                "resume": {
                    "artifact_relative_path": "checkpoints/sub/attempt-1/current",
                    "path": str(resume_dir),
                },
            }
        )

    result, artifacts, _resume_dir = _run_embedded_runner(
        tmp_path,
        RUNNER_MODEL_WITHOUT_SAVE,
        payload_mutator=request_resume,
    )

    assert result.returncode != 0
    assert "resume checkpoint requested" in result.stderr
    assert "load_checkpoint hook is absent" in result.stderr
    assert "PRISM_METRICS_JSON=" not in result.stdout
    assert not (artifacts / "prism_run_manifest.v1.json").exists()


def test_embedded_runner_nonzero_rank_does_not_write_shared_artifacts(tmp_path) -> None:
    result, artifacts, checkpoint_dir = _run_embedded_runner(
        tmp_path,
        RUNNER_MODEL_WITH_SAVE,
        env_overrides={"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "PRISM_METRICS_JSON=" not in result.stdout
    assert not (artifacts / "prism_run_manifest.v1.json").exists()
    assert not (checkpoint_dir / "rank0" / "checkpoint_metadata.v1.json").exists()


def test_embedded_runner_torchrun_reduces_loss_and_saves_unwrapped_model(tmp_path) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "runner.py",
        "payload.json",
    ]
    result, artifacts, checkpoint_dir = _run_embedded_runner(
        tmp_path,
        RUNNER_MODEL_DISTRIBUTED_WITH_SAVE,
        env_overrides={"PRISM_DISTRIBUTED_BACKEND": "gloo"},
        command=command,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PRISM_METRICS_JSON=") == 1
    manifest = json.loads((artifacts / "prism_run_manifest.v1.json").read_text())
    metadata = json.loads(
        (checkpoint_dir / "rank0" / "checkpoint_metadata.v1.json").read_text()
    )
    assert manifest["compute"]["world_size"] == 2
    assert manifest["compute"]["rank"] == 0
    assert manifest["compute"]["local_rank"] == 0
    assert manifest["compute"]["distributed_backend"] == "gloo"
    assert manifest["metrics"]["final_loss"] == pytest.approx(1.5)
    assert metadata["hook_return"]["metadata"] == {
        "model_type": "Sequential",
        "world_size": 2,
    }
    assert metadata["world_size"] == 2


def test_container_evaluator_reports_timeout(monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult("container", "", "", 124, timed_out=True)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        _evaluator().evaluate(
            submission_id="sub",
            code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
            code_hash="code",
            arch_hash="arch",
            backend="platform_gpu",
        )


def test_container_evaluator_payload_declares_contract(monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        captured["payload"] = (spec.mounts[0].source / "payload.json").read_text()
        return DockerRunResult(
            "container",
            'PRISM_METRICS_JSON={"q_arch":1.2,"final_loss":2.0,"val_loss":3.0}\n',
            "",
            0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    result = _evaluator().evaluate(
        submission_id="sub",
        code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
        code_hash="code",
        arch_hash="arch",
        backend="platform_gpu",
    )

    assert '"build_model"' in captured["payload"]
    assert '"inference_logits"' in captured["payload"]
    assert result.metrics["q_arch"] == 1.0
    assert result.metrics["q_recipe"] == 0.5
    assert result.metrics["train_loss"] == 2.0
    assert result.metrics["eval_loss"] == 3.0


def test_container_evaluator_reports_nonzero_exit(monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult("container", "stdout", "stderr", 2)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    with pytest.raises(RuntimeError, match="stderr"):
        _evaluator().evaluate(
            submission_id="sub",
            code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
            code_hash="code",
            arch_hash="arch",
            backend="platform_gpu",
        )


def test_parse_metrics_rejects_invalid_output() -> None:
    with pytest.raises(RuntimeError, match="invalid metrics"):
        _parse_metrics("PRISM_METRICS_JSON=[]")
    with pytest.raises(RuntimeError, match="q_arch"):
        _parse_metrics('PRISM_METRICS_JSON={"q_recipe":0.9}')
    with pytest.raises(RuntimeError, match="did not return metrics"):
        _parse_metrics("no metrics here")


def test_parse_metrics_preserves_hook_usage_metrics() -> None:
    metrics = _parse_metrics(
        'PRISM_METRICS_JSON={"q_arch":0.5,'
        '"hook.configure_optimizer.used":1,'
        '"hook.inference_logits.used":1,'
        '"hook.compute_loss.used":1,'
        '"hook.train_step.used":1}\n'
    )
    assert metrics["hook.configure_optimizer.used"] == 1.0
    assert metrics["hook.inference_logits.used"] == 1.0
    assert metrics["hook.compute_loss.used"] == 1.0
    assert metrics["hook.train_step.used"] == 1.0
