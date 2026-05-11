from __future__ import annotations

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import PrismContainerEvaluator, _parse_metrics
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
