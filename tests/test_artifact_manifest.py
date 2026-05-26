from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from prism_challenge.evaluator.schemas import (
    RUN_MANIFEST_FILENAME,
    ExecutionMode,
    PrismRunManifest,
    validate_run_manifest_for_official_scoring,
)

GRAPH_HASH = "a" * 64
LOG_HASH = "b" * 64
METADATA_HASH = "c" * 64
METRICS_HASH = "d" * 64
CREATED_AT = "2026-05-25T00:00:00Z"


def _valid_diagnostics() -> dict:
    return {
        "activation_entropy": {
            "status": "ok",
            "aggregate": 0.75,
            "per_layer": {"layer_0": 0.75},
            "warnings": [],
        },
        "useful_sparsity": {
            "status": "ok",
            "aggregate": 0.25,
            "per_layer": {"layer_0": 0.25},
            "warnings": [],
        },
        "attention_diversity": {
            "status": "not_applicable",
            "aggregate": None,
            "per_layer": {},
            "warnings": [],
            "not_applicable_reason": "architecture exposes no attention weights",
            "redistribution": {
                "enabled": True,
                "policy_key": "loss_comparability_policy.redistribution_policy",
                "target": "diagnostics_health",
                "reason": "architecture exposes no attention weights",
            },
        },
        "representation_collapse": {
            "status": "ok",
            "aggregate": 0.1,
            "per_layer": {"layer_0": 0.1},
            "warnings": [],
        },
        "gradient_noise_scale": {
            "status": "ok",
            "aggregate": 0.2,
            "per_layer": {"all": 0.2},
            "warnings": [],
        },
        "activation_norm_stability": {
            "status": "ok",
            "aggregate": 0.9,
            "per_layer": {"layer_0": 0.9},
            "warnings": [],
        },
        "neuron_specialization": {
            "status": "ok",
            "aggregate": 0.3,
            "per_layer": {"layer_0": 0.3},
            "warnings": [],
        },
    }


def _valid_manifest(mode: str) -> dict:
    return {
        "schema_version": "prism_run_manifest.v1",
        "submission_id": "submission-1",
        "architecture_id": "architecture-1",
        "architecture_version_id": "architecture-version-1",
        "training_script_version_id": "training-script-1",
        "run_id": f"run-{mode}",
        "mode": mode,
        "dataset": {
            "name": "fineweb-edu",
            "revision": "2026-05-fixture",
            "train_split_fingerprint": "train-fixture",
            "validation_split_fingerprint": "validation-fixture",
            "test_split_fingerprint": "test-fixture",
            "tokenizer_fingerprint": "tokenizer-fixture",
            "evaluator_fingerprint": "evaluator-fixture",
            "benchmark_fingerprints": {"mmlu": "mmlu-fixture"},
            "contamination_report_path": "artifacts/contamination.json",
        },
        "model": {
            "parameter_count": 1024,
            "architecture_graph_hash": GRAPH_HASH,
            "tokenizer_kind": "fixed_prism_fixture",
            "vocab_size": 4096,
            "max_sequence_length": 128,
        },
        "compute": {
            "gpu_count": 0 if mode == ExecutionMode.LOCAL_CPU_SMOKE.value else 1,
            "gpu_type": None if mode == ExecutionMode.LOCAL_CPU_SMOKE.value else "fixture-gpu",
            "gpu_server": None if mode == ExecutionMode.LOCAL_CPU_SMOKE.value else "gpu-host-1",
            "gpu_device_ids": [] if mode == ExecutionMode.LOCAL_CPU_SMOKE.value else ["0"],
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "distributed_backend": None,
            "effective_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "tokens_seen": 128,
            "estimated_flops": 2048.0,
            "wall_clock_seconds": 1.25,
            "checkpoint_path": None,
            "resume_checkpoint_path": None,
        },
        "metrics": {
            "loss_vs_tokens": [{"x": 0.0, "loss": 3.0}, {"x": 128.0, "loss": 2.5}],
            "loss_vs_compute": [{"x": 0.0, "loss": 3.0}, {"x": 2048.0, "loss": 2.5}],
            "loss_vs_params": [{"x": 1024.0, "loss": 2.5}],
            "learning_speed_slope": -0.1,
            "tokens_seen": 128,
            "estimated_flops": 2048.0,
            "parameter_count": 1024,
            "benchmark_scores": {"mmlu": 0.25},
            "diagnostics": _valid_diagnostics(),
            "gpu_count": 0 if mode == ExecutionMode.LOCAL_CPU_SMOKE.value else 1,
            "loss": {
                "raw_final_loss": 2.5,
                "standardized_eval_loss": 2.4,
                "loss_normalization_scope": "fixed_tokenizer",
                "baseline_run_id": "baseline-run-1",
                "relative_loss_reduction": 0.2,
                "architecture_normalized_heldout_improvement": 0.1,
                "loss_comparable": True,
                "loss_component_redistribution": {"enabled": False},
            },
            "final_loss": 2.5,
        },
        "artifacts": {
            "architecture_graph": {
                "path": "artifacts/architecture_graph.json",
                "sha256": GRAPH_HASH,
                "content_type": "application/json",
                "bytes": 512,
            },
            "architecture_metadata": {
                "path": "artifacts/architecture_metadata.v1.json",
                "sha256": METADATA_HASH,
                "content_type": "application/json",
                "bytes": 1024,
            },
            "run_log": {
                "path": "artifacts/run.log",
                "sha256": LOG_HASH,
                "content_type": "text/plain",
                "bytes": 2048,
            },
            "checkpoints": [],
            "metrics": {
                "path": "artifacts/metrics.json",
                "sha256": METRICS_HASH,
                "content_type": "application/json",
                "bytes": 1024,
            },
        },
        "validation": {
            "passed": True,
            "score_eligible": mode != ExecutionMode.LOCAL_CPU_SMOKE.value,
            "deterministic_evidence": [],
            "warnings": [],
            "errors": [],
        },
    }


def _checkpoint_entry(
    path: str = "checkpoints/submission-1/attempt-1/current/checkpoint.pt",
    metadata_path: str = (
        "checkpoints/submission-1/attempt-1/current/checkpoint_metadata.v1.json"
    ),
) -> dict:
    return {
        "path": path,
        "metadata_path": metadata_path,
        "bytes": 4096,
        "attempt": 1,
        "world_size": 1,
        "rank_writer": 0,
        "created_at": CREATED_AT,
    }


@pytest.mark.parametrize(
    "mode",
    [
        ExecutionMode.LOCAL_CPU_SMOKE.value,
        ExecutionMode.GPU_PROXY_EVAL.value,
        ExecutionMode.FULL_SCALE_EVAL.value,
    ],
)
def test_valid_manifest_fixtures_pass_validation(mode: str) -> None:
    manifest = PrismRunManifest.model_validate(_valid_manifest(mode))

    assert RUN_MANIFEST_FILENAME == "prism_run_manifest.v1.json"
    assert manifest.mode == ExecutionMode(mode)
    assert manifest.metrics.loss_comparable is True


def test_valid_checkpoint_manifest_fields_parse() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    checkpoint = _checkpoint_entry()
    payload["compute"]["checkpoint_path"] = checkpoint["path"]
    payload["compute"]["resume_checkpoint_path"] = (
        "checkpoints/submission-1/attempt-0/current/checkpoint.pt"
    )
    payload["artifacts"]["checkpoints"] = [checkpoint]

    manifest = PrismRunManifest.model_validate(payload)

    assert manifest.compute.checkpoint_path == checkpoint["path"]
    assert manifest.compute.resume_checkpoint_path == (
        "checkpoints/submission-1/attempt-0/current/checkpoint.pt"
    )
    assert manifest.artifacts.checkpoints[0].metadata_path == checkpoint["metadata_path"]
    assert manifest.artifacts.checkpoints[0].bytes == 4096
    assert manifest.artifacts.checkpoints[0].rank_writer == 0


def test_checkpoint_fields_are_backward_compatible_when_absent() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    payload["compute"].pop("checkpoint_path")
    payload["compute"].pop("resume_checkpoint_path")
    payload["artifacts"].pop("checkpoints")

    manifest = PrismRunManifest.model_validate(payload)

    assert manifest.compute.checkpoint_path is None
    assert manifest.compute.resume_checkpoint_path is None
    assert manifest.artifacts.checkpoints == []


def test_compute_checkpoint_path_requires_checkpoint_artifact_entry() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    payload["compute"]["checkpoint_path"] = (
        "checkpoints/submission-1/attempt-1/current/checkpoint.pt"
    )

    with pytest.raises(ValidationError, match="checkpoint_path"):
        PrismRunManifest.model_validate(payload)


def test_artifacts_checkpoints_use_exact_v1_shape() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    checkpoint = _checkpoint_entry()
    checkpoint["sha256"] = GRAPH_HASH
    payload["compute"]["checkpoint_path"] = checkpoint["path"]
    payload["artifacts"]["checkpoints"] = [checkpoint]

    with pytest.raises(ValidationError):
        PrismRunManifest.model_validate(payload)


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/artifacts/checkpoints/submission-1/attempt-1/current/checkpoint.pt",
        "/tmp/prism/checkpoint.pt",
        "../checkpoint.pt",
        "checkpoints/submission-1/../checkpoint.pt",
        "C:/Users/prism/checkpoint.pt",
        "checkpoints\\submission-1\\checkpoint.pt",
    ],
)
def test_compute_checkpoint_paths_reject_unsafe_host_or_container_paths(
    checkpoint_path: str,
) -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    payload["compute"]["checkpoint_path"] = checkpoint_path

    with pytest.raises(ValidationError):
        PrismRunManifest.model_validate(payload)


def test_compute_resume_checkpoint_path_rejects_container_absolute_path() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    payload["compute"]["resume_checkpoint_path"] = (
        "/artifacts/checkpoints/submission-1/attempt-1/current/checkpoint.pt"
    )

    with pytest.raises(ValidationError):
        PrismRunManifest.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["path", "metadata_path"],
)
def test_artifact_checkpoint_paths_reject_host_absolute_paths(field: str) -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))
    checkpoint = _checkpoint_entry()
    checkpoint[field] = "/artifacts/checkpoints/submission-1/attempt-1/current/checkpoint.pt"
    payload["artifacts"]["checkpoints"] = [checkpoint]

    with pytest.raises(ValidationError):
        PrismRunManifest.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("metrics", "loss_vs_tokens"),
        ("metrics", "learning_speed_slope"),
        ("metrics", "parameter_count"),
        ("metrics", "gpu_count"),
        ("schema_version",),
        ("metrics", "loss", "loss_comparable"),
    ],
)
def test_missing_required_manifest_fields_fail_validation(path: tuple[str, ...]) -> None:
    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target.pop(path[-1])

    with pytest.raises(ValidationError):
        PrismRunManifest.model_validate(payload)


def test_raw_final_loss_alone_is_not_official_scoring_validation() -> None:
    payload = _valid_manifest(ExecutionMode.FULL_SCALE_EVAL.value)
    payload["metrics"] = {"final_loss": 1.23}

    with pytest.raises(ValidationError):
        validate_run_manifest_for_official_scoring(payload)


def test_non_comparable_loss_blocks_official_scoring() -> None:
    payload = _valid_manifest(ExecutionMode.FULL_SCALE_EVAL.value)
    payload["metrics"]["loss"]["loss_comparable"] = False
    payload["metrics"]["loss"]["loss_component_redistribution"] = {
        "enabled": True,
        "target_track": "non_comparable",
        "reason": "standardized evaluator is not meaningful for this architecture",
    }

    manifest = PrismRunManifest.model_validate(payload)
    with pytest.raises(ValueError, match="loss_comparable"):
        manifest.require_official_scoring_ready()


def test_official_scoring_validation_accepts_complete_gpu_manifest() -> None:
    payload = deepcopy(_valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value))

    manifest = validate_run_manifest_for_official_scoring(payload)

    assert manifest.validation.score_eligible is True
