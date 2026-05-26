from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prism_challenge.evaluator.metrics import REQUIRED_DIAGNOSTICS

RUN_MANIFEST_FILENAME = "prism_run_manifest.v1.json"
RUN_MANIFEST_SCHEMA_VERSION = "prism_run_manifest.v1"
ARCHITECTURE_GRAPH_FILENAME: Final[Literal["architecture_graph.json"]] = "architecture_graph.json"
ARCHITECTURE_METADATA_FILENAME = "architecture_metadata.v1.json"
ARCHITECTURE_METADATA_SCHEMA_VERSION = "architecture_metadata.v1"


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutionMode(StrEnum):
    LOCAL_CPU_SMOKE = "local_cpu_smoke"
    GPU_PROXY_EVAL = "gpu_proxy_eval"
    FULL_SCALE_EVAL = "full_scale_eval"


class DeterministicEvidence(SchemaModel):
    rule_id: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    line: int | None = Field(default=None, ge=1)
    ast_node: str | None = Field(default=None, min_length=1)
    snippet_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_location(self) -> DeterministicEvidence:
        if self.line is None and self.ast_node is None:
            raise ValueError("deterministic evidence requires line or ast_node")
        return self


class HashMetadata(SchemaModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    canonicalization: str = Field(default="json-sort-keys-no-whitespace", min_length=1)


class ArchitectureGraph(SchemaModel):
    schema_version: str = Field(default="architecture_graph.v1", min_length=1)
    modules: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    parameterized_blocks: list[dict[str, Any]] = Field(default_factory=list)
    tokenizer_constraints: dict[str, Any] = Field(default_factory=dict)
    dynamic_routing: dict[str, Any] = Field(default_factory=dict)
    interface: dict[str, Any] = Field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ArchitectureIdentity(SchemaModel):
    architecture_id: str = Field(min_length=1)
    architecture_version_id: str = Field(min_length=1)
    architecture_graph_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    architecture_source_hash: str | None = Field(default=None, min_length=64, max_length=64)
    evaluation_config_id: str = Field(min_length=1)
    owner_hotkey: str | None = Field(default=None, min_length=1)


class ArchitectureGraphMetadata(SchemaModel):
    canonical_artifact: Literal["architecture_graph.json"] = ARCHITECTURE_GRAPH_FILENAME
    hash: HashMetadata
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    source_free_comparison_keys: list[str] = Field(min_length=1)


class ArchitectureMetadata(SchemaModel):
    schema_version: Literal["architecture_metadata.v1"]
    identity: ArchitectureIdentity
    graph: ArchitectureGraphMetadata
    derived_mermaid_path: str | None = Field(default=None, min_length=1)
    architecture_summary: str = Field(min_length=1)
    training_summary: str | None = Field(default=None, min_length=1)
    overview: dict[str, Any] = Field(default_factory=dict)
    difficulty: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    comparison_tags: list[str] = Field(default_factory=list)
    deterministic_evidence: list[DeterministicEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_graph_hash_match(self) -> ArchitectureMetadata:
        if self.identity.architecture_graph_hash != self.graph.hash.value:
            raise ValueError("identity architecture_graph_hash must match graph hash metadata")
        return self


class DatasetManifest(SchemaModel):
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    train_split_fingerprint: str = Field(min_length=1)
    validation_split_fingerprint: str = Field(min_length=1)
    test_split_fingerprint: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(min_length=1)
    evaluator_fingerprint: str = Field(min_length=1)
    benchmark_fingerprints: dict[str, str] = Field(default_factory=dict)
    contamination_report_path: str | None = Field(default=None, min_length=1)


class ModelManifest(SchemaModel):
    parameter_count: int = Field(gt=0)
    architecture_graph_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    tokenizer_kind: str = Field(min_length=1)
    vocab_size: int | None = Field(default=None, gt=0)
    max_sequence_length: int = Field(gt=0)


class ComputeManifest(SchemaModel):
    gpu_count: int = Field(ge=0)
    gpu_type: str | None = Field(default=None, min_length=1)
    gpu_server: str | None = Field(default=None, min_length=1)
    gpu_device_ids: list[str] = Field(default_factory=list)
    world_size: int = Field(ge=1)
    rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    distributed_backend: str | None = Field(default=None, min_length=1)
    effective_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(ge=1)
    tokens_seen: int = Field(ge=0)
    estimated_flops: float = Field(ge=0.0)
    wall_clock_seconds: float = Field(ge=0.0)
    checkpoint_path: str | None = Field(default=None, min_length=1)
    resume_checkpoint_path: str | None = Field(default=None, min_length=1)

    @field_validator("checkpoint_path", "resume_checkpoint_path")
    @classmethod
    def validate_checkpoint_artifact_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _artifact_relative_posix_path(value)


class MetricPoint(SchemaModel):
    x: float = Field(ge=0.0)
    loss: float = Field(ge=0.0)


class DiagnosticRedistributionMetadata(SchemaModel):
    enabled: bool
    policy_key: str | None = Field(default=None, min_length=1)
    target: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_sql_policy_metadata_when_enabled(self) -> DiagnosticRedistributionMetadata:
        if self.enabled and (not self.policy_key or not self.target or not self.reason):
            raise ValueError(
                "diagnostic redistribution requires policy_key, target, and reason"
            )
        return self


class DiagnosticRecord(SchemaModel):
    status: Literal["ok", "warning", "not_applicable"]
    aggregate: float | None = None
    per_layer: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    not_applicable_reason: str | None = Field(default=None, min_length=1)
    redistribution: DiagnosticRedistributionMetadata | None = None

    @model_validator(mode="after")
    def require_valid_diagnostic_payload(self) -> DiagnosticRecord:
        if self.status == "not_applicable":
            if self.aggregate is not None:
                raise ValueError("not_applicable diagnostics must not set aggregate")
            if not self.not_applicable_reason:
                raise ValueError("not_applicable diagnostics require a reason")
            if self.redistribution is None or not self.redistribution.enabled:
                raise ValueError(
                    "not_applicable diagnostics require enabled redistribution metadata"
                )
            return self

        if self.aggregate is None or not math.isfinite(self.aggregate):
            raise ValueError("diagnostic aggregate must be finite")
        for layer, value in self.per_layer.items():
            if not math.isfinite(value):
                raise ValueError(f"diagnostic per_layer value for {layer} must be finite")
        if self.status == "warning" and not self.warnings:
            raise ValueError("warning diagnostics require at least one warning")
        if self.redistribution is not None and self.redistribution.enabled:
            raise ValueError("applicable diagnostics must not enable redistribution")
        return self


class LossComponentRedistribution(SchemaModel):
    enabled: bool
    target_track: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_reason_when_enabled(self) -> LossComponentRedistribution:
        if self.enabled and not self.reason:
            raise ValueError("loss redistribution requires reason when enabled")
        return self


class LossComparabilityMetadata(SchemaModel):
    raw_final_loss: float = Field(ge=0.0)
    standardized_eval_loss: float = Field(ge=0.0)
    loss_normalization_scope: Literal[
        "fixed_tokenizer",
        "byte_normalized",
        "architecture_baseline",
    ]
    baseline_run_id: str = Field(min_length=1)
    relative_loss_reduction: float
    architecture_normalized_heldout_improvement: float
    loss_comparable: bool
    loss_component_redistribution: LossComponentRedistribution

    @model_validator(mode="after")
    def require_comparable_loss_without_redistribution(self) -> LossComparabilityMetadata:
        if self.loss_comparable and self.loss_component_redistribution.enabled:
            raise ValueError("comparable loss must not redistribute the loss component")
        return self


class RunMetrics(SchemaModel):
    loss_vs_tokens: list[MetricPoint] = Field(min_length=1)
    loss_vs_compute: list[MetricPoint] = Field(min_length=1)
    loss_vs_params: list[MetricPoint] = Field(min_length=1)
    learning_speed_slope: float
    tokens_seen: int = Field(ge=0)
    estimated_flops: float = Field(ge=0.0)
    parameter_count: int = Field(gt=0)
    benchmark_scores: dict[str, float]
    benchmark_capability_metadata: dict[str, Any] = Field(default_factory=dict)
    benchmark_noise_metadata: dict[str, Any] = Field(default_factory=dict)
    benchmark_contamination_metadata: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, DiagnosticRecord] = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    loss: LossComparabilityMetadata
    final_loss: float | None = Field(default=None, ge=0.0)

    @property
    def loss_comparable(self) -> bool:
        return self.loss.loss_comparable

    @model_validator(mode="after")
    def require_complete_diagnostics(self) -> RunMetrics:
        missing = sorted(set(REQUIRED_DIAGNOSTICS) - set(self.diagnostics))
        if missing:
            raise ValueError(f"run metrics missing required diagnostics: {missing}")
        return self


class ArtifactReference(SchemaModel):
    path: str = Field(min_length=1)
    sha256: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"
    )
    content_type: str | None = Field(default=None, min_length=1)
    bytes: int | None = Field(default=None, ge=0)


class CheckpointArtifact(SchemaModel):
    path: str = Field(min_length=1)
    metadata_path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    attempt: int = Field(ge=1)
    world_size: int = Field(ge=1)
    rank_writer: int = Field(ge=0)
    created_at: str = Field(min_length=1)

    @field_validator("path", "metadata_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _artifact_relative_posix_path(value)


class ArtifactManifest(SchemaModel):
    architecture_graph: ArtifactReference
    architecture_metadata: ArtifactReference
    run_log: ArtifactReference
    checkpoints: list[CheckpointArtifact] = Field(default_factory=list)
    metrics: ArtifactReference | None = None


class ValidationManifest(SchemaModel):
    passed: bool
    score_eligible: bool
    deterministic_evidence: list[DeterministicEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PrismRunManifest(SchemaModel):
    schema_version: Literal["prism_run_manifest.v1"]
    submission_id: str = Field(min_length=1)
    architecture_id: str = Field(min_length=1)
    architecture_version_id: str = Field(min_length=1)
    training_script_version_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    mode: ExecutionMode
    dataset: DatasetManifest
    model: ModelManifest
    compute: ComputeManifest
    metrics: RunMetrics
    artifacts: ArtifactManifest
    validation: ValidationManifest

    @model_validator(mode="after")
    def require_consistent_counts_and_hashes(self) -> PrismRunManifest:
        if self.model.parameter_count != self.metrics.parameter_count:
            raise ValueError("model parameter_count must match metrics parameter_count")
        if self.compute.gpu_count != self.metrics.gpu_count:
            raise ValueError("compute gpu_count must match metrics gpu_count")
        if self.compute.tokens_seen != self.metrics.tokens_seen:
            raise ValueError("compute tokens_seen must match metrics tokens_seen")
        if self.compute.estimated_flops != self.metrics.estimated_flops:
            raise ValueError("compute estimated_flops must match metrics estimated_flops")
        if self.model.architecture_graph_hash != self.artifacts.architecture_graph.sha256:
            raise ValueError("architecture graph artifact hash must match model graph hash")
        if self.compute.checkpoint_path is not None:
            checkpoint_paths = {checkpoint.path for checkpoint in self.artifacts.checkpoints}
            if self.compute.checkpoint_path not in checkpoint_paths:
                raise ValueError(
                    "compute checkpoint_path must reference an artifacts.checkpoints path"
                )
        return self

    def require_official_scoring_ready(self) -> PrismRunManifest:
        if self.mode is ExecutionMode.LOCAL_CPU_SMOKE:
            raise ValueError("local_cpu_smoke manifests are not official score eligible")
        if not self.validation.passed or not self.validation.score_eligible:
            raise ValueError("manifest validation must pass and be score eligible")
        if not self.metrics.loss_comparable:
            raise ValueError("official scoring requires loss_comparable=true")
        if not self.dataset.contamination_report_path:
            raise ValueError("official scoring requires benchmark-contamination report metadata")
        redistribution = self.metrics.loss.loss_component_redistribution
        if redistribution.enabled:
            raise ValueError("main-track official scoring cannot redistribute loss component")
        return self


def _artifact_relative_posix_path(value: str) -> str:
    if value.startswith("/"):
        raise ValueError("checkpoint paths must be artifact-relative POSIX paths")
    if "\\" in value:
        raise ValueError("checkpoint paths must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path == PurePosixPath("."):
        raise ValueError("checkpoint paths must be artifact-relative POSIX paths")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("checkpoint paths must not contain empty, '.', or '..' segments")
    if len(path.parts[0]) == 2 and path.parts[0][1] == ":":
        raise ValueError("checkpoint paths must not use host drive prefixes")
    return path.as_posix()



class DuplicateReport(SchemaModel):
    schema_version: str = Field(default="duplicate_report.v1", min_length=1)
    submission_id: str = Field(min_length=1)
    architecture_graph_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    candidate_architecture_id: str | None = Field(default=None, min_length=1)
    candidate_architecture_graph_hash: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"
    )
    source_similarity: float = Field(ge=0.0, le=1.0)
    graph_similarity: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    outcome: Literal["reject", "attach", "quarantine", "allow"]
    evidence: list[DeterministicEvidence] = Field(default_factory=list)
    reason: str = Field(min_length=1)


def validate_run_manifest_for_official_scoring(payload: dict[str, Any]) -> PrismRunManifest:
    manifest = PrismRunManifest.model_validate(payload)
    return manifest.require_official_scoring_ready()
