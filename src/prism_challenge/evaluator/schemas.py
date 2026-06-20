from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# v2 re-execution manifest (architecture.md sections 4.3, 5). Authored by the challenge runner;
# any miner-written manifest is ignored. The prequential bits-per-byte scoring fields are filled
# in by the scoring recast; the runner records the forced-init re-execution provenance.
RUN_MANIFEST_V2_FILENAME = "prism_run_manifest.v2.json"
RUN_MANIFEST_V2_SCHEMA_VERSION = "prism_run_manifest.v2"
# Typed, observability-only `compute` block recorded on the v2 run manifest.
COMPUTE_BLOCK_SCHEMA = "prism_compute.v1"


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComputeBlock(SchemaModel):
    """Typed, observability-only compute record for the v2 run manifest.

    Records the GPUs actually LEASED for the scored re-execution (``gpu_count``; ``== 1`` for the
    scored single-GPU ``nproc=1`` path) plus the launch shape already known to the harness
    (``world_size`` / ``nproc_per_node`` / ``device``). It is a RECORDED field for observability and
    so VAL-GPU-005 can be asserted by field: the prequential bits-per-byte ``final_score`` derives
    SOLELY from compute-normalized learning metrics and never reads ``gpu_count`` (there is no
    GPU-count reward and no multi-GPU scaling bonus). For the single-node scored run
    ``gpu_count == world_size == nproc_per_node`` and equals the DB ``eval_jobs.actual_gpu_count``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Aliased to the JSON key "schema" (matching the score block convention) without shadowing the
    # deprecated ``BaseModel.schema`` attribute.
    compute_schema: str = Field(default=COMPUTE_BLOCK_SCHEMA, min_length=1, alias="schema")
    gpu_count: int = Field(ge=0)
    world_size: int = Field(ge=1)
    nproc_per_node: int = Field(ge=1)
    device: str = Field(min_length=1)
    max_gpu_count: int | None = Field(default=None, ge=1)


class ExecutionMode(StrEnum):
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
