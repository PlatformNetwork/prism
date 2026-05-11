from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class SubmissionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    HELD = "held"


class JobLevel(StrEnum):
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    L4 = "l4"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationAssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SubmissionCreate(BaseModel):
    code: str = Field(min_length=1)
    filename: str = Field(default="model.py", pattern=r"^[A-Za-z0-9_.-]+\.(py|zip)$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubmissionResponse(BaseModel):
    id: str
    hotkey: str
    epoch_id: int
    status: SubmissionStatus
    code_hash: str
    created_at: datetime


class SubmissionStatusResponse(SubmissionResponse):
    error: str | None = None
    final_score: float | None = None
    q_arch: float | None = None
    q_recipe: float | None = None


class EvaluationAssignmentResponse(BaseModel):
    id: str
    submission_id: str
    validator_hotkey: str
    status: EvaluationAssignmentStatus
    attempt: int
    deadline_at: datetime
    code: str
    filename: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    code_hash: str
    arch_hash: str


class EvaluationAssignmentDecision(BaseModel):
    reason: str | None = None


class EvaluationResultCreate(BaseModel):
    metrics: dict[str, float]


class LeaderboardEntry(BaseModel):
    rank: int
    hotkey: str
    score: float
    submission_id: str


class LeaderboardResponse(BaseModel):
    epoch_id: int
    entries: list[LeaderboardEntry]


class WeightsResponse(BaseModel):
    challenge_slug: str
    epoch: int
    weights: dict[str, float]


class ArchitectureFamilyResponse(BaseModel):
    id: str
    family_hash: str
    owner_hotkey: str
    owner_submission_id: str
    canonical_submission_id: str
    q_arch_best: float
    created_at: datetime
    updated_at: datetime


class TrainingVariantResponse(BaseModel):
    id: str
    architecture_id: str
    training_hash: str
    owner_hotkey: str
    submission_id: str
    q_recipe: float
    metric_mean: float
    metric_std: float
    is_current_best: bool
    created_at: datetime
    updated_at: datetime


class ComponentReviewHoldResponse(BaseModel):
    id: str
    submission_id: str
    status: str
    reason: str
    confidence: float
    created_at: datetime
    updated_at: datetime


class ComponentReviewHoldDecision(BaseModel):
    architecture_action: str = Field(pattern=r"^(new|existing|transfer|reject)$")
    training_action: str = Field(default="none", pattern=r"^(new|existing|transfer|reject|none)$")
    architecture_id: str | None = None
    training_variant_id: str | None = None
    reason: str | None = None
