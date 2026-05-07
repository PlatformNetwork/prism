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
