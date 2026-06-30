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
    anti_cheat_multiplier: float | None = None
    diversity_bonus: float | None = None
    penalty: float | None = None


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


class EpochResponse(BaseModel):
    id: int
    starts_at: datetime
    ends_at: datetime
    status: str


class EvalJobHealthEntry(BaseModel):
    id: str
    submission_id: str
    level: str
    status: str
    attempts: int
    created_at: datetime
    updated_at: datetime


class GpuStatusSummary(BaseModel):
    total_gpus: int
    active_leases: int
    by_status: dict[str, int]
    by_tier: dict[str, int]


class SubmissionHistoryBucket(BaseModel):
    date: str
    count: int


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


class ArchitectureSummary(BaseModel):
    rank: int
    architecture_id: str
    arch_hash: str
    name: str | None
    owner_hotkey: str
    best_final_score: float
    best_submission_id: str
    variant_count: int
    submission_count: int
    updated_at: datetime


class ArchitectureListResponse(BaseModel):
    epoch_id: int
    architectures: list[ArchitectureSummary]


class ArchitectureDetailResponse(BaseModel):
    architecture_id: str
    arch_hash: str
    name: str | None
    owner_hotkey: str
    best_final_score: float
    best_submission_id: str
    variant_count: int
    submission_count: int
    first_seen_at: datetime
    updated_at: datetime


class TrainingVariantEntry(BaseModel):
    variant_id: str
    training_hash: str
    owner_hotkey: str
    submission_id: str
    final_score: float
    metric_mean: float
    metric_std: float
    is_current_best: bool
    created_at: datetime


class ArchitectureVariantsResponse(BaseModel):
    architecture_id: str
    variants: list[TrainingVariantEntry]


class LossCurveSeries(BaseModel):
    online_loss: list[float]
    covered_bytes_cumulative: list[float]
    step0_loss: float | None
    baseline_nats: float | None
    points: int
    downsampled: bool


class CurveBpb(BaseModel):
    prequential_bpb: float | None
    bits_per_byte: float | None


class CurveCompute(BaseModel):
    gpu_count: int | None
    device: str | None
    gpu_tier: str | None
    model_params: int | None
    tokens_consumed: int | None
    estimated_flops: float | None
    wall_clock_seconds: float | None
    gpu_hours: float | None
    peak_vram_bytes: int | None
    peak_rss_bytes: int | None


class SubmissionCurveResponse(BaseModel):
    submission_id: str
    loss_curve: LossCurveSeries
    bpb: CurveBpb
    compute: CurveCompute


class ArchitectureReport(BaseModel):
    status: str
    content: str | None
    model: str | None
    generated_at: datetime | None


class ArchitectureReportResponse(BaseModel):
    architecture_id: str
    report: ArchitectureReport



