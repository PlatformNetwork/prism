from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    slug: str
    version: str


class VersionResponse(BaseModel):
    api_version: str
    challenge_version: str
    sdk_version: str
    capabilities: list[str] = Field(default_factory=list)


class WeightsResponse(BaseModel):
    challenge_slug: str
    epoch: int | None = None
    weights: dict[str, float]
    metadata: dict[str, str] = Field(
        default_factory=lambda: {"computed_at": datetime.now(UTC).isoformat()}
    )
