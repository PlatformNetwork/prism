from __future__ import annotations

from datetime import UTC, datetime
from typing import SupportsFloat, SupportsInt, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import ValidationError

from .auth import authenticate_miner
from .models import (
    EpochResponse,
    EvalJobHealthEntry,
    GpuStatusSummary,
    LeaderboardEntry,
    LeaderboardResponse,
    SubmissionHistoryBucket,
    SubmissionResponse,
    SubmissionStatusResponse,
)
from .repository import PrismRepository, epoch_id_for
from .sdk import public_route

router = APIRouter(prefix="/v1")


def repo_from_request(request: Request) -> PrismRepository:
    return request.app.state.repository


@public_route(tags=["submissions"], auth_required=True)
@router.post("/submissions", response_model=SubmissionResponse)
async def submit_model(
    request: Request,
    hotkey: str = Depends(authenticate_miner),
    repository: PrismRepository = Depends(repo_from_request),
) -> SubmissionResponse:
    from .app import _bridge_submission_create

    body = await request.body()
    try:
        request_body = _bridge_submission_create(
            body=body,
            content_type=request.headers.get("content-type", ""),
            filename=request.headers.get("x-submission-filename"),
        )
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors()) from exc
    if len(request_body.code.encode()) > request.app.state.settings.max_code_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "submission too large")
    return await repository.create_submission(hotkey, request_body)


@public_route(tags=["submissions"])
@router.get("/submissions/history", response_model=list[SubmissionHistoryBucket])
async def submission_history(
    days: int = Query(default=90, ge=1, le=366),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[SubmissionHistoryBucket]:
    return [
        SubmissionHistoryBucket(
            date=str(row["day"]),
            count=int(cast(SupportsInt, row["count"])),
        )
        for row in await repository.submission_history(days=days)
    ]


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}", response_model=SubmissionStatusResponse)
async def submission_status(
    submission_id: str, repository: PrismRepository = Depends(repo_from_request)
) -> SubmissionStatusResponse:
    submission = await repository.get_submission(submission_id)
    if submission is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "submission not found")
    return submission


@public_route(tags=["leaderboard"])
@router.get("/leaderboard", response_model=LeaderboardResponse)
async def leaderboard(
    request: Request,
    epoch_id: int | None = Query(default=None, ge=0),
    repository: PrismRepository = Depends(repo_from_request),
) -> LeaderboardResponse:
    resolved_epoch_id = (
        epoch_id
        if epoch_id is not None
        else epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    )
    rows = await repository.leaderboard(resolved_epoch_id)
    entries = [
        LeaderboardEntry(
            rank=index + 1,
            hotkey=str(row["hotkey"]),
            score=float(cast(SupportsFloat, row["final_score"])),
            submission_id=str(row["id"]),
        )
        for index, row in enumerate(rows)
    ]
    return LeaderboardResponse(epoch_id=resolved_epoch_id, entries=entries)


@public_route(tags=["epochs"])
@router.get("/epochs/current")
async def current_epoch(request: Request) -> dict[str, int]:
    epoch_id = epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    return {"epoch_id": epoch_id, "epoch_seconds": request.app.state.settings.epoch_seconds}


@public_route(tags=["epochs"])
@router.get("/epochs", response_model=list[EpochResponse])
async def list_epochs(
    limit: int = Query(default=50, ge=1, le=200),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[EpochResponse]:
    return [
        EpochResponse(
            id=int(cast(SupportsInt, row["id"])),
            starts_at=datetime.fromisoformat(str(row["starts_at"])),
            ends_at=datetime.fromisoformat(str(row["ends_at"])),
            status=str(row["status"]),
        )
        for row in await repository.list_epochs(limit=limit)
    ]


@public_route(tags=["health"])
@router.get("/health/eval-jobs", response_model=list[EvalJobHealthEntry])
async def eval_job_health(
    limit: int = Query(default=50, ge=1, le=200),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[EvalJobHealthEntry]:
    return [
        EvalJobHealthEntry(
            id=str(row["id"]),
            submission_id=str(row["submission_id"]),
            level=str(row["level"]),
            status=str(row["status"]),
            attempts=int(cast(SupportsInt, row["attempts"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        for row in await repository.list_eval_job_health(limit=limit)
    ]


@public_route(tags=["gpu"])
@router.get("/gpu/status", response_model=GpuStatusSummary)
async def gpu_status(
    repository: PrismRepository = Depends(repo_from_request),
) -> GpuStatusSummary:
    status_rows, tier_rows = await repository.gpu_status_summary()
    by_status: dict[str, int] = {}
    total_gpus = 0
    for row in status_rows:
        status_value = str(row["status"])
        by_status[status_value] = int(cast(SupportsInt, row["lease_count"]))
        if status_value == "active":
            total_gpus = int(cast(SupportsInt, row["gpu_total"]))
    by_tier = {
        str(row["tier"]): int(cast(SupportsInt, row["lease_count"])) for row in tier_rows
    }
    return GpuStatusSummary(
        total_gpus=total_gpus,
        active_leases=by_status.get("active", 0),
        by_status=by_status,
        by_tier=by_tier,
    )
