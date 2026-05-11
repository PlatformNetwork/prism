from __future__ import annotations

from datetime import UTC, datetime
from typing import SupportsFloat, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .auth import authenticate_miner
from .models import (
    ArchitectureFamilyResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    SubmissionCreate,
    SubmissionResponse,
    SubmissionStatusResponse,
    TrainingVariantResponse,
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
    request_body: SubmissionCreate,
    hotkey: str = Depends(authenticate_miner),
    repository: PrismRepository = Depends(repo_from_request),
) -> SubmissionResponse:
    if not request.app.state.settings.public_submissions_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "submission route disabled")
    if len(request_body.code.encode()) > request.app.state.settings.max_code_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "submission too large")
    return await repository.create_submission(hotkey, request_body)


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
    request: Request, repository: PrismRepository = Depends(repo_from_request)
) -> LeaderboardResponse:
    epoch_id = epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    rows = await repository.leaderboard(epoch_id)
    entries = [
        LeaderboardEntry(
            rank=index + 1,
            hotkey=str(row["hotkey"]),
            score=float(cast(SupportsFloat, row["final_score"])),
            submission_id=str(row["id"]),
        )
        for index, row in enumerate(rows)
    ]
    return LeaderboardResponse(epoch_id=epoch_id, entries=entries)


@public_route(tags=["components"])
@router.get("/architectures", response_model=list[ArchitectureFamilyResponse])
async def architectures(
    limit: int = 50, repository: PrismRepository = Depends(repo_from_request)
) -> list[ArchitectureFamilyResponse]:
    return [
        ArchitectureFamilyResponse(
            id=str(row["id"]),
            family_hash=str(row["family_hash"]),
            owner_hotkey=str(row["owner_hotkey"]),
            owner_submission_id=str(row["owner_submission_id"]),
            canonical_submission_id=str(row["canonical_submission_id"]),
            q_arch_best=float(cast(SupportsFloat, row["q_arch_best"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        for row in await repository.list_architectures(limit=limit)
    ]


@public_route(tags=["components"])
@router.get("/training-variants", response_model=list[TrainingVariantResponse])
async def training_variants(
    architecture_id: str | None = None,
    limit: int = 100,
    repository: PrismRepository = Depends(repo_from_request),
) -> list[TrainingVariantResponse]:
    return [
        TrainingVariantResponse(
            id=str(row["id"]),
            architecture_id=str(row["architecture_id"]),
            training_hash=str(row["training_hash"]),
            owner_hotkey=str(row["owner_hotkey"]),
            submission_id=str(row["submission_id"]),
            q_recipe=float(cast(SupportsFloat, row["q_recipe"])),
            metric_mean=float(cast(SupportsFloat, row["metric_mean"])),
            metric_std=float(cast(SupportsFloat, row["metric_std"])),
            is_current_best=bool(row["is_current_best"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        for row in await repository.list_training_variants(
            architecture_id=architecture_id,
            limit=limit,
        )
    ]


@public_route(tags=["epochs"])
@router.get("/epochs/current")
async def current_epoch(request: Request) -> dict[str, int]:
    epoch_id = epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    return {"epoch_id": epoch_id, "epoch_seconds": request.app.state.settings.epoch_seconds}
