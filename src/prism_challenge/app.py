from __future__ import annotations

import base64
import json
from typing import Annotated, SupportsFloat, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .auth import authenticate_internal
from .config import PrismSettings, settings
from .db import Database
from .evaluator.interface import PrismContext
from .models import (
    ComponentReviewHoldDecision,
    ComponentReviewHoldResponse,
    EvaluationAssignmentDecision,
    EvaluationAssignmentResponse,
    EvaluationResultCreate,
    SubmissionCreate,
    SubmissionResponse,
)
from .queue import PrismWorker
from .repository import PrismRepository
from .routes import router
from .sdk import create_challenge_app
from .weights import get_weights


def create_app(app_settings: PrismSettings = settings) -> FastAPI:
    database = Database(app_settings.resolved_database_path)
    repository = PrismRepository(database, app_settings.epoch_seconds)
    ctx = PrismContext(
        sequence_length=app_settings.sequence_length,
        max_layers=app_settings.max_layers,
        max_parameters=app_settings.max_parameters,
    )
    worker = PrismWorker(
        repository,
        ctx,
        execution_backend=app_settings.execution_backend,
        settings=app_settings,
    )

    async def get_weights_fn() -> dict[str, float]:
        return await get_weights(
            repository,
            app_settings.epoch_seconds,
            architecture_weight=app_settings.architecture_reward_weight,
            training_weight=app_settings.training_reward_weight,
        )

    app = create_challenge_app(
        settings=app_settings,
        database=database,
        public_router=router,
        get_weights_fn=get_weights_fn,
    )
    app.state.settings = app_settings
    app.state.database = database
    app.state.repository = repository
    app.state.worker = worker

    @app.post("/internal/v1/worker/process-next", dependencies=[Depends(authenticate_internal)])
    async def process_next() -> dict[str, str | None]:
        return {"submission_id": await worker.process_next()}

    @app.post("/internal/v1/worker/poll", dependencies=[Depends(authenticate_internal)])
    async def poll_workers() -> dict[str, list[str]]:
        return {"completed_submission_ids": await worker.poll_remote_jobs()}

    @app.post(
        "/internal/v1/bridge/submissions",
        response_model=SubmissionResponse,
        dependencies=[Depends(authenticate_internal)],
    )
    async def bridge_submission(
        request: Request,
        x_platform_verified_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
        x_submission_filename: Annotated[str | None, Header()] = None,
    ) -> SubmissionResponse:
        body = await request.body()
        submission = _bridge_submission_create(
            body=body,
            content_type=request.headers.get("content-type", ""),
            filename=x_submission_filename,
        )
        if len(submission.code.encode()) > app_settings.max_code_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "submission too large")
        return await repository.create_submission(x_platform_verified_hotkey, submission)

    @app.post(
        "/internal/v1/validators/assignments/next",
        response_model=EvaluationAssignmentResponse | None,
        dependencies=[Depends(authenticate_internal)],
    )
    async def next_assignment(
        x_validator_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
    ) -> EvaluationAssignmentResponse | None:
        _ensure_validator_allowed(app_settings, x_validator_hotkey)
        return await worker.assign_next_to_validator(x_validator_hotkey)

    @app.post(
        "/internal/v1/validators/assignments/{assignment_id}/accept",
        dependencies=[Depends(authenticate_internal)],
    )
    async def accept_assignment(assignment_id: str) -> dict[str, str]:
        from .models import EvaluationAssignmentStatus

        assignment = await repository.get_assignment(assignment_id)
        if assignment is None:
            raise HTTPException(404, "assignment not found")
        await repository.set_assignment_status(assignment_id, EvaluationAssignmentStatus.ACCEPTED)
        return {"status": "accepted"}

    @app.post(
        "/internal/v1/validators/assignments/{assignment_id}/reject",
        dependencies=[Depends(authenticate_internal)],
    )
    async def reject_assignment(
        assignment_id: str, decision: EvaluationAssignmentDecision
    ) -> dict[str, str]:
        try:
            await worker.reject_assignment(assignment_id, decision.reason)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "rejected"}

    @app.post(
        "/internal/v1/validators/assignments/{assignment_id}/result",
        dependencies=[Depends(authenticate_internal)],
    )
    async def submit_assignment_result(
        assignment_id: str, result: EvaluationResultCreate
    ) -> dict[str, str]:
        try:
            await worker.complete_assignment(assignment_id, result.metrics)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "completed"}

    @app.post(
        "/internal/v1/validators/assignments/expire",
        dependencies=[Depends(authenticate_internal)],
    )
    async def expire_assignments() -> dict[str, list[str]]:
        return {"expired_submission_ids": await worker.expire_assignments()}

    @app.get(
        "/internal/v1/component-review/holds",
        response_model=list[ComponentReviewHoldResponse],
        dependencies=[Depends(authenticate_internal)],
    )
    async def component_review_holds() -> list[ComponentReviewHoldResponse]:
        return [
            ComponentReviewHoldResponse(
                id=str(row["id"]),
                submission_id=str(row["submission_id"]),
                status=str(row["status"]),
                reason=str(row["reason"]),
                confidence=float(cast(SupportsFloat, row["confidence"])),
                created_at=_parse_dt(str(row["created_at"])),
                updated_at=_parse_dt(str(row["updated_at"])),
            )
            for row in await repository.list_component_review_holds()
        ]

    @app.post(
        "/internal/v1/component-review/holds/{hold_id}/resolve",
        dependencies=[Depends(authenticate_internal)],
    )
    async def resolve_component_review_hold(
        hold_id: str, decision: ComponentReviewHoldDecision
    ) -> dict[str, object]:
        try:
            return await repository.resolve_component_hold(
                hold_id=hold_id,
                architecture_action=decision.architecture_action,
                training_action=decision.training_action,
                architecture_id=decision.architecture_id,
                training_variant_id=decision.training_variant_id,
                reason=decision.reason or "manual component attribution resolution",
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app


def _ensure_validator_allowed(settings: PrismSettings, validator_hotkey: str) -> None:
    if settings.validator_hotkeys and validator_hotkey not in settings.validator_hotkeys:
        raise HTTPException(403, "validator is not allowed")


def _bridge_submission_create(
    *, body: bytes, content_type: str, filename: str | None
) -> SubmissionCreate:
    if "application/json" in content_type.lower():
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON submission") from exc
        return SubmissionCreate.model_validate(payload)
    safe_filename = filename or "submission.zip"
    if not safe_filename.endswith((".py", ".zip")):
        safe_filename = "submission.zip"
    return SubmissionCreate(
        code=base64.b64encode(body).decode("ascii"),
        filename=safe_filename,
        metadata={"content_type": content_type or "application/octet-stream", "bridge": True},
    )


def _parse_dt(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)


app = create_app()
