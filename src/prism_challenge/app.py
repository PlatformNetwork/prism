from __future__ import annotations

import base64
import json
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .auth import authenticate_internal
from .config import PrismSettings, settings
from .coordination import list_pending_prism_work_units, work_unit_to_payload
from .db import Database
from .evaluator.interface import PrismContext
from .models import (
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
    repository = PrismRepository(
        database,
        app_settings.epoch_seconds,
        worker_claim_timeout_seconds=app_settings.worker_claim_timeout_seconds,
        held_review_timeout_seconds=app_settings.held_review_timeout_seconds,
    )
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
        runtime_config = await repository.runtime_config(app_settings, official=True)
        return await get_weights(
            repository,
            app_settings.epoch_seconds,
            architecture_weight=runtime_config.reward_pools.architecture,
            training_weight=runtime_config.reward_pools.training,
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

    @app.get("/internal/v1/work_units", dependencies=[Depends(authenticate_internal)])
    async def work_units() -> dict[str, object]:
        """Expose pending prism work units (one gpu unit per submission) to the master plane.

        The master coordination plane reads this to create exactly one assignable work unit per
        submission and assign it - with concurrency 1 - to a single online gpu validator. This
        endpoint is execution-free: enumerating work units never invokes the broker/executor.
        """
        units = await list_pending_prism_work_units(repository)
        return {
            "challenge_slug": app_settings.slug,
            "work_units": [work_unit_to_payload(unit) for unit in units],
        }

    @app.post(
        "/internal/v1/bridge/submissions",
        response_model=SubmissionResponse,
        dependencies=[Depends(authenticate_internal)],
    )
    async def bridge_submission(
        request: Request,
        x_base_verified_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
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
        return await repository.create_submission(x_base_verified_hotkey, submission)

    return app


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


app = create_app()
