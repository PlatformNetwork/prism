from __future__ import annotations

import base64
import binascii
import json
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .auth import authenticate_internal, authenticate_validator
from .config import PrismSettings, settings
from .coordination import list_pending_prism_work_units, work_unit_to_payload
from .db import Database
from .evaluator.checkpoint_intake import CheckpointIntakeError, CheckpointIntakeService
from .evaluator.checkpoint_publisher import (
    CheckpointPublisher,
    HuggingFaceCheckpointPublisher,
)
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


def create_app(
    app_settings: PrismSettings = settings,
    *,
    checkpoint_publisher: CheckpointPublisher | None = None,
) -> FastAPI:
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
    # Tests inject a MockCheckpointPublisher (no network); deploy uses the real lazy HF client.
    # Constructing the real client never imports huggingface_hub, so this stays offline-safe.
    publisher = checkpoint_publisher or HuggingFaceCheckpointPublisher(
        repo_id=app_settings.checkpoint_repo_id, token=app_settings.hf_token_value()
    )
    checkpoint_intake = CheckpointIntakeService(publisher=publisher, repository=repository)

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
    app.state.checkpoint_publisher = publisher
    app.state.checkpoint_intake = checkpoint_intake
    # In-process guards for lazy, non-blocking LLM auto-report generation: ``report_inflight``
    # dedupes concurrent generations for the same architecture; ``report_failed`` maps an
    # architecture_id to the best-submission cache key whose last generation errored (so GETs
    # return ``unavailable`` until a new best arrives and a retry is allowed).
    app.state.report_inflight = set()
    app.state.report_failed = {}

    @app.post("/internal/v1/worker/process-next", dependencies=[Depends(authenticate_internal)])
    async def process_next() -> dict[str, str | None]:
        return {"submission_id": await worker.process_next()}

    @app.post("/internal/v1/checkpoints")
    async def publish_checkpoint(
        request: Request,
        validator_hotkey: Annotated[str, Depends(authenticate_validator)],
    ) -> dict[str, object]:
        """Receive a validator's pushed checkpoint and publish it to HuggingFace (mocked in tests).

        Hotkey-signed + validator-permit gated via ``authenticate_validator`` (a rejected caller
        never reaches this body, so no ``checkpoint_ref`` is recorded on rejection). On success the
        checkpoint is published through the publisher interface and its public ``checkpoint_ref`` is
        recorded on the submission's assignment for resume-on-reassignment (VAL-PRISM-022/038).
        """
        intake: CheckpointIntakeService = request.app.state.checkpoint_intake
        body = await request.body()
        submission_id, attempt, files, revision = _parse_checkpoint_upload(body)
        try:
            published = await intake.publish(
                submission_id=submission_id,
                attempt=attempt,
                validator_hotkey=validator_hotkey,
                files=files,
                revision=revision,
            )
        except CheckpointIntakeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {
            "checkpoint_ref": published.checkpoint_ref,
            "repo_id": published.repo_id,
            "revision": published.revision,
            "files": list(published.files),
        }

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


def _parse_checkpoint_upload(body: bytes) -> tuple[str, int, dict[str, bytes], str | None]:
    """Parse + validate a validator checkpoint-upload payload into (submission_id, attempt, files).

    The payload is JSON ``{submission_id, attempt, files:{name: base64}, revision?}``. A malformed
    body / non-integer attempt / non-base64 file bytes is a 400 (never a publish).
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON checkpoint upload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "checkpoint upload must be an object")
    submission_id = payload.get("submission_id")
    if not isinstance(submission_id, str) or not submission_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "submission_id is required")
    attempt_raw = payload.get("attempt", 1)
    try:
        attempt = int(attempt_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "attempt must be an integer") from exc
    if attempt < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "attempt must be >= 1")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "files must be a non-empty object")
    files: dict[str, bytes] = {}
    for name, encoded in raw_files.items():
        if not isinstance(name, str) or not isinstance(encoded, str):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "files entries must be base64 strings")
        try:
            files[name] = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"file {name} is not valid base64"
            ) from exc
    revision = payload.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "revision must be a string")
    return submission_id, attempt, files, revision


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
