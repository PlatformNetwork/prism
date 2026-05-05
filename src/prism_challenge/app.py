from __future__ import annotations

from fastapi import Depends, FastAPI

from .auth import authenticate_internal
from .config import PrismSettings, settings
from .db import Database
from .evaluator.interface import PrismContext
from .evaluator.lium_client import LiumClient
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
    lium = LiumClient(
        base_url=app_settings.lium_base_url,
        token=app_settings.lium_auth_token(),
        backend=app_settings.lium_backend,
        executor_id=app_settings.lium_executor_id,
        gpu_type=app_settings.lium_gpu_type,
        gpu_count=app_settings.lium_gpu_count,
        template_id=app_settings.lium_template_id,
        ssh_key_path=app_settings.lium_ssh_key_path,
        keep_pod=app_settings.lium_keep_pod,
        pod_timeout_seconds=app_settings.lium_pod_timeout_seconds,
        eval_timeout_seconds=app_settings.lium_eval_timeout_seconds,
        allow_fake=app_settings.allow_fake_lium,
    )
    worker = PrismWorker(
        repository,
        ctx,
        lium,
        execution_backend=app_settings.execution_backend,
    )

    async def get_weights_fn() -> dict[str, float]:
        return await get_weights(repository, app_settings.epoch_seconds)

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

    return app


app = create_app()
