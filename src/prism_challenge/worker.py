from __future__ import annotations

import argparse
import asyncio
import logging

from .app import create_app
from .config import PrismSettings

logger = logging.getLogger("prism.worker")


async def run_worker(settings: PrismSettings, *, interval_seconds: float) -> None:
    app = create_app(settings)
    await app.state.database.init()
    try:
        while True:
            completed = await app.state.worker.poll_remote_jobs()
            submission_id = await app.state.worker.process_next()
            if submission_id or completed:
                logger.info(
                    "worker iteration completed",
                    extra={"submission_id": submission_id, "completed": completed},
                )
            await asyncio.sleep(interval_seconds)
    finally:
        await app.state.database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prism evaluation worker")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker(PrismSettings(), interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
