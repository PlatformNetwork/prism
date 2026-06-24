from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from base.challenge_sdk.executors.docker import (
        DockerContainerInfo,
        DockerExecutor,
        DockerExecutorError,
        DockerLimits,
        DockerMount,
        DockerRunResult,
        DockerRunSpec,
    )
else:
    _platform_docker = None


def _load_platform_docker():
    try:
        return import_module("base.challenge_sdk.executors.docker")
    except ModuleNotFoundError as exc:
        if exc.name != "base":
            raise
        platform_src = Path("/droid/platform-v10/src")
        if platform_src.is_dir():
            sys.path.insert(0, str(platform_src))
            return import_module("base.challenge_sdk.executors.docker")
        raise


if not TYPE_CHECKING:
    _platform_docker = _load_platform_docker()
    DockerContainerInfo = _platform_docker.DockerContainerInfo
    DockerExecutor = _platform_docker.DockerExecutor
    DockerExecutorError = _platform_docker.DockerExecutorError
    DockerLimits = _platform_docker.DockerLimits
    DockerMount = _platform_docker.DockerMount
    DockerRunResult = _platform_docker.DockerRunResult
    DockerRunSpec = _platform_docker.DockerRunSpec
    sys.modules[__name__] = _platform_docker

__all__ = [
    "DockerContainerInfo",
    "DockerExecutor",
    "DockerExecutorError",
    "DockerLimits",
    "DockerMount",
    "DockerRunResult",
    "DockerRunSpec",
]
