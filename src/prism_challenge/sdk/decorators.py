from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def public_route(
    *, tags: list[str] | None = None, auth_required: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func.__platform_public_route__ = True  # type: ignore[attr-defined]
        func.__platform_public_tags__ = tags or []  # type: ignore[attr-defined]
        func.__platform_public_auth_required__ = auth_required  # type: ignore[attr-defined]
        return func

    return decorator


def is_public_route(func: Callable[..., object]) -> bool:
    return bool(getattr(func, "__platform_public_route__", False))
