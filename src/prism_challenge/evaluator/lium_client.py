from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class LiumJob:
    id: str
    status: str
    metrics: dict[str, float]


class LiumClient:
    def __init__(self, *, base_url: str | None, token: str | None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.timeout = timeout

    def enabled(self) -> bool:
        return bool(self.base_url and self.token)

    async def submit_job(self, payload: dict[str, Any], *, idempotency_key: str) -> LiumJob:
        if not self.enabled():
            q = float(payload.get("q_train", payload.get("q_arch", 0.42)))
            return LiumJob(
                f"fake-{idempotency_key}",
                "completed",
                {
                    "q_arch": max(0.0, min(1.0, q)),
                    "q_recipe": 0.75,
                    "val_loss": 1 / max(q, 0.01),
                },
            )
        headers = {"Authorization": f"Bearer {self.token}", "Idempotency-Key": idempotency_key}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/jobs", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return LiumJob(str(data["id"]), str(data.get("status", "pending")), data.get("metrics", {}))

    async def poll_job(self, job_id: str) -> LiumJob:
        if not self.enabled():
            return LiumJob(job_id, "completed", {"val_loss": 2.0})
        headers = {"Authorization": f"Bearer {self.token}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/jobs/{job_id}", headers=headers)
            response.raise_for_status()
            data = response.json()
        return LiumJob(str(data["id"]), str(data.get("status", "pending")), data.get("metrics", {}))
