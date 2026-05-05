from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class LiumJob:
    id: str
    status: str
    metrics: dict[str, float]


class LiumClient:
    def __init__(
        self,
        *,
        base_url: str | None,
        token: str | None,
        timeout: float = 30.0,
        backend: str = "jobs_api",
        executor_id: str | None = None,
        gpu_type: str | None = None,
        gpu_count: int = 1,
        template_id: str | None = None,
        ssh_key_path: str | None = None,
        keep_pod: bool = False,
        pod_timeout_seconds: int = 600,
        eval_timeout_seconds: int = 900,
        allow_fake: bool = False,
        sdk_factory: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.timeout = timeout
        self.backend = backend
        self.executor_id = executor_id
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.template_id = template_id
        self.ssh_key_path = ssh_key_path
        self.keep_pod = keep_pod
        self.pod_timeout_seconds = pod_timeout_seconds
        self.eval_timeout_seconds = eval_timeout_seconds
        self.allow_fake = allow_fake
        self.sdk_factory = sdk_factory

    def enabled(self) -> bool:
        if self.backend == "sdk":
            return bool(self.token)
        return bool(self.base_url and self.token)

    async def submit_job(self, payload: dict[str, Any], *, idempotency_key: str) -> LiumJob:
        if not self.enabled():
            if not self.allow_fake:
                raise RuntimeError("Lium backend is not configured")
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
        if self.backend == "sdk":
            return await asyncio.to_thread(self._submit_sdk_job, payload, idempotency_key)
        headers = {"Authorization": f"Bearer {self.token}", "Idempotency-Key": idempotency_key}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/jobs", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return LiumJob(str(data["id"]), str(data.get("status", "pending")), data.get("metrics", {}))

    async def poll_job(self, job_id: str) -> LiumJob:
        if not self.enabled():
            return LiumJob(job_id, "completed", {"val_loss": 2.0})
        if self.backend == "sdk":
            return LiumJob(job_id, "completed", {})
        headers = {"Authorization": f"Bearer {self.token}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/jobs/{job_id}", headers=headers)
            response.raise_for_status()
            data = response.json()
        return LiumJob(str(data["id"]), str(data.get("status", "pending")), data.get("metrics", {}))

    def _submit_sdk_job(self, payload: dict[str, Any], idempotency_key: str) -> LiumJob:
        lium = self._sdk_client()
        executor_id = self.executor_id or self._select_executor(lium)
        pod_data = lium.up(
            executor_id=executor_id,
            name=f"prism-{idempotency_key[:12]}",
            template_id=self.template_id,
            ports=None,
        )
        pod = lium.wait_ready(pod_data, timeout=self.pod_timeout_seconds, poll_interval=10)
        if pod is None:
            raise RuntimeError("Lium pod did not become ready before timeout")
        try:
            command = self._prepare_remote_eval(lium, pod, payload)
            result = lium.exec(pod, command=command)
            if not result.get("success"):
                raise RuntimeError(
                    f"Lium remote evaluation failed: {str(result.get('stderr', ''))[-2000:]}"
                )
            metrics = self._parse_metrics(str(result.get("stdout", "")))
            return LiumJob(str(pod.id), "completed", metrics)
        finally:
            if not self.keep_pod:
                with suppress(Exception):
                    lium.rm(pod)

    def _sdk_client(self) -> Any:
        if self.sdk_factory is not None:
            return self.sdk_factory(self)
        try:
            from lium.sdk import Config, Lium  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Lium SDK is not installed; install prism-challenge[lium] and lium.io"
            ) from exc
        config_kwargs: dict[str, Any] = {"api_key": self.token}
        if self.base_url:
            config_kwargs["base_url"] = self.base_url
        ssh_key_path = self._resolve_ssh_key_path()
        if ssh_key_path is not None:
            config_kwargs["ssh_key_path"] = ssh_key_path
        return Lium(Config(**config_kwargs), source="prism-challenge")

    def _resolve_ssh_key_path(self) -> Path | None:
        configured = self.ssh_key_path or os.getenv("LIUM_SSH_KEY_PATH")
        if configured:
            return Path(configured)
        for key_name in ("id_ed25519", "id_rsa", "id_ecdsa"):
            path = Path.home() / ".ssh" / key_name
            if path.exists() and path.with_suffix(".pub").exists():
                return path
        return None

    def _select_executor(self, lium: Any) -> str:
        executors = lium.ls(gpu_type=self.gpu_type, gpu_count=self.gpu_count)
        if not executors:
            raise RuntimeError("No matching Lium executors available")

        def price(executor: Any) -> float:
            return float(getattr(executor, "price_per_hour", 0.0) or 0.0)

        return str(min(executors, key=price).id)

    def _remote_eval_command(self, payload: dict[str, Any]) -> str:
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        return (
            f"timeout {self.eval_timeout_seconds}s python3 - '{encoded}' <<'PY'\n"
            f"{_REMOTE_EVAL_SCRIPT}\nPY\n"
        )

    def _prepare_remote_eval(self, lium: Any, pod: Any, payload: dict[str, Any]) -> str:
        if not hasattr(lium, "upload"):
            return self._remote_eval_command(payload)
        with tempfile.TemporaryDirectory(prefix="prism-lium-") as tmp:
            payload_path = Path(tmp) / "payload.json"
            runner_path = Path(tmp) / "runner.py"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            runner_path.write_text(_REMOTE_EVAL_SCRIPT, encoding="utf-8")
            remote_payload = f"/tmp/prism_payload_{os.getpid()}.json"
            remote_runner = f"/tmp/prism_eval_{os.getpid()}.py"
            lium.upload(pod, local=str(payload_path), remote=remote_payload)
            lium.upload(pod, local=str(runner_path), remote=remote_runner)
        return f"timeout {self.eval_timeout_seconds}s python3 {remote_runner} {remote_payload}"

    def _parse_metrics(self, stdout: str) -> dict[str, float]:
        for line in reversed(stdout.splitlines()):
            if line.startswith("PRISM_METRICS_JSON="):
                data = json.loads(line.removeprefix("PRISM_METRICS_JSON="))
                return {str(k): float(v) for k, v in data.items()}
        raise RuntimeError("Lium remote evaluation did not return metrics")


_REMOTE_EVAL_SCRIPT = r"""
import base64
import dataclasses
import json
import math
import sys
import types

if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)
else:
    payload = json.loads(base64.b64decode(sys.argv[1]).decode())

interface = types.ModuleType("prism_challenge.evaluator.interface")

@dataclasses.dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 4096
    sequence_length: int = 128
    max_layers: int = 96
    max_parameters: int = 150_000_000
    seed: int = 1337

@dataclasses.dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 3e-4
    batch_size: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01

@dataclasses.dataclass(frozen=True)
class PrismBatch:
    tokens: object
    targets: object | None = None
    metadata: dict | None = None

interface.PrismContext = PrismContext
interface.TrainingRecipe = TrainingRecipe
interface.PrismBatch = PrismBatch
pkg = types.ModuleType("prism_challenge")
evaluator = types.ModuleType("prism_challenge.evaluator")
sys.modules["prism_challenge"] = pkg
sys.modules["prism_challenge.evaluator"] = evaluator
sys.modules["prism_challenge.evaluator.interface"] = interface

import torch
import torch.nn.functional as F

ctx_data = payload.get("context", {})
ctx = PrismContext(
    vocab_size=int(ctx_data.get("vocab_size", 4096)),
    sequence_length=min(int(ctx_data.get("sequence_length", 128)), 128),
    max_parameters=int(ctx_data.get("max_parameters", 150_000_000)),
)
module = types.ModuleType("prism_submission")
exec(compile(payload["code"], "<prism_submission>", "exec"), module.__dict__)
torch.manual_seed(ctx.seed)
model = module.build_model(ctx)
recipe = module.get_recipe(ctx)
if isinstance(recipe, dict):
    recipe = TrainingRecipe(**recipe)
params = sum(p.numel() for p in model.parameters())
if params <= 0 or params > ctx.max_parameters:
    raise RuntimeError(f"invalid parameter count: {params}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
seq = ctx.sequence_length
batch_size = max(1, min(int(getattr(recipe, "batch_size", 1)), 2))
tokens = torch.randint(0, ctx.vocab_size, (batch_size, seq), device=device)

def prism_batch(t):
    return PrismBatch(tokens=t[:, :-1], targets=t[:, 1:], metadata={})

def logits_for(t):
    custom = getattr(module, "inference_logits", None) or getattr(module, "infer", None)
    if callable(custom):
        return custom(model, prism_batch(t), ctx)
    return model(t[:, :-1])

def loss_for(t):
    custom = getattr(module, "compute_loss", None)
    if callable(custom):
        return custom(model, prism_batch(t), ctx)
    logits = logits_for(t)
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), t[:, 1:].reshape(-1) % vocab)

custom_opt = getattr(module, "configure_optimizer", None)
if callable(custom_opt):
    optimizer = custom_opt(model, recipe, ctx)
else:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=min(float(getattr(recipe, "learning_rate", 3e-4)), 3e-4),
        weight_decay=float(getattr(recipe, "weight_decay", 0.01)),
    )
initial_loss = float(loss_for(tokens).detach().cpu())
final_loss = initial_loss
for _ in range(3):
    batch = torch.randint(0, ctx.vocab_size, (batch_size, seq), device=device)
    custom_step = getattr(module, "train_step", None)
    if callable(custom_step):
        loss = custom_step(model, prism_batch(batch), optimizer, ctx)
    else:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final_loss = float(loss.detach().cpu())
improvement = max(0.0, initial_loss - final_loss)
quality = max(0.0, min(1.0, improvement / max(initial_loss, 1e-6)))
efficiency = 1.0 / (1.0 + math.log10(max(params, 1)))
q_arch = max(0.0, min(1.0, 0.82 * quality + 0.18 * efficiency))
metrics = {
    "q_arch": q_arch,
    "q_recipe": 1.0 if 1e-5 <= float(getattr(recipe, "learning_rate", 3e-4)) <= 3e-3 else 0.5,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "val_loss": final_loss,
    "parameter_count": float(params),
}
print("PRISM_METRICS_JSON=" + json.dumps(metrics, separators=(",", ":")))
"""
