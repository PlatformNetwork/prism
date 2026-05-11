from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..config import PrismSettings
from ..sdk.executors.docker import DockerExecutor, DockerLimits, DockerMount, DockerRunSpec
from .interface import PrismContext
from .sandbox import OPTIONAL_CONTRACT_FUNCTIONS, REQUIRED_CONTRACT_FUNCTIONS
from .source_similarity import SourceFile


@dataclass(frozen=True)
class ContainerEvaluationResult:
    container_name: str
    metrics: dict[str, float]


class PrismContainerEvaluator:
    def __init__(self, *, settings: PrismSettings, ctx: PrismContext) -> None:
        self.settings = settings
        self.ctx = ctx

    def evaluate(
        self,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        backend: str,
        files: tuple[SourceFile, ...] = (),
        entrypoint: str | None = None,
    ) -> ContainerEvaluationResult:
        with TemporaryDirectory(prefix=f"prism-eval-{submission_id[:12]}-") as tmp:
            workspace = Path(tmp)
            payload_path = workspace / "payload.json"
            runner_path = workspace / "runner.py"
            payload_path.write_text(
                json.dumps(
                    self._payload(
                        submission_id=submission_id,
                        code=code,
                        code_hash=code_hash,
                        arch_hash=arch_hash,
                        files=files,
                        entrypoint=entrypoint,
                    ),
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            project = workspace / "project"
            project.mkdir()
            for file in files or (SourceFile("model.py", code, code_hash),):
                target = project / file.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(file.content, encoding="utf-8")
            runner_path.write_text(_CONTAINER_EVAL_SCRIPT, encoding="utf-8")
            result = self._executor().run(
                DockerRunSpec(
                    image=self.settings.platform_eval_image,
                    command=("python", "/workspace/runner.py", "/workspace/payload.json"),
                    mounts=(DockerMount(workspace, "/workspace"),),
                    workdir="/workspace",
                    env=self._env(submission_id, code_hash, arch_hash, backend),
                    labels={
                        "platform.job": submission_id,
                        "platform.task": self.settings.platform_eval_task,
                        "platform.backend": backend,
                    },
                    limits=DockerLimits(
                        cpus=self.settings.platform_eval_cpus,
                        memory=self.settings.platform_eval_memory,
                        memory_swap=self.settings.platform_eval_memory_swap,
                        pids_limit=self.settings.platform_eval_pids_limit,
                        network=self.settings.docker_network,
                        read_only=self.settings.platform_eval_read_only,
                        user=self.settings.docker_user,
                    ),
                ),
                self.settings.platform_eval_timeout_seconds,
            )
        if result.timed_out:
            raise RuntimeError("Prism container evaluation timed out")
        if result.returncode != 0:
            detail = result.stderr or result.stdout or "container returned non-zero status"
            raise RuntimeError(f"Prism container evaluation failed: {detail[-2000:]}")
        return ContainerEvaluationResult(
            container_name=result.container_name,
            metrics=_parse_metrics(result.stdout),
        )

    def _executor(self) -> DockerExecutor:
        return DockerExecutor(
            challenge=self.settings.slug,
            docker_bin=self.settings.docker_bin,
            allowed_images=tuple(self.settings.docker_allowed_images)
            or (self.settings.platform_eval_image,),
            backend=self.settings.docker_backend,
            broker_url=self.settings.docker_broker_url,
            broker_token=self.settings.docker_broker_token,
            broker_token_file=str(self.settings.docker_broker_token_file)
            if self.settings.docker_broker_token_file
            else None,
        )

    def _payload(
        self,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        files: tuple[SourceFile, ...],
        entrypoint: str | None,
    ) -> dict[str, Any]:
        payload_files = files or (SourceFile("model.py", code, code_hash),)
        return {
            "challenge": self.settings.slug,
            "submission_id": submission_id,
            "code": code,
            "files": [
                {"path": file.path, "content": file.content, "sha256": file.sha256}
                for file in payload_files
            ],
            "entrypoint": entrypoint or _entrypoint(payload_files),
            "code_hash": code_hash,
            "arch_hash": arch_hash,
            "contract": {
                "required": sorted(REQUIRED_CONTRACT_FUNCTIONS),
                "optional": sorted(OPTIONAL_CONTRACT_FUNCTIONS),
                "metrics": [
                    "q_arch",
                    "q_recipe",
                    "train_loss",
                    "eval_loss",
                    "parameter_count",
                    "inference_latency_ms",
                ],
            },
            "context": {
                "vocab_size": self.ctx.vocab_size,
                "sequence_length": self.ctx.sequence_length,
                "max_parameters": self.ctx.max_parameters,
            },
        }

    def _env(
        self,
        submission_id: str,
        code_hash: str,
        arch_hash: str,
        backend: str,
    ) -> dict[str, str]:
        env = {
            "PRISM_SUBMISSION_ID": submission_id,
            "PRISM_CODE_HASH": code_hash,
            "PRISM_ARCH_HASH": arch_hash,
            "PRISM_EXECUTION_BACKEND": backend,
            "PRISM_GPU_COUNT": str(self.settings.platform_eval_gpu_count),
        }
        if self.settings.platform_eval_gpu_type:
            env["PRISM_GPU_TYPE"] = self.settings.platform_eval_gpu_type
        return env


def _parse_metrics(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PRISM_METRICS_JSON="):
            payload = json.loads(line.removeprefix("PRISM_METRICS_JSON="))
            if not isinstance(payload, dict):
                raise RuntimeError("Prism container evaluation returned invalid metrics")
            return _normalize_metrics({str(key): float(value) for key, value in payload.items()})
    raise RuntimeError("Prism container evaluation did not return metrics")


def _normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if "q_arch" not in metrics:
        raise RuntimeError("Prism container evaluation did not return q_arch")
    metrics["q_arch"] = max(0.0, min(1.0, metrics["q_arch"]))
    metrics["q_recipe"] = max(0.0, min(1.0, metrics.get("q_recipe", 0.5)))
    if "train_loss" not in metrics and "final_loss" in metrics:
        metrics["train_loss"] = metrics["final_loss"]
    if "eval_loss" not in metrics and "val_loss" in metrics:
        metrics["eval_loss"] = metrics["val_loss"]
    if "val_loss" not in metrics and "eval_loss" in metrics:
        metrics["val_loss"] = metrics["eval_loss"]
    return metrics


def _entrypoint(files: tuple[SourceFile, ...]) -> str:
    for candidate in ("prism_submission.py", "model.py", "main.py"):
        match = next((file for file in files if file.path.endswith(candidate)), None)
        if match:
            return match.path
    python_file = next((file for file in files if file.path.endswith(".py")), None)
    return python_file.path if python_file else "model.py"


_CONTAINER_EVAL_SCRIPT = r"""
import dataclasses
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: runner.py payload.json")
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

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

project_root = Path("/workspace/project")
sys.path.insert(0, str(project_root))
entrypoint = project_root / payload.get("entrypoint", "model.py")
sys.path.insert(0, str(entrypoint.parent))
spec = importlib.util.spec_from_file_location("prism_submission", entrypoint)
if spec is None or spec.loader is None:
    raise RuntimeError("invalid Prism project entrypoint")
module = importlib.util.module_from_spec(spec)
sys.modules["prism_submission"] = module
spec.loader.exec_module(module)

ctx_data = payload.get("context", {})
ctx = PrismContext(
    vocab_size=int(ctx_data.get("vocab_size", 4096)),
    sequence_length=min(int(ctx_data.get("sequence_length", 128)), 128),
    max_parameters=int(ctx_data.get("max_parameters", 150_000_000)),
)
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
hooks_present = {
    name: callable(getattr(module, name, None))
    for name in (
        "configure_optimizer",
        "inference_logits",
        "infer",
        "compute_loss",
        "train_step",
    )
}
hook_usage = {
    "configure_optimizer": False,
    "inference_logits": False,
    "infer": False,
    "compute_loss": False,
    "train_step": False,
}

def prism_batch(t):
    return PrismBatch(tokens=t[:, :-1], targets=t[:, 1:], metadata={})

def logits_for(t):
    logits_hook = getattr(module, "inference_logits", None)
    infer_hook = getattr(module, "infer", None)
    if callable(logits_hook):
        hook_usage["inference_logits"] = True
        return logits_hook(model, prism_batch(t), ctx)
    if callable(infer_hook):
        hook_usage["infer"] = True
        return infer_hook(model, prism_batch(t), ctx)
    return model(t[:, :-1])

def loss_for(t):
    custom = getattr(module, "compute_loss", None)
    if callable(custom):
        hook_usage["compute_loss"] = True
        return custom(model, prism_batch(t), ctx)
    logits = logits_for(t)
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), t[:, 1:].reshape(-1) % vocab)

custom_opt = getattr(module, "configure_optimizer", None)
if callable(custom_opt):
    hook_usage["configure_optimizer"] = True
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
        hook_usage["train_step"] = True
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
    "train_loss": final_loss,
    "eval_loss": final_loss,
    "val_loss": final_loss,
    "parameter_count": float(params),
    "hook.configure_optimizer.present": float(hooks_present["configure_optimizer"]),
    "hook.inference_logits.present": float(hooks_present["inference_logits"]),
    "hook.infer.present": float(hooks_present["infer"]),
    "hook.compute_loss.present": float(hooks_present["compute_loss"]),
    "hook.train_step.present": float(hooks_present["train_step"]),
    "hook.configure_optimizer.used": float(hook_usage["configure_optimizer"]),
    "hook.inference_logits.used": float(hook_usage["inference_logits"]),
    "hook.infer.used": float(hook_usage["infer"]),
    "hook.compute_loss.used": float(hook_usage["compute_loss"]),
    "hook.train_step.used": float(hook_usage["train_step"]),
}
print("PRISM_METRICS_JSON=" + json.dumps(metrics, separators=(",", ":")))
"""
