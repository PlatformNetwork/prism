from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 4096
    sequence_length: int = 128
    max_layers: int = 96
    max_parameters: int = 150_000_000
    seed: int = 1337
    checkpoint_dir: Path | None = None
    resume_checkpoint_dir: Path | None = None
    checkpoint_api_version: int = 1
    attempt: int = 1
    is_resume: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed_backend: str | None = None
    device: str = "cpu"
    checkpoint_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 3e-4
    batch_size: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01


@dataclass(frozen=True)
class PrismBatch:
    tokens: Any
    targets: Any | None = None
    metadata: dict[str, Any] | None = None


class PrismModelModule(Protocol):
    def build_model(self, ctx: PrismContext) -> Any: ...

    def get_recipe(self, ctx: PrismContext) -> TrainingRecipe: ...

    def configure_optimizer(self, model: Any, recipe: TrainingRecipe, ctx: PrismContext) -> Any: ...

    def inference_logits(self, model: Any, batch: PrismBatch, ctx: PrismContext) -> Any: ...

    def compute_loss(self, model: Any, batch: PrismBatch, ctx: PrismContext) -> Any: ...

    def train_step(
        self, model: Any, batch: PrismBatch, optimizer: Any, ctx: PrismContext
    ) -> Any: ...

    def save_checkpoint(
        self, model: Any, checkpoint_dir: Path, ctx: PrismContext
    ) -> str | dict[str, object] | None: ...

    def load_checkpoint(
        self, model: Any, checkpoint_dir: Path, ctx: PrismContext
    ) -> dict[str, object] | None: ...


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Prism evaluation") from exc
