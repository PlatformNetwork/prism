from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 4096
    sequence_length: int = 128
    max_layers: int = 96
    max_parameters: int = 150_000_000
    seed: int = 1337


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


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Prism evaluation") from exc
