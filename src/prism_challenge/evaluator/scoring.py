from __future__ import annotations

from dataclasses import dataclass

from .interface import TrainingRecipe


@dataclass(frozen=True)
class ScoreResult:
    q_arch: float
    q_recipe: float
    final_score: float


def score_recipe(recipe: TrainingRecipe) -> float:
    lr_score = 1.0 if 1e-5 <= recipe.learning_rate <= 3e-3 else 0.4
    batch_score = 1.0 if 1 <= recipe.batch_size <= 64 else 0.5
    opt_score = 1.0 if recipe.optimizer.lower() in {"adamw", "adam", "sgd"} else 0.5
    return max(0.0, min(1.0, 0.45 * lr_score + 0.35 * batch_score + 0.2 * opt_score))


def final_score(
    *,
    q_arch: float,
    q_recipe: float,
    anti_cheat_multiplier: float,
    diversity_bonus: float,
    penalty: float,
    arch_weight: float = 0.7,
    recipe_weight: float = 0.3,
) -> ScoreResult:
    base = arch_weight * q_arch + recipe_weight * q_recipe
    score = max(0.0, base * anti_cheat_multiplier + diversity_bonus - penalty)
    return ScoreResult(q_arch, q_recipe, score)
