from __future__ import annotations

from copy import deepcopy

import pytest
from test_artifact_manifest import _valid_manifest

from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator.schemas import ExecutionMode
from prism_challenge.evaluator.scoring import (
    TRAINING_SCORE_COMPONENTS,
    RankedScore,
    final_score,
    rank_official_scores,
    score_training_manifest,
)
from prism_challenge.repository import PrismRepository
from prism_challenge.runtime_config import RuntimeConfigError


async def test_weights_sum_validation_blocks_official_scoring_config(tmp_path) -> None:
    database = Database(tmp_path / "weights.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=60)
    await repository.store_runtime_config(
        config_key="score_weights",
        value={"final_architecture_weight": 0.8, "final_recipe_weight": 0.8},
        updated_by="ops",
    )

    with pytest.raises(RuntimeConfigError, match="final_score weights"):
        await repository.runtime_config(PrismSettings(), official=True)


async def test_sql_score_weights_feed_final_score_formula(tmp_path) -> None:
    database = Database(tmp_path / "weights-override.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=60)
    await repository.store_runtime_config(
        config_key="score_weights",
        value={"final_architecture_weight": 0.2, "final_recipe_weight": 0.8},
        updated_by="ops",
    )
    runtime_config = await repository.runtime_config(PrismSettings(), official=True)

    scored = final_score(
        q_arch=1.0,
        q_recipe=0.5,
        anti_cheat_multiplier=1.0,
        diversity_bonus=0.0,
        penalty=0.0,
        arch_weight=runtime_config.score_weights.final_architecture_weight,
        recipe_weight=runtime_config.score_weights.final_recipe_weight,
    )

    assert scored.final_score == pytest.approx(0.6)


def test_training_score_uses_architecture_normalized_improvement_not_raw_final_loss() -> None:
    payload = _manifest_with_official_benchmarks()
    payload["metrics"]["final_loss"] = 99.0
    payload["metrics"]["loss"]["raw_final_loss"] = 99.0
    payload["metrics"]["loss"]["architecture_normalized_heldout_improvement"] = 0.42

    scored = score_training_manifest(payload)

    assert scored.component_weights == TRAINING_SCORE_COMPONENTS
    assert scored.component_values["architecture_normalized_heldout_improvement"] == pytest.approx(
        0.42
    )
    assert scored.components[0].weighted_contribution == pytest.approx(0.30 * 0.42)
    assert scored.details["raw_final_loss_used"] is False


async def test_component_weight_rows_normalize_60_40_from_canonical_architecture_scores(
    tmp_path,
) -> None:
    database = Database(tmp_path / "component-weights.sqlite3")
    await database.init()
    repository = PrismRepository(database, epoch_seconds=60)
    async with database.connect() as conn:
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
            "owner_submission_id, canonical_submission_id, q_arch_best, created_at, updated_at, "
            "canonical_graph_hash) VALUES "
            "('arch-a', 'fam-a', 'fp-a', 'beh-a', 'arch-owner-a', 'sub-a', 'sub-a', "
            "0.75, '2026-05-25T00:00:00+00:00', '2026-05-25T00:00:00+00:00', ?),"
            "('arch-b', 'fam-b', 'fp-b', 'beh-b', 'arch-owner-b', 'sub-b', 'sub-b', "
            "0.25, '2026-05-25T00:00:01+00:00', '2026-05-25T00:00:01+00:00', ?)",
            ("a" * 64, "b" * 64),
        )
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at) VALUES "
            "('train-a', 'arch-a', 'hash-a', 'trainer-a', 'train-sub-a', 0.10, 0.10, 0.0, "
            "1, '2026-05-25T00:00:02+00:00', '2026-05-25T00:00:02+00:00'),"
            "('train-b', 'arch-b', 'hash-b', 'trainer-b', 'train-sub-b', 1.00, 1.00, 0.0, "
            "1, '2026-05-25T00:00:03+00:00', '2026-05-25T00:00:03+00:00')"
        )

    rows = await repository.component_weight_rows(architecture_weight=0.60, training_weight=0.40)
    scores = {(row["component"], row["hotkey"]): row["score"] for row in rows}

    assert scores[("architecture", "arch-owner-a")] == pytest.approx(0.45)
    assert scores[("architecture", "arch-owner-b")] == pytest.approx(0.15)
    assert scores[("training", "trainer-a")] == pytest.approx(0.30)
    assert scores[("training", "trainer-b")] == pytest.approx(0.10)
    assert sum(float(row["score"]) for row in rows) == pytest.approx(1.0)


def test_tie_breakers_use_score_compute_time_then_submission_id() -> None:
    ranked = rank_official_scores(
        [
            RankedScore("z-sub", 0.8, 20.0, "2026-05-25T00:00:01+00:00"),
            RankedScore("a-sub", 0.8, 10.0, "2026-05-25T00:00:02+00:00"),
            RankedScore("b-sub", 0.8, 10.0, "2026-05-25T00:00:01+00:00"),
            RankedScore("c-sub", 0.9, 50.0, "2026-05-25T00:00:03+00:00"),
        ]
    )

    assert [score.submission_id for score in ranked] == ["c-sub", "b-sub", "a-sub", "z-sub"]


def _manifest_with_official_benchmarks(
    *,
    mode: str = ExecutionMode.GPU_PROXY_EVAL.value,
    submission_id: str = "submission-1",
    architecture_id: str = "architecture-1",
) -> dict:
    payload = deepcopy(_valid_manifest(mode))
    payload["submission_id"] = submission_id
    payload["architecture_id"] = architecture_id
    payload["metrics"]["benchmark_scores"] = {
        "gsm8k": 1.0,
        "math": 1.0,
        "arc_challenge": 1.0,
        "humaneval": 1.0,
        "mmlu": 1.0,
        "ifeval": 1.0,
        "truthfulqa": 1.0,
        "needle": 1.0,
    }
    return payload
