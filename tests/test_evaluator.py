from __future__ import annotations

import pytest
from conftest import VALID_CODE

from prism_challenge.evaluator.anti_cheat import ast_similarity, evaluate_anti_cheat
from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.interface import PrismContext, TrainingRecipe
from prism_challenge.evaluator.l1_syntax import validate_l1
from prism_challenge.evaluator.l2_proxy import score_l2
from prism_challenge.evaluator.l3_train import kendall_tau, score_l3
from prism_challenge.evaluator.l4_benchmark import score_l4
from prism_challenge.evaluator.lium_client import LiumClient
from prism_challenge.evaluator.sandbox import (
    SandboxViolation,
    inspect_code,
    load_submission_contract,
)
from prism_challenge.evaluator.scoring import final_score, score_recipe


def test_sandbox_contract_and_l1():
    ctx = PrismContext(max_parameters=1_000_000)
    model, recipe, report = load_submission_contract(VALID_CODE, ctx)
    assert model is not None
    assert isinstance(recipe, TrainingRecipe)
    assert "Import" in report.ast_fingerprint
    result = validate_l1(VALID_CODE, ctx)
    assert result.valid
    assert result.parameter_count > 0


def test_sandbox_blocks_forbidden_imports():
    with pytest.raises(SandboxViolation):
        inspect_code("import os\ndef build_model(ctx): pass\ndef get_recipe(ctx): pass")


def test_proxy_train_and_final_scoring():
    ctx = PrismContext(max_parameters=1_000_000)
    l2 = score_l2(VALID_CODE, ctx)
    l3 = score_l3(VALID_CODE, ctx)
    assert 0 <= l2.q_proxy <= 1
    assert -1 <= kendall_tau([1, 2, 3], [1, 3, 2]) <= 1
    score = final_score(
        q_arch=l3.q_train,
        q_recipe=score_recipe(TrainingRecipe()),
        anti_cheat_multiplier=1,
        diversity_bonus=0.01,
        penalty=0,
    )
    assert score.final_score > 0


async def test_l4_fake_lium():
    result = await score_l4(
        VALID_CODE,
        PrismContext(vocab_size=256, sequence_length=16, max_parameters=1_000_000),
        LiumClient(base_url=None, token=None),
        BenchConfig(
            train_steps=1,
            seeds=(1,),
            sequence_lengths=(16,),
            batch_size=1,
            vocab_size=256,
        ),
    )
    assert result.q_arch >= 0
    assert "q_arch" in result.metrics


def test_anti_cheat_similarity_and_diversity():
    assert ast_similarity(VALID_CODE, VALID_CODE) == 1.0
    result = evaluate_anti_cheat(VALID_CODE, [VALID_CODE])
    assert result.multiplier < 1
    assert result.findings


def test_novelty_bonus_is_quality_gated():
    weak_novel = final_score(
        q_arch=0.05,
        q_recipe=1.0,
        anti_cheat_multiplier=1.0,
        diversity_bonus=0.05,
        penalty=0.0,
    )
    stronger_old = final_score(
        q_arch=0.12,
        q_recipe=1.0,
        anti_cheat_multiplier=1.0,
        diversity_bonus=0.0,
        penalty=0.0,
    )
    assert stronger_old.final_score > weak_novel.final_score
