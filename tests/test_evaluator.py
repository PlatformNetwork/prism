from __future__ import annotations

import base64
import io
import zipfile
from types import SimpleNamespace

import pytest
from conftest import VALID_CODE

from prism_challenge.evaluator import llm_review, source_similarity
from prism_challenge.evaluator.anti_cheat import ast_similarity, evaluate_anti_cheat
from prism_challenge.evaluator.interface import PrismContext, TrainingRecipe
from prism_challenge.evaluator.l1_syntax import validate_l1
from prism_challenge.evaluator.l2_proxy import score_l2
from prism_challenge.evaluator.l3_train import kendall_tau, score_l3
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


def test_sandbox_blocks_dynamic_escape_patterns():
    with pytest.raises(SandboxViolation, match="forbidden attribute"):
        inspect_code(
            "def build_model(ctx):\n    return (1).__class__\ndef get_recipe(ctx):\n    return {}\n"
        )
    with pytest.raises(SandboxViolation, match="top-level code"):
        inspect_code("print('side effect')\ndef build_model(ctx): pass\ndef get_recipe(ctx): pass")


def test_sandbox_accepts_full_miner_training_and_inference_contract():
    code = (
        VALID_CODE
        + """

def configure_optimizer(model, recipe, ctx):
    return None

def inference_logits(model, batch, ctx):
    return model(batch.tokens)

def compute_loss(model, batch, ctx):
    return model(batch.tokens).sum()

def train_step(model, batch, optimizer, ctx):
    return compute_loss(model, batch, ctx)
"""
    )
    report = inspect_code(code)
    assert "function:train_step" in report.ast_fingerprint
    assert "function:inference_logits" in report.ast_fingerprint


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


def test_anti_cheat_similarity_and_diversity():
    assert ast_similarity(VALID_CODE, VALID_CODE) == 1.0
    result = evaluate_anti_cheat(VALID_CODE, [VALID_CODE])
    assert result.multiplier < 1
    assert result.findings


def test_prism_chutes_tool_call_review(monkeypatch):
    class FakeChat:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "model"

        def bind_tools(self, tools, strict=False):
            assert strict is True
            assert list(tools[0].model_fields)[:2] == ["reason", "verdict"]
            return self

        def invoke(self, _messages):
            return SimpleNamespace(
                tool_calls=[
                    {
                        "name": "SubmitVerdict",
                        "args": {
                            "reason": "valid Prism model",
                            "verdict": True,
                            "confidence": 0.8,
                        },
                    }
                ]
            )

    monkeypatch.setattr(llm_review, "_load_chat_openai", lambda: FakeChat)
    result = llm_review.review_code(
        VALID_CODE,
        config=llm_review.LlmReviewConfig(enabled=True, model="model", api_key="key"),
    )
    assert result.approved
    assert result.reason == "valid Prism model"


def test_prism_zip_snapshot_similarity():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("model.py", VALID_CODE)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    snapshot = source_similarity.snapshot_from_submission(encoded, "model.zip")
    assert source_similarity.primary_python_code(snapshot).strip().startswith("import torch")
    candidate = source_similarity.snapshot_from_submission(VALID_CODE, "model.py")
    ranked = source_similarity.rank_similar(
        snapshot,
        [{"submission_id": "old", "hotkey": "hk", **candidate.to_payload()}],
        min_similarity=0.1,
    )
    assert ranked and ranked[0].submission_id == "old"


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
