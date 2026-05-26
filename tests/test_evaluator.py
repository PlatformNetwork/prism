from __future__ import annotations

import base64
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from conftest import VALID_CODE

from prism_challenge.evaluator import llm_review, source_similarity
from prism_challenge.evaluator.anti_cheat import ast_similarity, evaluate_anti_cheat
from prism_challenge.evaluator.bench_config import BenchConfig
from prism_challenge.evaluator.checkpoints import checkpoint_workspace
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
from prism_challenge.evaluator.training import (
    train_language_model,
    training_recipe_fingerprint,
)

CHECKPOINT_CODE = (
    VALID_CODE
    + """

def save_checkpoint(model, checkpoint_dir, ctx):
    torch.save(
        {"state_dict": model.state_dict(), "marker": torch.tensor([7])},
        checkpoint_dir / "model.pt",
    )
    return {"path": "model.pt", "metadata": {"marker": 7}}

def load_checkpoint(model, checkpoint_dir, ctx):
    payload = torch.load(checkpoint_dir / "model.pt", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return {"marker": int(payload["marker"].item())}
"""
)


def _checkpoint_cfg() -> BenchConfig:
    return BenchConfig(
        train_steps=1,
        eval_steps=1,
        sequence_lengths=(8,),
        batch_size=1,
        max_tokens=16,
    )


def _checkpoint_ctx(
    artifact_output,
    *,
    attempt: int,
    checkpoint_dir,
    resume_checkpoint_dir=None,
    submission_id: str = "sub-1",
    code_hash: str = "c" * 64,
    arch_hash: str = "a" * 64,
) -> PrismContext:
    return PrismContext(
        vocab_size=256,
        sequence_length=8,
        max_parameters=50_000,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_dir=resume_checkpoint_dir,
        attempt=attempt,
        checkpoint_metadata={
            "artifact_output_path": str(artifact_output),
            "submission_id": submission_id,
            "code_hash": code_hash,
            "arch_hash": arch_hash,
            "recipe_fingerprint": training_recipe_fingerprint(
                TrainingRecipe(learning_rate=0.0003, batch_size=4)
            ),
        },
    )


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


def test_sandbox_accepts_existing_miner_without_checkpoint_hooks():
    report = inspect_code(VALID_CODE)
    assert "function:build_model" in report.ast_fingerprint
    assert "function:get_recipe" in report.ast_fingerprint
    assert "function:save_checkpoint" not in report.ast_fingerprint
    assert "function:load_checkpoint" not in report.ast_fingerprint


def test_prism_context_exposes_checkpoint_defaults():
    ctx = PrismContext()
    assert ctx.checkpoint_dir is None
    assert ctx.resume_checkpoint_dir is None
    assert ctx.checkpoint_api_version == 1
    assert ctx.attempt == 1
    assert ctx.is_resume is False
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.world_size == 1
    assert ctx.distributed_backend is None
    assert ctx.device == "cpu"
    assert ctx.checkpoint_metadata == {}


def test_sandbox_accepts_valid_checkpoint_hooks():
    code = (
        VALID_CODE
        + """

def save_checkpoint(model, checkpoint_dir, ctx):
    return None

def load_checkpoint(model, checkpoint_dir, ctx):
    return None
"""
    )
    report = inspect_code(code)
    assert "function:save_checkpoint" in report.ast_fingerprint
    assert "function:load_checkpoint" in report.ast_fingerprint


def test_local_training_checkpoint_save_writes_metadata(tmp_path):
    artifact_output = tmp_path / "artifacts"
    workspace = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=1)
    ctx = _checkpoint_ctx(
        artifact_output,
        attempt=1,
        checkpoint_dir=workspace.current_dir,
    )

    run = train_language_model(
        CHECKPOINT_CODE,
        ctx,
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )

    metadata_path = workspace.current_dir / "checkpoint_metadata.v1.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert run.checkpoint_path == "checkpoints/sub-1/attempt-1/current/model.pt"
    assert run.checkpoint_metadata_path == (
        "checkpoints/sub-1/attempt-1/current/checkpoint_metadata.v1.json"
    )
    assert (workspace.current_dir / "model.pt").is_file()
    assert payload["checkpoint_api_version"] == 1
    assert payload["submission_id"] == "sub-1"
    assert payload["attempt"] == 1
    assert payload["code_hash"] == "c" * 64
    assert payload["arch_hash"] == "a" * 64
    assert payload["checkpoint_path"] == "checkpoints/sub-1/attempt-1/current/model.pt"
    assert payload["hook_return"] == {"path": "model.pt", "metadata": {"marker": 7}}
    assert payload["world_size"] == 1
    assert payload["rank_writer"] == 0
    assert payload["checkpoint_dir"] == "checkpoints/sub-1/attempt-1/current"
    assert payload["bytes_total"] > 0


def test_local_training_checkpoint_resume_invokes_load_hook(tmp_path):
    artifact_output = tmp_path / "artifacts"
    first = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=1)
    train_language_model(
        CHECKPOINT_CODE,
        _checkpoint_ctx(artifact_output, attempt=1, checkpoint_dir=first.current_dir),
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )
    second = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=2)
    ctx = _checkpoint_ctx(
        artifact_output,
        attempt=2,
        checkpoint_dir=second.current_dir,
        resume_checkpoint_dir=second.resume_dir,
    )

    run = train_language_model(
        CHECKPOINT_CODE,
        ctx,
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )

    assert run.resume_checkpoint_path == "checkpoints/sub-1/attempt-1/current/model.pt"
    assert run.load_checkpoint_metadata == {"marker": 7}
    assert run.checkpoint_path == "checkpoints/sub-1/attempt-2/current/model.pt"


def test_local_training_checkpoint_resume_rejects_missing_or_corrupt_metadata(tmp_path):
    artifact_output = tmp_path / "artifacts"
    first = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=1)
    first.current_dir.mkdir(parents=True)
    second = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=2)
    ctx = _checkpoint_ctx(
        artifact_output,
        attempt=2,
        checkpoint_dir=second.current_dir,
        resume_checkpoint_dir=second.resume_dir,
    )

    with pytest.raises(ValueError, match="missing metadata"):
        train_language_model(CHECKPOINT_CODE, ctx, _checkpoint_cfg(), seed=1)

    (first.current_dir / "checkpoint_metadata.v1.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        train_language_model(CHECKPOINT_CODE, ctx, _checkpoint_cfg(), seed=1)


def test_local_training_checkpoint_resume_rejects_wrong_version_and_provenance(tmp_path):
    artifact_output = tmp_path / "artifacts"
    first = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=1)
    train_language_model(
        CHECKPOINT_CODE,
        _checkpoint_ctx(artifact_output, attempt=1, checkpoint_dir=first.current_dir),
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )
    metadata_path = first.current_dir / "checkpoint_metadata.v1.json"
    original = json.loads(metadata_path.read_text(encoding="utf-8"))
    second = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=2)
    ctx = _checkpoint_ctx(
        artifact_output,
        attempt=2,
        checkpoint_dir=second.current_dir,
        resume_checkpoint_dir=second.resume_dir,
    )

    wrong_version = {**original, "checkpoint_api_version": 2}
    metadata_path.write_text(json.dumps(wrong_version), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        train_language_model(CHECKPOINT_CODE, ctx, _checkpoint_cfg(), seed=1)

    wrong_submission = {**original, "submission_id": "other-submission"}
    metadata_path.write_text(json.dumps(wrong_submission), encoding="utf-8")
    with pytest.raises(ValueError, match="submission_id"):
        train_language_model(CHECKPOINT_CODE, ctx, _checkpoint_cfg(), seed=1)


def test_local_training_existing_miner_without_checkpoint_hooks_still_runs():
    run = train_language_model(
        VALID_CODE,
        PrismContext(vocab_size=256, sequence_length=8, max_parameters=50_000),
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )

    assert run.tokens_seen > 0
    assert run.checkpoint_path is None
    assert run.resume_checkpoint_path is None


def test_local_training_resume_requires_load_checkpoint_hook(tmp_path):
    artifact_output = tmp_path / "artifacts"
    first = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=1)
    train_language_model(
        CHECKPOINT_CODE,
        _checkpoint_ctx(artifact_output, attempt=1, checkpoint_dir=first.current_dir),
        _checkpoint_cfg(),
        seed=1,
        texts=["alpha beta gamma delta"],
    )
    second = checkpoint_workspace(artifact_output, submission_id="sub-1", attempt=2)
    ctx = _checkpoint_ctx(
        artifact_output,
        attempt=2,
        checkpoint_dir=second.current_dir,
        resume_checkpoint_dir=second.resume_dir,
    )

    with pytest.raises(ValueError, match="load_checkpoint hook is absent"):
        train_language_model(VALID_CODE, ctx, _checkpoint_cfg(), seed=1)


@pytest.mark.parametrize(
    ("hook_source", "match"),
    [
        (
            "def save_checkpoint(model, ctx):\n    return None\n",
            "save_checkpoint must have signature",
        ),
        (
            "def load_checkpoint(model, checkpoint_dir, context):\n    return None\n",
            "load_checkpoint must have signature",
        ),
        (
            "def save_checkpoint(*args, **kwargs):\n    return None\n",
            "save_checkpoint may not use",
        ),
    ],
)
def test_sandbox_rejects_invalid_checkpoint_hook_signatures(hook_source, match):
    with pytest.raises(SandboxViolation, match=match):
        inspect_code(f"{VALID_CODE}\n{hook_source}")


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
            assert [tool.__name__ for tool in tools] == ["SubmitMermaid", "SubmitVerdict"]
            return self

        def invoke(self, _messages):
            return SimpleNamespace(
                tool_calls=[
                    {
                        "name": "SubmitMermaid",
                        "args": {
                            "mermaid": "flowchart LR\n  A[Model] --> B[Logits]",
                            "notes": "valid architecture summary",
                        },
                    },
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
