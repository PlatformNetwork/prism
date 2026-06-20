from __future__ import annotations

from hashlib import sha256

import pytest

from prism_challenge.evaluator import llm_review as llm
from prism_challenge.evaluator.llm_review import (
    LlmReviewConfig,
    SubmitVerdict,
    review_code,
)

_SAFE_CODE = "def build_model(ctx):\n    return None\n"


def _flow(verdict: dict[str, object]):
    def _invoke(config, *, system, prompt):
        return {
            "mermaid": {"mermaid": "flowchart LR\n  A[Source] --> B[Review]"},
            "verdict": verdict,
            "tool_order": ["SubmitMermaid", "SubmitVerdict"],
        }

    return _invoke


def test_submit_verdict_accepts_locatorless_evidence():
    verdict = SubmitVerdict.model_validate(
        {
            "reason": "defines build_model and get_recipe; no escapes detected",
            "verdict": True,
            "violations": [],
            "confidence": 0.92,
            "rule_ids": [],
            "evidence": [
                {
                    "rule_id": "prism:safe",
                    "artifact_path": "submission.py",
                    "explanation": "no violations observed in the reviewed source",
                }
            ],
        }
    )
    assert verdict.verdict is True
    assert verdict.evidence[0].line is None


def test_wellformed_approval_passes_review(monkeypatch):
    verdict = {
        "reason": "defines build_model and get_recipe; no escapes detected",
        "verdict": True,
        "violations": [],
        "confidence": 0.92,
        "rule_ids": [],
        "evidence": [
            {
                "rule_id": "prism:safe",
                "artifact_path": "submission.py",
                "explanation": "no violations observed in the reviewed source",
            }
        ],
    }
    monkeypatch.setattr(llm, "_invoke_review_flow", _flow(verdict))

    review = review_code(
        _SAFE_CODE,
        config=LlmReviewConfig(enabled=True, model="gpt-4o-mini", api_key="test"),
    )

    assert review.approved is True
    assert review.held is False
    assert review.reason == "defines build_model and get_recipe; no escapes detected"


def test_deterministic_evidence_still_rejects(monkeypatch):
    evidence = {
        "rule_id": "subnet:no-escape",
        "artifact_path": "submission.py",
        "line": 3,
        "snippet_hash": sha256(b"os.system('curl bad')").hexdigest(),
        "explanation": "process escape attempt present in submitted source",
    }
    verdict = {
        "reason": "deterministic escape evidence found",
        "verdict": False,
        "violations": ["subnet:no-escape"],
        "confidence": 0.96,
        "rule_ids": [],
        "evidence": [evidence],
    }
    monkeypatch.setattr(llm, "_invoke_review_flow", _flow(verdict))

    review = review_code(
        _SAFE_CODE,
        config=LlmReviewConfig(enabled=True, model="gpt-4o-mini", api_key="test"),
    )

    assert review.approved is False
    assert review.held is False
    assert len(review.evidence) == 1


def test_reject_without_deterministic_evidence_is_terminal(monkeypatch):
    # Inverted evidence-gating (hard gate): a model `reject` is TERMINAL even when it cannot
    # cite a precise locator / 64-char snippet hash -- it is never downgraded to a hold.
    verdict = {
        "reason": "suspects hidden behavior but cannot cite a precise location",
        "verdict": False,
        "violations": ["suspicion"],
        "confidence": 0.55,
        "rule_ids": [],
        "evidence": [
            {
                "rule_id": "suspicion",
                "artifact_path": "submission.py",
                "explanation": "maybe obfuscated, no concrete locator",
            }
        ],
    }
    monkeypatch.setattr(llm, "_invoke_review_flow", _flow(verdict))

    review = review_code(
        _SAFE_CODE,
        config=LlmReviewConfig(enabled=True, model="gpt-4o-mini", api_key="test"),
    )

    assert review.approved is False
    assert review.held is False
    assert review.reason == "suspects hidden behavior but cannot cite a precise location"


@pytest.mark.parametrize("bad", [[{"rule_id": "x", "artifact_path": "s", "explanation": "e"}]])
def test_as_evidence_payload_drops_non_deterministic_items(bad):
    assert llm._as_evidence_payload(bad) == []
