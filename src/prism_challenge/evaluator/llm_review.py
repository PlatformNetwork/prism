from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .review_rules import ReviewRule, rules_prompt

REJECTION_PATTERNS = [
    (
        re.compile(r"\b(os\.system|subprocess|socket|requests|httpx|paramiko)\b"),
        "network/process escape attempt",
    ),
    (re.compile(r"\b(open\s*\(|eval\s*\(|exec\s*\()"), "unsafe file or dynamic code use"),
    (
        re.compile(r"/etc/passwd|\.ssh|AWS_SECRET|OPENAI_API_KEY", re.I),
        "secret exfiltration indicator",
    ),
]


class SubmitVerdict(BaseModel):
    """Submit the final review verdict. Fill reason before verdict."""

    reason: str = Field(description="Reason for the verdict. This field must be populated first.")
    verdict: bool = Field(
        description=(
            "For safety review: true approves. "
            "For plagiarism review: true means copied/plagiarized."
        )
    )
    violations: list[str] = Field(default_factory=list, description="Rule ids or violation labels.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rule_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class LlmReviewConfig:
    enabled: bool = False
    required: bool = False
    base_url: str = "https://llm.chutes.ai/v1"
    model: str | None = None
    api_key: str | None = None
    api_key_file: str | Path | None = None
    timeout_seconds: int = 60
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 1


@dataclass
class LlmReview:
    approved: bool
    reason: str
    violations: list[str] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlagiarismReview:
    copied: bool
    reason: str
    violations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def review_code(
    code: str,
    *,
    config: LlmReviewConfig | None = None,
    rules: tuple[ReviewRule, ...] = (),
    subject: str = "Prism model",
) -> LlmReview:
    config = config or LlmReviewConfig()
    static_violations = _static_violations(code)
    if static_violations:
        return LlmReview(False, "submission violates Prism safety rules", static_violations, [0, 0])
    if not config.enabled:
        if config.required:
            return LlmReview(
                False, "real LLM review is required but disabled", ["llm_review_disabled"], [0, 0]
            )
        return LlmReview(True, "submission passed deterministic safety review", [], [1, 1])
    try:
        verdict = _invoke_verdict(
            config,
            system=(
                "You are a strict security reviewer for a Platform Network subnet. "
                "You must call SubmitVerdict exactly once. For this safety review, "
                "verdict=true approves the code and verdict=false rejects it."
            ),
            prompt=(
                f"Review this {subject} for rule compliance.\n\n"
                f"Dynamic subnet rules:\n{rules_prompt(rules)}\n\n"
                "Reject code that exfiltrates secrets, opens process/network escapes, "
                "hides payloads, or violates the dynamic subnet rules.\n\n"
                f"Source:\n```python\n{code[:80_000]}\n```"
            ),
        )
    except Exception as exc:
        return LlmReview(False, f"LLM review failed closed: {exc}", ["llm_review_failed"], [0, 0])
    approved = bool(verdict["verdict"])
    return LlmReview(
        approved=approved,
        reason=str(verdict["reason"]),
        violations=_as_list(verdict.get("violations")) + _as_list(verdict.get("rule_ids")),
        scores=[1, 1] if approved else [0, 0],
        confidence=float(verdict.get("confidence") or 0.0),
        raw=verdict,
    )


def review_plagiarism(
    *,
    current_code: str,
    candidate_code: str,
    comparison_report: dict[str, Any],
    config: LlmReviewConfig | None = None,
    rules: tuple[ReviewRule, ...] = (),
) -> PlagiarismReview:
    config = config or LlmReviewConfig()
    if not config.enabled:
        if config.required:
            return PlagiarismReview(
                True,
                "real plagiarism LLM review is required but disabled",
                ["llm_review_disabled"],
            )
        return PlagiarismReview(False, "LLM plagiarism review disabled")
    comparison_json = json.dumps(comparison_report, sort_keys=True)[:40_000]
    try:
        verdict = _invoke_verdict(
            config,
            system=(
                "You are a plagiarism adjudicator for a Platform Network subnet. "
                "You must call SubmitVerdict exactly once. For this plagiarism review, "
                "verdict=true means the current submission is copied or plagiarized "
                "from the candidate."
            ),
            prompt=(
                f"Dynamic subnet rules:\n{rules_prompt(rules)}\n\n"
                "Decide whether the current code is copied or trivially derived "
                "from the candidate. Use the static comparison report first, "
                "then inspect both code excerpts.\n\n"
                f"Comparison report:\n```json\n{comparison_json}\n```\n\n"
                f"Current code:\n```python\n{current_code[:60_000]}\n```\n\n"
                f"Candidate code:\n```python\n{candidate_code[:60_000]}\n```"
            ),
        )
    except Exception as exc:
        return PlagiarismReview(
            True, f"LLM plagiarism review failed closed: {exc}", ["llm_review_failed"]
        )
    return PlagiarismReview(
        copied=bool(verdict["verdict"]),
        reason=str(verdict["reason"]),
        violations=_as_list(verdict.get("violations")) + _as_list(verdict.get("rule_ids")),
        confidence=float(verdict.get("confidence") or 0.0),
        raw=verdict,
    )


def _invoke_verdict(config: LlmReviewConfig, *, system: str, prompt: str) -> dict[str, Any]:
    api_key = _api_key(config)
    if not api_key:
        raise RuntimeError("Chutes API key is not configured")
    if not config.model:
        raise RuntimeError("Chutes model is not configured")
    chat_openai = _load_chat_openai()
    chat = chat_openai(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        max_tokens=config.max_tokens,
    )
    bound = chat.bind_tools([SubmitVerdict], strict=True)
    message = bound.invoke([("system", system), ("user", prompt)])
    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) != 1:
        raise RuntimeError("model did not return exactly one SubmitVerdict tool call")
    call = tool_calls[0]
    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
    verdict = SubmitVerdict.model_validate(args or {})
    return verdict.model_dump()


def _api_key(config: LlmReviewConfig) -> str | None:
    if config.api_key:
        return config.api_key
    if config.api_key_file:
        path = Path(config.api_key_file)
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            return token or None
    return None


def _load_chat_openai() -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain-openai is not installed") from exc
    return ChatOpenAI


def _static_violations(code: str) -> list[str]:
    return sorted({reason for pattern, reason in REJECTION_PATTERNS if pattern.search(code)})


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
