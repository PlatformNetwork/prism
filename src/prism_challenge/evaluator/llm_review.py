from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .review_rules import ReviewRule, rules_prompt
from .schemas import DeterministicEvidence

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

SAFETY_REVIEW_SYSTEM = (
    "You are a strict security and integrity reviewer and a HARD GATE for a Platform "
    "Network 'ability-to-learn' ML subnet. Each submission is TWO scripts -- architecture.py "
    "(the model factory) and training.py (the miner's from-scratch learning loop) -- "
    "concatenated below with `# file:` headers; you MUST review BOTH scripts together. "
    "You must call SubmitMermaid first, then SubmitVerdict exactly once. verdict=true ALLOWS "
    "the submission to proceed to GPU evaluation; verdict=false REJECTS it. A reject is "
    "TERMINAL and stops the pipeline before any GPU work -- it does NOT require a line number "
    "or a 64-char evidence hash, so reject on a clear, human-readable reason whenever the "
    "bundle is unsafe, incoherent, or cheating. When in doubt about safety, reject."
)

SAFETY_REVIEW_INSTRUCTIONS = (
    "Review BOTH scripts together and REJECT (verdict=false) if ANY of the following holds:\n"
    "1. Architecture<->training INCOHERENCE: training.py is not a sensible, real learning "
    "procedure for the architecture.py model (trains a different model than the one declared, "
    "optimizer never steps the model's parameters, loss computed on unrelated tensors, or an "
    "architecture/recipe mismatch).\n"
    "2. Hidden no-op / dead / fake training loop: empty loop body, gradients detached, "
    "optimizer.step() never reached, immediate return, or sleeping to burn the budget without "
    "actually learning.\n"
    "3. Obfuscated code: base64/hex-encoded blobs decoded at runtime, single-letter "
    "indirection to hide intent, or string-built attribute access designed to evade the static "
    "AST sandbox.\n"
    "4. Smuggled / pretrained weights: embedding a weight blob and loading it, "
    "torch.load/from_pretrained of an external/untrusted path, or reconstructing weights from "
    "constants (the validator forces random init, so any pretrained load is a cheat).\n"
    "5. Hidden network / exfiltration: sockets, requests/urllib/httpx, or reading secrets "
    "(/etc/passwd, .ssh, API keys).\n"
    "6. Security-risky dangerous ops the static sandbox might miss: os.system, subprocess, "
    "eval/exec/compile, ctypes, dynamic importlib, or filesystem writes outside artifacts_dir.\n"
    "7. Metric gaming: writing its own prism_run_manifest, monkeypatching the loss-capture "
    "hook, reporting fabricated low loss, or hardcoding outputs to fake a low bits-per-byte.\n"
    "8. RESIDUAL INDIRECTION the static sandbox CANNOT statically resolve -- treat as "
    "obfuscation and REJECT: any variable-driven / non-constant-foldable getattr or setattr "
    "that targets a blocked symbol, e.g. `n = 'load'; getattr(torch, n)(...)`, "
    "`getattr(torch, some_var)`, `setattr(ctx, name, ...)`, or operator.attrgetter built from a "
    "runtime value. A literal, constant-foldable getattr/setattr is fine; a name assembled or "
    "chosen at runtime to reach a forbidden op is NOT.\n"
)


def build_safety_review_prompt(*, subject: str, rules_text: str, code: str) -> str:
    return (
        f"Review this {subject} (two scripts: architecture.py + training.py) as a hard gate.\n\n"
        f"Dynamic subnet rules:\n{rules_text}\n\n"
        f"{SAFETY_REVIEW_INSTRUCTIONS}\n"
        "If the bundle is a benign, coherent learner with none of the above, ALLOW it "
        "(verdict=true).\n\n"
        f"Source (both scripts):\n```python\n{code[:80_000]}\n```"
    )


class SubmitMermaid(BaseModel):
    mermaid: str = Field(min_length=1, description="Readable Mermaid diagram of reviewed logic.")
    notes: str = Field(default="", description="Short review notes for the diagram.")


class ReviewEvidence(BaseModel):
    """Tolerant tool-input schema for LLM-reported evidence.

    Intentionally loose: real model output omits line numbers and cannot
    compute a 64-char SHA-256 snippet_hash, so requiring them here fails closed
    and rejects every submission. Do NOT tighten. Hard rejection is still gated
    by the strict DeterministicEvidence re-validation in _as_evidence_payload;
    imperfect citations quarantine (held) rather than approve.
    """

    rule_id: str = Field(min_length=1)
    artifact_path: str = Field(default="submission.py", min_length=1)
    line: int | None = Field(default=None, ge=1)
    ast_node: str | None = Field(default=None, min_length=1)
    snippet_hash: str | None = Field(default=None)
    explanation: str = Field(min_length=1)


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
    evidence: list[ReviewEvidence] = Field(
        default_factory=list,
        description=(
            "Deterministic evidence required for rejection; suspicion without it quarantines."
        ),
    )


@dataclass(frozen=True)
class LlmReviewConfig:
    enabled: bool = True
    required: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    model: str | None = "openai/gpt-4o"
    api_key: str | None = None
    api_key_file: str | Path | None = "/run/secrets/openrouter_api_key"
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
    mermaid: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    held: bool = False


@dataclass
class PlagiarismReview:
    copied: bool
    reason: str
    violations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    mermaid: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    held: bool = False


def review_code(
    code: str,
    *,
    config: LlmReviewConfig | None = None,
    rules: tuple[ReviewRule, ...] = (),
    subject: str = "Prism model",
) -> LlmReview:
    config = config or LlmReviewConfig()
    static_evidence = _static_evidence(code)
    static_violations = sorted({item.explanation for item in static_evidence})
    if static_violations:
        evidence_payload = [item.model_dump(mode="json") for item in static_evidence]
        return LlmReview(
            False,
            "submission violates Prism safety rules",
            static_violations,
            [0, 0],
            raw={"evidence": evidence_payload},
            evidence=evidence_payload,
        )
    if not config.enabled:
        if config.required:
            return LlmReview(
                False, "real LLM review is required but disabled", ["llm_review_disabled"], [0, 0]
            )
        return LlmReview(True, "submission passed deterministic safety review", [], [1, 1])
    try:
        review = _invoke_review_flow(
            config,
            system=SAFETY_REVIEW_SYSTEM,
            prompt=build_safety_review_prompt(
                subject=subject,
                rules_text=rules_prompt(rules),
                code=code,
            ),
        )
    except Exception as exc:
        return LlmReview(False, f"LLM review failed closed: {exc}", ["llm_review_failed"], [0, 0])
    verdict = review["verdict"]
    approved = bool(verdict["verdict"])
    evidence = _as_evidence_payload(verdict.get("evidence"))
    # Hard gate: a model verdict is authoritative. A reject is TERMINAL and never downgraded
    # to a hold for lacking a 64-char evidence hash (the inverted evidence-gating).
    return LlmReview(
        approved=approved,
        reason=str(verdict["reason"]),
        violations=_as_list(verdict.get("violations")) + _as_list(verdict.get("rule_ids")),
        scores=[1, 1] if approved else [0, 0],
        confidence=float(verdict.get("confidence") or 0.0),
        raw=review,
        mermaid=str(review["mermaid"].get("mermaid") or ""),
        evidence=evidence,
        held=False,
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
        review = _invoke_review_flow(
            config,
            system=(
                "You are a plagiarism adjudicator for a Platform Network subnet. "
                "You must call SubmitMermaid first, then SubmitVerdict exactly once. "
                "For this plagiarism review, "
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
    verdict = review["verdict"]
    copied = bool(verdict["verdict"])
    evidence = _as_evidence_payload(verdict.get("evidence"))
    held = copied and not evidence
    return PlagiarismReview(
        copied=copied and not held,
        reason=str(verdict["reason"])
        if not held
        else f"LLM plagiarism suspicion without evidence: {verdict['reason']}",
        violations=_as_list(verdict.get("violations")) + _as_list(verdict.get("rule_ids")),
        confidence=float(verdict.get("confidence") or 0.0),
        raw=review,
        mermaid=str(review["mermaid"].get("mermaid") or ""),
        evidence=evidence,
        held=held,
    )


def _invoke_verdict(config: LlmReviewConfig, *, system: str, prompt: str) -> dict[str, Any]:
    return _invoke_review_flow(config, system=system, prompt=prompt)["verdict"]


def _invoke_review_flow(config: LlmReviewConfig, *, system: str, prompt: str) -> dict[str, Any]:
    api_key = _api_key(config)
    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured")
    if not config.model:
        raise RuntimeError("OpenRouter model is not configured")
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
    mermaid = _forced_tool_call(chat, SubmitMermaid, [("system", system), ("user", prompt)])
    verdict = _forced_tool_call(
        chat,
        SubmitVerdict,
        [
            ("system", system),
            ("user", prompt),
            ("assistant", f"Reviewed logic diagram:\n{mermaid.mermaid}\nNotes: {mermaid.notes}"),
            ("user", "Now call SubmitVerdict exactly once with your final verdict."),
        ],
    )
    return {
        "mermaid": mermaid.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
        "tool_order": ["SubmitMermaid", "SubmitVerdict"],
    }


def _forced_tool_call(chat: Any, tool: type[BaseModel], messages: list[tuple[str, str]]) -> Any:
    # The OpenRouter chat model emits one tool call per turn, so the two tools are forced
    # sequentially via tool_choice; strict=False keeps ReviewEvidence's tolerant optionals
    # provider-valid.
    bound = chat.bind_tools([tool], tool_choice=tool.__name__, strict=False)
    message = bound.invoke(messages)
    tool_calls = getattr(message, "tool_calls", None) or []
    match = next((call for call in tool_calls if _tool_name(call) == tool.__name__), None)
    if match is None:
        raise RuntimeError(f"llm_review_order_error: model did not call {tool.__name__}")
    return tool.model_validate(_tool_args(match))


def _tool_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", ""))


def _tool_args(call: Any) -> Any:
    if isinstance(call, dict):
        return call.get("args") or {}
    return getattr(call, "args", None) or {}


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


def _static_evidence(code: str) -> list[DeterministicEvidence]:
    evidence: list[DeterministicEvidence] = []
    for pattern, reason in REJECTION_PATTERNS:
        match = pattern.search(code)
        if match is None:
            continue
        line = code.count("\n", 0, match.start()) + 1
        snippet = code.splitlines()[line - 1] if code.splitlines() else match.group(0)
        evidence.append(
            DeterministicEvidence(
                rule_id=_rule_id(reason),
                artifact_path="submission.py",
                line=line,
                snippet_hash=sha256(snippet.encode("utf-8")).hexdigest(),
                explanation=reason,
            )
        )
    return evidence


def _rule_id(reason: str) -> str:
    return "prism:llm-review:" + re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")


def _as_evidence_payload(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        value = [value]
    payload: list[dict[str, Any]] = []
    for item in value:
        try:
            deterministic = DeterministicEvidence.model_validate(item)
        except ValidationError:
            continue
        payload.append(deterministic.model_dump(mode="json"))
    return payload


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
