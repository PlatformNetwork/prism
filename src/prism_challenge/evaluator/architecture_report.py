"""Grounded LLM auto-report generator for the architecture lab.

Reuses the existing OpenRouter wiring in :mod:`prism_challenge.evaluator.llm_review` (the same
``LlmReviewConfig`` model, endpoint/credential resolution, and ``ChatOpenAI`` loader) so reports go
through the SAME gateway/provider-key boundary as the safety gate -- no new HTTP client and no
hardcoded keys. The prompt is built ONLY from persisted facts (name, owner, best final score,
prequential bpb, the reconciled compute profile, the loss-curve trend, and the variant count) so the
model has nothing to hallucinate from.
"""

from __future__ import annotations

from typing import Any

from ..config import PrismSettings
from .llm_review import LlmReviewConfig, _load_chat_openai, _resolve_endpoint

REPORT_SYSTEM = (
    "You are a concise, rigorous ML systems analyst writing a short scientific note about ONE "
    "neural-architecture submission on the PRISM 'ability-to-learn' subnet. You are given a set of "
    "VERIFIED FACTS measured by the validator (prequential bits-per-byte, a compute profile, and a "
    "loss-curve trend). Write a brief markdown report (a '## Summary' plus a few bullet points on "
    "learning efficiency and compute cost). Ground every statement ONLY in the provided facts: do "
    "NOT invent numbers, layer types, datasets, or comparisons that are not in the facts. If a "
    "fact is missing (null), say it is not available rather than guessing."
)


def llm_report_config(settings: PrismSettings) -> LlmReviewConfig:
    """Build the OpenRouter config for report generation from the same settings the gate uses."""
    return LlmReviewConfig(
        enabled=settings.llm_review_enabled,
        required=settings.llm_review_required,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key_value(),
        api_key_file=settings.openrouter_api_key_file,
        gateway_url=settings.llm_gateway_url,
        gateway_token=settings.llm_gateway_token_value(),
        timeout_seconds=settings.llm_review_timeout_seconds,
        temperature=settings.llm_review_temperature,
        max_tokens=settings.llm_review_max_tokens,
        max_retries=settings.llm_review_max_retries,
    )


def report_generation_available(config: LlmReviewConfig) -> bool:
    """True when a report can be generated (a model is set and an endpoint/credential resolves).

    Independent of the safety-gate ``enabled`` toggle: report availability is purely a function of
    whether the OpenRouter gateway/provider credential is resolvable, so a local/dev deployment
    without a key degrades cleanly to ``unavailable`` instead of raising.
    """
    if not config.model:
        return False
    try:
        _resolve_endpoint(config)
    except RuntimeError:
        return False
    return True


def _fmt(value: Any) -> str:
    return "not available" if value is None else str(value)


def build_report_prompt(facts: dict[str, Any]) -> str:
    compute = facts.get("compute") or {}
    lines = [
        "Verified facts for this architecture submission:",
        f"- Architecture name: {_fmt(facts.get('name'))}",
        f"- Owner hotkey: {_fmt(facts.get('owner_hotkey'))}",
        f"- Best final score (higher is better): {_fmt(facts.get('best_final_score'))}",
        f"- Distinct training-script variants: {_fmt(facts.get('variant_count'))}",
        f"- Prequential bits-per-byte (lower is better): {_fmt(facts.get('prequential_bpb'))}",
        f"- Model parameters: {_fmt(compute.get('model_params'))}",
        f"- Tokens consumed: {_fmt(facts.get('tokens_consumed'))}",
        f"- Estimated total training FLOPs (6ND): {_fmt(compute.get('estimated_flops'))}",
        f"- GPUs used: {_fmt(compute.get('gpu_count'))}",
        f"- Wall-clock seconds: {_fmt(compute.get('wall_clock_seconds'))}",
        f"- Peak VRAM bytes: {_fmt(compute.get('peak_vram_bytes'))}",
        f"- Peak RSS bytes: {_fmt(compute.get('peak_rss_bytes'))}",
        f"- First online-loss sample: {_fmt(facts.get('first_loss'))}",
        f"- Last online-loss sample: {_fmt(facts.get('last_loss'))}",
        f"- Online-loss samples recorded: {_fmt(facts.get('loss_samples'))}",
        "",
        "Write the markdown report now, grounded only in the facts above.",
    ]
    return "\n".join(lines)


def generate_report_content(facts: dict[str, Any], *, config: LlmReviewConfig) -> tuple[str, str]:
    """Synchronously call OpenRouter to produce the markdown report; returns (content, model).

    Raises on any failure (no model, unresolvable endpoint, empty completion); callers run this in
    a worker thread and treat any exception as a generation error.
    """
    base_url, credential = _resolve_endpoint(config)
    if not config.model:
        raise RuntimeError("OpenRouter model is not configured")
    chat_openai = _load_chat_openai()
    chat = chat_openai(
        model=config.model,
        base_url=base_url,
        api_key=credential,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        max_tokens=config.max_tokens,
    )
    message = chat.invoke(
        [("system", REPORT_SYSTEM), ("user", build_report_prompt(facts))]
    )
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    text = str(content).strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty report")
    return text, config.model
