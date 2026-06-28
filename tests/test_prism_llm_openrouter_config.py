from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import llm_review as llm
from prism_challenge.evaluator.llm_review import LlmReviewConfig, review_code
from prism_challenge.queue import PrismWorker
from prism_challenge.runtime_config import resolve_runtime_policy, runtime_policy_defaults

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STRONG_MODEL = "anthropic/claude-opus-4.8"
DEAD_MODEL = "anthropic/claude-3.5-sonnet"
SECRET_PATH = Path("/run/secrets/openrouter_api_key")


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PRISM_LLM_REVIEW_ENABLED",
        "PRISM_OPENROUTER_BASE_URL",
        "PRISM_OPENROUTER_MODEL",
        "PRISM_OPENROUTER_API_KEY",
        "PRISM_OPENROUTER_API_KEY_FILE",
        "PRISM_CHUTES_BASE_URL",
        "PRISM_CHUTES_MODEL",
        "PRISM_CHUTES_API_KEY",
        "PRISM_CHUTES_API_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_prism_settings_default_to_openrouter_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    settings = PrismSettings()

    assert settings.llm_review_enabled is True
    assert settings.openrouter_base_url == OPENROUTER_BASE_URL
    assert settings.openrouter_model == STRONG_MODEL
    assert settings.openrouter_model != DEAD_MODEL
    assert settings.openrouter_api_key_file == SECRET_PATH
    assert settings.llm_review_temperature == 0.0


def test_openrouter_key_only_sourced_from_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_llm_env(monkeypatch)
    secret = tmp_path / "openrouter_api_key"
    secret.write_text("sk-or-secret\n", encoding="utf-8")

    settings = PrismSettings(openrouter_api_key_file=secret)
    assert settings.openrouter_api_key_value() == "sk-or-secret"

    # With no inline key and a missing file, no key is resolved (fails closed).
    missing = PrismSettings(openrouter_api_key=None, openrouter_api_key_file=tmp_path / "nope")
    assert missing.openrouter_api_key_value() is None


def test_hf_token_only_sourced_from_secret_file(tmp_path: Path) -> None:
    secret = tmp_path / "hf_token"
    secret.write_text("hf_secret\n", encoding="utf-8")
    settings = PrismSettings(hf_token_file=secret)
    assert settings.hf_token_value() == "hf_secret"

    # FineWeb-Edu is public (anonymous works), so a missing token file resolves
    # to None instead of failing: the prep download simply runs unauthenticated.
    missing = PrismSettings(hf_token=None, hf_token_file=tmp_path / "nope")
    assert missing.hf_token_value() is None


def test_legacy_chutes_env_alias_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PRISM_CHUTES_MODEL", "openai/gpt-4o-mini")
    settings = PrismSettings()
    assert settings.openrouter_model == "openai/gpt-4o-mini"


def test_llm_review_config_defaults_to_openrouter_temperature_zero() -> None:
    config = LlmReviewConfig()

    assert config.enabled is True
    assert config.base_url == OPENROUTER_BASE_URL
    assert config.model == STRONG_MODEL
    assert config.model != DEAD_MODEL
    assert config.temperature == 0.0
    assert str(config.api_key_file) == str(SECRET_PATH)


def test_worker_llm_config_maps_openrouter_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_llm_env(monkeypatch)
    secret = tmp_path / "openrouter_api_key"
    secret.write_text("sk-or-worker\n", encoding="utf-8")
    settings = PrismSettings(openrouter_api_key_file=secret)

    config = PrismWorker._llm_config(SimpleNamespace(settings=settings))

    assert config.enabled is True
    assert config.base_url == OPENROUTER_BASE_URL
    assert config.model == STRONG_MODEL
    assert config.temperature == 0.0
    assert config.api_key == "sk-or-worker"


def test_runtime_policy_reports_openrouter_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    settings = PrismSettings()
    defaults = runtime_policy_defaults(settings)

    policy = defaults["llm_review_policy"]
    assert policy["enabled"] is True
    assert policy["base_url"] == OPENROUTER_BASE_URL
    assert policy["model"] == STRONG_MODEL

    # The typed model must accept and surface the new fields (extra='forbid').
    model = resolve_runtime_policy(settings, [])
    assert model.llm_review_policy.enabled is True
    assert model.llm_review_policy.base_url == OPENROUTER_BASE_URL
    assert model.llm_review_policy.model == STRONG_MODEL


def test_invoke_review_flow_makes_real_openrouter_call_at_temperature_zero() -> None:
    captured: dict[str, Any] = {}

    class _FakeMessage:
        def __init__(self, tool_name: str) -> None:
            self.tool_calls = [
                {
                    "name": tool_name,
                    "args": _ARGS_BY_TOOL[tool_name],
                }
            ]

    class _FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self._tool: str | None = None

        def bind_tools(self, tools: list[Any], tool_choice: str, strict: bool) -> _FakeChat:
            self._tool = tool_choice
            return self

        def invoke(self, messages: list[tuple[str, str]]) -> _FakeMessage:
            assert self._tool is not None
            return _FakeMessage(self._tool)

    monkeypatch_target = llm
    original = monkeypatch_target._load_chat_openai
    monkeypatch_target._load_chat_openai = lambda: _FakeChat  # type: ignore[assignment]
    try:
        review = review_code(
            "def build_model(ctx):\n    return None\n",
            config=LlmReviewConfig(api_key="sk-or-test"),
        )
    finally:
        monkeypatch_target._load_chat_openai = original  # type: ignore[assignment]

    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert captured["model"] == STRONG_MODEL
    assert captured["model"] != DEAD_MODEL
    assert captured["temperature"] == 0.0
    assert captured["api_key"] == "sk-or-test"
    assert review.approved is True


_ARGS_BY_TOOL: dict[str, dict[str, Any]] = {
    "SubmitMermaid": {"mermaid": "flowchart LR\n  A[Source] --> B[Review]", "notes": "ok"},
    "SubmitVerdict": {
        "reason": "defines build_model; no escapes detected",
        "verdict": True,
        "violations": [],
        "confidence": 0.9,
        "rule_ids": [],
        "evidence": [],
    },
}
