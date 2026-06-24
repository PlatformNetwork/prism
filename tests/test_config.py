from __future__ import annotations

from pathlib import Path

import yaml

from prism_challenge.config import PrismSettings


def test_base_challenge_env_aliases_are_loaded(monkeypatch):
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", "sqlite+aiosqlite:////data/challenge.sqlite3")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "/run/secrets/base/challenge_token")
    monkeypatch.setenv("CHALLENGE_DOCKER_ENABLED", "true")
    monkeypatch.setenv("CHALLENGE_DOCKER_BACKEND", "broker")
    monkeypatch.setenv("CHALLENGE_DOCKER_BROKER_URL", "http://base-docker-broker:8082")
    monkeypatch.setenv(
        "CHALLENGE_DOCKER_BROKER_TOKEN_FILE", "/run/secrets/base/challenge_token"
    )

    settings = PrismSettings()

    assert settings.database_url == "sqlite+aiosqlite:////data/challenge.sqlite3"
    assert settings.shared_token_file == "/run/secrets/base/challenge_token"
    assert settings.docker_enabled is True
    assert settings.docker_backend == "broker"
    assert settings.docker_broker_url == "http://base-docker-broker:8082"
    assert str(settings.docker_broker_token_file) == "/run/secrets/base/challenge_token"


def test_docker_backend_default_is_broker_safe_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("PRISM_DOCKER_BACKEND", raising=False)
    monkeypatch.delenv("CHALLENGE_DOCKER_BACKEND", raising=False)

    settings = PrismSettings()

    assert settings.docker_backend == "broker"
    assert settings.docker_backend != "cli"


def test_docker_backend_explicit_env_overrides_default(monkeypatch) -> None:
    for env_name in ("CHALLENGE_DOCKER_BACKEND", "PRISM_DOCKER_BACKEND"):
        monkeypatch.delenv("PRISM_DOCKER_BACKEND", raising=False)
        monkeypatch.delenv("CHALLENGE_DOCKER_BACKEND", raising=False)
        for explicit in ("cli", "direct", "broker"):
            monkeypatch.setenv(env_name, explicit)
            assert PrismSettings().docker_backend == explicit
            monkeypatch.delenv(env_name, raising=False)

    assert PrismSettings(docker_backend="cli").docker_backend == "cli"


def test_settings_still_accept_field_names() -> None:
    settings = PrismSettings(
        database_url="sqlite+aiosqlite:////tmp/prism.sqlite3",
        shared_token="secret",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://broker",
    )

    assert settings.database_url == "sqlite+aiosqlite:////tmp/prism.sqlite3"
    assert settings.shared_token == "secret"
    assert settings.docker_enabled is True
    assert settings.docker_backend == "broker"
    assert settings.docker_broker_url == "http://broker"


def test_secret_file_helpers(tmp_path) -> None:
    shared = tmp_path / "shared-token"
    shared.write_text("shared\n", encoding="utf-8")
    openrouter = tmp_path / "openrouter-token"
    openrouter.write_text("openrouter\n", encoding="utf-8")

    settings = PrismSettings(
        database_url="postgresql://db/prism",
        database_path=tmp_path / "fallback.sqlite3",
        shared_token_file=str(shared),
        openrouter_api_key_file=openrouter,
    )

    assert settings.internal_token() == "shared"
    assert settings.resolved_database_path == tmp_path / "fallback.sqlite3"
    assert settings.openrouter_api_key_value() == "openrouter"


def test_internal_token_requires_secret() -> None:
    settings = PrismSettings(shared_token_file=None)

    try:
        settings.internal_token()
    except RuntimeError as exc:
        assert "PRISM_SHARED_TOKEN" in str(exc)
    else:
        raise AssertionError("internal_token should require a configured secret")



def test_max_code_bytes_holds_five_mib_zip_base64() -> None:
    # 5 MiB raw zip -> base64 length 6,990,508; cap must comfortably exceed it.
    raw_five_mib = 5 * 1024 * 1024
    base64_len = 4 * ((raw_five_mib + 2) // 3)

    settings = PrismSettings()

    assert settings.max_code_bytes == 7_500_000
    assert settings.max_code_bytes > base64_len


def test_example_config_parses_with_nas_defaults() -> None:
    payload = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    settings = PrismSettings(**payload)

    assert settings.slug == "prism"
    assert settings.execution_backend == "base_gpu"
    assert settings.public_submissions_enabled is True
    assert settings.arch_weight == 0.7
    assert settings.recipe_weight == 0.3
    assert settings.base_eval_max_gpu_count == 8
    assert settings.base_eval_gpu_count == 1
    assert settings.docker_enabled is False
    assert settings.docker_backend == "cli"
    assert settings.shared_token is None
    assert settings.openrouter_api_key is None
    assert settings.docker_broker_token is None
    assert "shared_token" not in payload
    assert "openrouter_api_key" not in payload
    assert "docker_broker_token" not in payload
