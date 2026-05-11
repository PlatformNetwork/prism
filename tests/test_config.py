from __future__ import annotations

from prism_challenge.config import PrismSettings


def test_platform_challenge_env_aliases_are_loaded(monkeypatch):
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", "sqlite+aiosqlite:////data/challenge.sqlite3")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "/run/secrets/platform/challenge_token")
    monkeypatch.setenv("CHALLENGE_DOCKER_ENABLED", "true")
    monkeypatch.setenv("CHALLENGE_DOCKER_BACKEND", "broker")
    monkeypatch.setenv("CHALLENGE_DOCKER_BROKER_URL", "http://platform-docker-broker:8082")
    monkeypatch.setenv(
        "CHALLENGE_DOCKER_BROKER_TOKEN_FILE", "/run/secrets/platform/challenge_token"
    )

    settings = PrismSettings()

    assert settings.database_url == "sqlite+aiosqlite:////data/challenge.sqlite3"
    assert settings.shared_token_file == "/run/secrets/platform/challenge_token"
    assert settings.docker_enabled is True
    assert settings.docker_backend == "broker"
    assert settings.docker_broker_url == "http://platform-docker-broker:8082"
    assert str(settings.docker_broker_token_file) == "/run/secrets/platform/challenge_token"


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
    chutes = tmp_path / "chutes-token"
    chutes.write_text("chutes\n", encoding="utf-8")

    settings = PrismSettings(
        database_url="postgresql://db/prism",
        database_path=tmp_path / "fallback.sqlite3",
        shared_token_file=str(shared),
        chutes_api_key_file=chutes,
    )

    assert settings.internal_token() == "shared"
    assert settings.resolved_database_path == tmp_path / "fallback.sqlite3"
    assert settings.chutes_api_key_value() == "chutes"


def test_internal_token_requires_secret() -> None:
    settings = PrismSettings(shared_token_file=None)

    try:
        settings.internal_token()
    except RuntimeError as exc:
        assert "PRISM_SHARED_TOKEN" in str(exc)
    else:
        raise AssertionError("internal_token should require a configured secret")
