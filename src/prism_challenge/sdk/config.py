from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChallengeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="ignore")

    slug: str = "prism"
    name: str = "Prism"
    version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "platform-challenge-1"
    database_url: str = "sqlite+aiosqlite:////data/prism.sqlite3"
    shared_token: str | None = Field(default=None, repr=False)
    shared_token_file: str | None = Field(
        default="/run/secrets/platform/challenge_token", repr=False
    )
    host: str = "0.0.0.0"
    port: int = 8000
