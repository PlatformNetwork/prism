from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .sdk.config import ChallengeSettings


class PrismSettings(ChallengeSettings):
    model_config = SettingsConfigDict(env_prefix="PRISM_", env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:////data/prism.sqlite3"
    slug: str = "prism"
    name: str = "Prism"
    version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "platform-challenge-1"
    database_path: Path = Path("/tmp/prism.sqlite3")
    shared_token: str | None = None
    shared_token_file: str | None = None
    allow_insecure_signatures: bool = False
    signature_ttl_seconds: int = 300
    epoch_seconds: int = 21_600
    max_code_bytes: int = 200_000
    max_parameters: int = 150_000_000
    max_layers: int = 96
    max_sequence_length: int = 512
    sequence_length: int = 128
    fineweb_sample_count: int = 128
    execution_backend: str = "local_cpu"
    lium_base_url: str | None = None
    lium_token: str | None = None
    lium_token_file: Path | None = None
    worker_claim_timeout_seconds: int = 900
    l2_top_k: int = 200
    l3_top_k: int = 20
    kendall_tau_min: float = 0.4
    arch_weight: float = Field(default=0.7, ge=0, le=1)
    recipe_weight: float = Field(default=0.3, ge=0, le=1)

    def internal_token(self) -> str:
        if self.shared_token:
            return self.shared_token
        if self.shared_token_file and Path(self.shared_token_file).exists():
            return Path(self.shared_token_file).read_text(encoding="utf-8").strip()
        return "dev-prism-token"

    @property
    def resolved_database_path(self) -> Path:
        if self.database_url.startswith("sqlite+aiosqlite:///"):
            return Path(self.database_url.removeprefix("sqlite+aiosqlite:///"))
        return self.database_path

    def lium_auth_token(self) -> str | None:
        if self.lium_token:
            return self.lium_token
        if self.lium_token_file and self.lium_token_file.exists():
            return self.lium_token_file.read_text(encoding="utf-8").strip()
        return None


settings = PrismSettings()
