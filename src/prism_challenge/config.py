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
    lium_backend: str = "jobs_api"
    lium_base_url: str | None = None
    lium_token: str | None = None
    lium_token_file: Path | None = None
    lium_executor_id: str | None = None
    lium_gpu_type: str | None = None
    lium_gpu_count: int = 1
    lium_template_id: str | None = None
    lium_ssh_key_path: str | None = None
    lium_keep_pod: bool = False
    lium_pod_timeout_seconds: int = 600
    lium_eval_timeout_seconds: int = 900
    allow_fake_lium: bool = False
    worker_claim_timeout_seconds: int = 900
    l2_top_k: int = 200
    l3_top_k: int = 20
    kendall_tau_min: float = 0.4
    arch_weight: float = Field(default=0.7, ge=0, le=1)
    recipe_weight: float = Field(default=0.3, ge=0, le=1)
    llm_review_enabled: bool = False
    llm_review_required: bool = False
    chutes_base_url: str = "https://llm.chutes.ai/v1"
    chutes_model: str | None = None
    chutes_api_key: str | None = None
    chutes_api_key_file: Path | None = None
    llm_review_timeout_seconds: int = 60
    llm_review_temperature: float = 0.0
    llm_review_max_tokens: int = 512
    llm_review_max_retries: int = 1
    subnet_rules_json: str | None = None
    subnet_rules_file: Path | None = None
    plagiarism_enabled: bool = True
    plagiarism_min_similarity: float = 0.65
    plagiarism_static_reject_threshold: float = 0.96
    plagiarism_top_k: int = 2
    plagiarism_sandbox_enabled: bool = False
    plagiarism_sandbox_image: str = "python:3.12-alpine"
    plagiarism_sandbox_timeout_seconds: int = 30
    plagiarism_storage_max_files: int = 200
    plagiarism_storage_max_bytes: int = 2_000_000
    docker_bin: str = "docker"
    docker_backend: str = "cli"
    docker_broker_url: str | None = None
    docker_broker_token: str | None = None
    docker_broker_token_file: Path | None = None
    docker_network: str = "none"
    docker_cpus: float = 1.0
    docker_memory: str = "512m"
    docker_memory_swap: str | None = "512m"
    docker_pids_limit: int = 128
    docker_read_only: bool = True
    docker_user: str | None = None

    def internal_token(self) -> str:
        if self.shared_token:
            return self.shared_token
        if self.shared_token_file and Path(self.shared_token_file).exists():
            return Path(self.shared_token_file).read_text(encoding="utf-8").strip()
        raise RuntimeError("PRISM_SHARED_TOKEN or PRISM_SHARED_TOKEN_FILE is required")

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

    def chutes_api_key_value(self) -> str | None:
        if self.chutes_api_key:
            return self.chutes_api_key
        if self.chutes_api_key_file and self.chutes_api_key_file.exists():
            token = self.chutes_api_key_file.read_text(encoding="utf-8").strip()
            return token or None
        return None


settings = PrismSettings()
