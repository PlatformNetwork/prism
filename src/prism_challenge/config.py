from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from .sdk.config import ChallengeSettings


class PrismSettings(ChallengeSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_", env_file=".env", extra="ignore", populate_by_name=True
    )

    database_url: str = Field(
        default="sqlite+aiosqlite:////data/prism.sqlite3",
        validation_alias=AliasChoices("PRISM_DATABASE_URL", "CHALLENGE_DATABASE_URL"),
    )
    slug: str = "prism"
    name: str = "Prism"
    version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "platform-challenge-1"
    database_path: Path = Path("/tmp/prism.sqlite3")
    shared_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN", "CHALLENGE_SHARED_TOKEN"),
    )
    shared_token_file: str | None = Field(
        default="/run/secrets/platform/challenge_token",
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN_FILE", "CHALLENGE_SHARED_TOKEN_FILE"),
    )
    allow_insecure_signatures: bool = False
    signature_ttl_seconds: int = 300
    epoch_seconds: int = 21_600
    max_code_bytes: int = 200_000
    max_parameters: int = 150_000_000
    max_layers: int = 96
    max_sequence_length: int = 512
    sequence_length: int = 128
    fineweb_sample_count: int = 128
    execution_backend: str = "platform_gpu"
    prism_role: str = "master"
    public_submissions_enabled: bool = True
    worker_claim_timeout_seconds: int = 900
    l2_top_k: int = 200
    l3_top_k: int = 20
    kendall_tau_min: float = 0.4
    arch_weight: float = Field(default=0.7, ge=0, le=1)
    recipe_weight: float = Field(default=0.3, ge=0, le=1)
    component_rewards_enabled: bool = True
    architecture_reward_weight: float = Field(default=0.65, ge=0, le=1)
    training_reward_weight: float = Field(default=0.35, ge=0, le=1)
    architecture_improvement_min_delta_abs: float = Field(default=0.01, ge=0)
    architecture_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    training_improvement_min_delta_abs: float = Field(default=0.02, ge=0)
    training_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    training_improvement_z_score: float = Field(default=1.0, ge=0)
    training_metric_default_std: float = Field(default=0.0, ge=0)
    component_eval_seed_count: int = Field(default=1, ge=1)
    component_eval_repeat_count: int = Field(default=1, ge=1)
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
    docker_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("PRISM_DOCKER_ENABLED", "CHALLENGE_DOCKER_ENABLED"),
    )
    docker_bin: str = Field(
        default="docker",
        validation_alias=AliasChoices("PRISM_DOCKER_BIN", "CHALLENGE_DOCKER_BIN"),
    )
    docker_backend: str = Field(
        default="cli",
        validation_alias=AliasChoices("PRISM_DOCKER_BACKEND", "CHALLENGE_DOCKER_BACKEND"),
    )
    docker_broker_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_DOCKER_BROKER_URL", "CHALLENGE_DOCKER_BROKER_URL"),
    )
    docker_broker_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_DOCKER_BROKER_TOKEN", "CHALLENGE_DOCKER_BROKER_TOKEN"),
    )
    docker_broker_token_file: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "PRISM_DOCKER_BROKER_TOKEN_FILE", "CHALLENGE_DOCKER_BROKER_TOKEN_FILE"
        ),
    )
    docker_allowed_images: tuple[str, ...] = Field(
        default=("platformnetwork/", "ghcr.io/platformnetwork/"),
        validation_alias=AliasChoices(
            "PRISM_DOCKER_ALLOWED_IMAGES", "CHALLENGE_DOCKER_ALLOWED_IMAGES"
        ),
    )
    docker_network: str = Field(
        default="none",
        validation_alias=AliasChoices("PRISM_DOCKER_NETWORK", "CHALLENGE_DOCKER_NETWORK"),
    )
    docker_cpus: float = Field(
        default=1.0,
        validation_alias=AliasChoices("PRISM_DOCKER_CPUS", "CHALLENGE_DOCKER_CPUS"),
    )
    docker_memory: str = Field(
        default="512m",
        validation_alias=AliasChoices("PRISM_DOCKER_MEMORY", "CHALLENGE_DOCKER_MEMORY"),
    )
    docker_memory_swap: str | None = Field(
        default="512m",
        validation_alias=AliasChoices("PRISM_DOCKER_MEMORY_SWAP", "CHALLENGE_DOCKER_MEMORY_SWAP"),
    )
    docker_pids_limit: int = Field(
        default=128,
        validation_alias=AliasChoices("PRISM_DOCKER_PIDS_LIMIT", "CHALLENGE_DOCKER_PIDS_LIMIT"),
    )
    docker_read_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("PRISM_DOCKER_READ_ONLY", "CHALLENGE_DOCKER_READ_ONLY"),
    )
    docker_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_DOCKER_USER", "CHALLENGE_DOCKER_USER"),
    )
    platform_eval_image: str = "ghcr.io/platformnetwork/prism-evaluator:latest"
    platform_eval_timeout_seconds: int = 900
    platform_eval_cpus: float = 2.0
    platform_eval_memory: str = "8g"
    platform_eval_memory_swap: str | None = "8g"
    platform_eval_pids_limit: int = 512
    platform_eval_read_only: bool = True
    platform_eval_gpu_count: int = 1
    platform_eval_gpu_type: str | None = None
    platform_eval_task: str = "architecture"
    validator_hotkeys: tuple[str, ...] = ()
    validator_assignment_timeout_seconds: int = 900
    validator_assignment_max_attempts: int = 3

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

    def chutes_api_key_value(self) -> str | None:
        if self.chutes_api_key:
            return self.chutes_api_key
        if self.chutes_api_key_file and self.chutes_api_key_file.exists():
            token = self.chutes_api_key_file.read_text(encoding="utf-8").strip()
            return token or None
        return None


settings = PrismSettings()
