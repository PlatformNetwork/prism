from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    sdk_version: str = "base-challenge-1"
    database_path: Path = Path("/tmp/prism.sqlite3")
    shared_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN", "CHALLENGE_SHARED_TOKEN"),
    )
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN_FILE", "CHALLENGE_SHARED_TOKEN_FILE"),
    )
    allow_insecure_signatures: bool = False
    signature_ttl_seconds: int = 300
    epoch_seconds: int = 21_600
    max_code_bytes: int = 7_500_000
    max_parameters: int = 150_000_000
    max_layers: int = 96
    max_sequence_length: int = 512
    sequence_length: int = 128
    # Static build_model instantiation gate (architecture.md section 4.1): the param-count phase
    # instantiates build_model under the forced seed in a bounded child process before any GPU
    # work, so hostile construction is time/memory-bounded at the static phase.
    static_instantiation_timeout_seconds: float = 30.0
    static_instantiation_memory_headroom_bytes: int = 8_589_934_592
    fineweb_sample_count: int = 128
    execution_backend: str = "base_gpu"
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
    component_agent_enabled: bool = True
    component_agent_required: bool = False
    component_agent_model: str | None = None
    component_agent_min_confidence: float = Field(default=0.72, ge=0, le=1)
    component_agent_transfer_confidence: float = Field(default=0.86, ge=0, le=1)
    component_agent_same_threshold: float = Field(default=0.82, ge=0, le=1)
    component_agent_hold_threshold: float = Field(default=0.55, ge=0, le=1)
    component_agent_candidate_top_k: int = Field(default=5, ge=1)
    component_agent_mermaid_enabled: bool = True
    component_hold_low_confidence: bool = True
    architecture_improvement_min_delta_abs: float = Field(default=0.01, ge=0)
    architecture_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    architecture_transfer_min_delta_abs: float = Field(default=0.08, ge=0)
    architecture_transfer_min_delta_rel: float = Field(default=0.05, ge=0)
    training_improvement_min_delta_abs: float = Field(default=0.02, ge=0)
    training_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    training_transfer_min_delta_abs: float = Field(default=0.05, ge=0)
    training_transfer_min_delta_rel: float = Field(default=0.03, ge=0)
    training_improvement_z_score: float = Field(default=1.0, ge=0)
    training_metric_default_std: float = Field(default=0.0, ge=0)
    component_eval_seed_count: int = Field(default=1, ge=1)
    component_eval_repeat_count: int = Field(default=1, ge=1)
    llm_review_enabled: bool = True
    # Fail-closed default (architecture.md sec 8, H4): when the LLM safety gate is disabled or
    # absent, the disabled-but-required branch in evaluator/llm_review.py rejects (approved=False)
    # rather than silently allowing every submission after only deterministic static checks.
    llm_review_required: bool = True
    # OpenRouter LLM hard-gate wiring (architecture.md section 7). Legacy PRISM_CHUTES_* env names
    # remain accepted so an already-running deployment keeps resolving until it is redeployed.
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("PRISM_OPENROUTER_BASE_URL", "PRISM_CHUTES_BASE_URL"),
    )
    openrouter_model: str = Field(
        default="anthropic/claude-opus-4.8",
        validation_alias=AliasChoices("PRISM_OPENROUTER_MODEL", "PRISM_CHUTES_MODEL"),
    )
    openrouter_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_OPENROUTER_API_KEY", "PRISM_CHUTES_API_KEY"),
    )
    openrouter_api_key_file: Path | None = Field(
        default=Path("/run/secrets/openrouter_api_key"),
        validation_alias=AliasChoices("PRISM_OPENROUTER_API_KEY_FILE", "PRISM_CHUTES_API_KEY_FILE"),
    )
    # The prism llm_review gate routes through the MASTER OpenRouter gateway: the gateway
    # injects the provider key server-side, so the challenge/validator holds NO provider key -- only
    # a scoped gateway token (architecture.md sections 5, 7; VAL-PRISM-031/034). When a gateway URL
    # + token are configured they take precedence over a direct OpenRouter call.
    llm_gateway_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_LLM_GATEWAY_URL", "BASE_LLM_GATEWAY_URL"),
    )
    llm_gateway_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_GATEWAY_TOKEN", "BASE_GATEWAY_TOKEN"),
    )
    llm_gateway_token_file: Path | None = Field(
        default=Path("/run/secrets/base_gateway_token"),
        validation_alias=AliasChoices("PRISM_GATEWAY_TOKEN_FILE", "BASE_GATEWAY_TOKEN_FILE"),
    )
    hf_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_HF_TOKEN", "HF_TOKEN"),
    )
    hf_token_file: Path | None = Field(
        default=Path("/run/secrets/hf_token"),
        validation_alias=AliasChoices("PRISM_HF_TOKEN_FILE", "HF_TOKEN_FILE"),
    )
    # Crash-recovery checkpoint cadence (architecture.md section 7). The validator persists a
    # training checkpoint on this cadence and pushes it to the master, which publishes it to
    # HuggingFace; a reassigned run resumes from the last public checkpoint. Hourly by default; a
    # smaller cadence (e.g. in tests) persists/publishes more frequently. Never part of the score.
    checkpoint_cadence_seconds: int = Field(
        default=3600,
        ge=1,
        validation_alias=AliasChoices(
            "PRISM_CHECKPOINT_CADENCE_SECONDS", "PRISM_HF_CHECKPOINT_CADENCE_SECONDS"
        ),
    )
    # HuggingFace model repo the master publishes crash-recovery checkpoints to (architecture.md
    # section 7). The publisher is an interface (mocked in tests); this only names the deploy repo.
    checkpoint_repo_id: str = Field(
        default="baseintelligence/prism-checkpoints",
        validation_alias=AliasChoices("PRISM_CHECKPOINT_REPO_ID", "PRISM_HF_CHECKPOINT_REPO_ID"),
    )
    llm_review_timeout_seconds: int = 60
    held_review_timeout_seconds: int = 86400
    llm_review_temperature: float = 0.0
    # Must be large enough to hold the forced SubmitMermaid tool call on real (~14KB) prompts;
    # 512 truncates it (finish_reason=length, empty tool_calls) -> the gate fails closed for every
    # real submission. 4096 leaves ample headroom (~630 output tokens observed in production).
    llm_review_max_tokens: int = 4096
    llm_review_max_retries: int = 1
    llm_review_max_source_chars: int = 200_000
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
        default="broker",
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
        default=("baseintelligence/", "ghcr.io/baseintelligence/"),
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
    base_eval_image: str = "ghcr.io/baseintelligence/prism-evaluator:latest"
    # Wall-clock budget hardening (architecture.md sections 4.3, 9). The score is
    # compute-normalized (tokens/FLOPs), so wall-clock is ONLY a safety cap, not part of the
    # score. Three layers, smallest first:
    #   1. ``base_eval_budget_seconds`` (graceful, 10-30 min): the challenge runner stops the
    #      single-pass loop at this point and scores on the PARTIAL captured stream.
    #   2. ``base_eval_budget_seconds + base_eval_watchdog_grace_seconds`` (hard): a
    #      runner watchdog thread terminates a loop that hangs OUTSIDE the instrumented iterator
    #      (so a non-iterating hang is still bounded), landing the run failed with a budget reason.
    #   3. ``base_eval_timeout_seconds`` (outer docker/broker cap): the absolute backstop, set
    #      strictly above budget+grace so the runner gets a chance to stop gracefully first.
    base_eval_budget_seconds: int = 1200
    base_eval_watchdog_grace_seconds: int = 120
    # Bound on the only writable path (``ctx.artifacts_dir``): a runner watchdog fails the run if
    # the artifacts dir grows past this quota so an artifacts disk-fill cannot take down the host
    # (architecture.md section 9; VAL-HARNESS-026).
    base_eval_artifacts_quota_bytes: int = 2_147_483_648
    base_eval_timeout_seconds: int = 1800
    # Orchestration-level HARD wall-time cap (architecture.md sections 4.3, 9). The inner docker /
    # broker timeout (``base_eval_hard_timeout_seconds``) should normally fire first, but a hung
    # broker / un-cancellable worker thread could otherwise hold the single GPU forever; this is
    # the absolute backstop the worker enforces around ``evaluator.evaluate`` so an over-time eval
    # is KILLED (its container reaped) and its GPU lease RELEASED. ``0`` auto-derives it as
    # ``base_eval_hard_timeout_seconds + base_eval_orchestration_grace_seconds`` so it always sits
    # strictly above the inner cap; a positive value overrides it (used to force a tiny cap).
    base_eval_orchestration_timeout_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_ORCHESTRATION_TIMEOUT_SECONDS",
            "CHALLENGE_BASE_EVAL_ORCHESTRATION_TIMEOUT_SECONDS",
        ),
    )
    base_eval_orchestration_grace_seconds: int = Field(default=300, ge=0)
    base_eval_cpus: float = 2.0
    base_eval_memory: str = "8g"
    base_eval_memory_swap: str | None = "8g"
    base_eval_pids_limit: int = 512
    base_eval_read_only: bool = True
    base_eval_max_gpu_count: int = Field(default=8, ge=1, le=8)
    base_eval_gpu_count: int = 1
    # Per-eval GPU VRAM cap in MiB (architecture.md section 9). Docker has no native per-container
    # VRAM cgroup, so the cap is propagated to the container env (``PRISM_GPU_VRAM_CAP_MIB``) and
    # the challenge runner clamps the torch CUDA allocator via
    # ``torch.cuda.set_per_process_memory_fraction`` BEFORE any miner code runs, so an oversized
    # model (``max_code_bytes`` up to 7.5MB) cannot exhaust GPU memory and wedge the single worker.
    # ``0`` disables the cap (deploys set a concrete value with headroom in config.example.yaml).
    base_eval_gpu_vram_mib: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_GPU_VRAM_MIB", "CHALLENGE_BASE_EVAL_GPU_VRAM_MIB"
        ),
    )
    # Multi-GPU static contract policy (architecture.md section 8). Gate A statically verifies the
    # miner training.py uses the distributed primitives + a rank-0 write guard and rejects a
    # gpu_count > 8 / multi-node request before any GPU work. ``reject`` (default) hard-rejects a
    # non-distributed script; ``flag`` advances but logs; ``off`` skips the check.
    distributed_contract_policy: Literal["reject", "flag", "off"] = Field(
        default="reject",
        validation_alias=AliasChoices(
            "PRISM_DISTRIBUTED_CONTRACT_POLICY", "CHALLENGE_DISTRIBUTED_CONTRACT_POLICY"
        ),
    )
    base_eval_gpu_type: str | None = None
    base_gpu_targets: str | None = None
    base_eval_gpu_server: str | None = None
    base_eval_gpu_device_ids: tuple[str, ...] = ()
    base_eval_task: str = "architecture"
    base_eval_artifact_root: Path = Path("/tmp/prism-eval-artifacts")
    # Read-only locked FineWeb-Edu train split mount (architecture.md section 3). The broker
    # bind-mounts the staged train shards here (RO); the challenge runner resolves ctx.data_dir to
    # this path and fails fast when it is missing/empty (no random-token fallback).
    base_eval_data_dir: str = Field(
        default="/data/fineweb-edu/train",
        validation_alias=AliasChoices("PRISM_BASE_EVAL_DATA_DIR", "PRISM_EVAL_DATA_DIR"),
    )
    # Secret held-out val split (architecture.md sections 5, 6). It is NEVER bind-mounted into the
    # eval container (VAL-HARNESS-015 / VAL-CHEAT-007) and never exposed via PrismContext; only the
    # CHALLENGE SCORER reads it (host-side) to compute the held-out delta-over-random-init
    # tie-breaker and the train-vs-held-out anti-memorization gap. An unset/empty path simply
    # skips the held-out delta (the run still scores on prequential bpb).
    base_eval_val_data_dir: str = Field(
        default="/data/fineweb-edu/val",
        validation_alias=AliasChoices("PRISM_BASE_EVAL_VAL_DATA_DIR", "PRISM_EVAL_VAL_DATA_DIR"),
    )
    # Host-readable train split for the anti-memorization gap (architecture.md section 6.2). The
    # held-out scorer re-evaluates the trained model byte-level over a fixed prefix of the EXPOSED
    # train split to obtain the CONVERGED (final-checkpoint) train bpb, used as the train side of
    # the train-vs-held-out gap. Measuring the gap against the converged model (not the
    # curve-averaged prequential AUC, which is inflated by early high-loss steps and shrinks the
    # gap) reliably flags a genuine memorizer while leaving a benign learner unflagged. The train
    # split is NOT secret, so the deploy may mount it into the scorer container; when this path is
    # unset/unavailable the gap gracefully falls back to the (basis-gated) prequential reference.
    base_eval_train_data_dir: str = Field(
        default="/data/fineweb-edu/train",
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_TRAIN_DATA_DIR", "PRISM_EVAL_TRAIN_DATA_DIR"
        ),
    )
    # Host-side held-out compute budget (architecture.md sections 4, 5; m4-heldout-live-budget-
    # tuning). The held-out delta + anti-memorization gap are computed on the worker host (CPU)
    # AFTER the container eval, evaluating a random-init twin + the trained model over the SECRET
    # val split. The full single-threaded eval overruns a tight timeout, so the scorer caps the
    # held-out eval to a FIXED, DETERMINISTIC val byte budget (a stable prefix, identical for both
    # models so the delta stays comparable) and uses a raised, configurable timeout. The byte
    # denominator keeps the delta tokenizer-agnostic; the fixed prefix keeps it deterministic. A
    # byte budget <= 0 scores the entire val split.
    base_eval_heldout_val_byte_budget: int = Field(
        default=65536,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_HELDOUT_VAL_BYTE_BUDGET", "PRISM_EVAL_HELDOUT_VAL_BYTE_BUDGET"
        ),
    )
    base_eval_heldout_timeout_seconds: float = Field(
        default=600.0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_HELDOUT_TIMEOUT_SECONDS", "PRISM_EVAL_HELDOUT_TIMEOUT_SECONDS"
        ),
    )
    base_eval_reference_tokenizer_dir: str = Field(
        default="/opt/reference-tokenizers",
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_REFERENCE_TOKENIZER_DIR", "PRISM_REFERENCE_TOKENIZER_DIR"
        ),
    )
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
    def base_eval_hard_timeout_seconds(self) -> int:
        """Outer docker/broker timeout, forced strictly above the graceful budget + watchdog grace.

        The runner's graceful budget and hard watchdog must both fire BEFORE this absolute backstop
        so an over-budget loop is stopped gracefully (or failed with a budget reason) rather than
        bluntly killed by the broker; a slack margin gives the runner time to author its manifest.
        """
        floor = self.base_eval_budget_seconds + self.base_eval_watchdog_grace_seconds + 60
        return max(self.base_eval_timeout_seconds, floor)

    @property
    def resolved_orchestration_timeout_seconds(self) -> float:
        if self.base_eval_orchestration_timeout_seconds > 0:
            return self.base_eval_orchestration_timeout_seconds
        return float(
            self.base_eval_hard_timeout_seconds + self.base_eval_orchestration_grace_seconds
        )

    @property
    def resolved_database_path(self) -> Path:
        if self.database_url.startswith("sqlite+aiosqlite:///"):
            return Path(self.database_url.removeprefix("sqlite+aiosqlite:///"))
        return self.database_path

    def openrouter_api_key_value(self) -> str | None:
        if self.openrouter_api_key:
            return self.openrouter_api_key
        if self.openrouter_api_key_file and self.openrouter_api_key_file.exists():
            token = self.openrouter_api_key_file.read_text(encoding="utf-8").strip()
            return token or None
        return None

    def llm_gateway_token_value(self) -> str | None:
        if self.llm_gateway_token:
            return self.llm_gateway_token
        if self.llm_gateway_token_file and self.llm_gateway_token_file.exists():
            token = self.llm_gateway_token_file.read_text(encoding="utf-8").strip()
            return token or None
        return None

    def hf_token_value(self) -> str | None:
        if self.hf_token:
            return self.hf_token
        if self.hf_token_file and self.hf_token_file.exists():
            token = self.hf_token_file.read_text(encoding="utf-8").strip()
            return token or None
        return None


settings = PrismSettings()
