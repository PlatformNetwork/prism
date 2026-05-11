# Operator Guide

This guide covers local validation and production-oriented configuration for running PRISM as a Platform challenge.

## Installation

```bash
git clone https://github.com/PlatformNetwork/prism.git
cd prism
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Local Validation

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy --config-file pyproject.toml src
.venv/bin/pytest tests
```

## Required Runtime Configuration

At minimum, PRISM needs:

```bash
PRISM_DATABASE_URL=sqlite+aiosqlite:////data/prism.sqlite3
PRISM_SHARED_TOKEN_FILE=/run/secrets/platform/challenge_token
PRISM_EXECUTION_BACKEND=platform_gpu
```

The shared token must match the token configured in the Platform master for this challenge.

## Docker Broker Configuration

Production evaluation should use the Platform Docker broker:

```bash
PRISM_DOCKER_ENABLED=true
PRISM_DOCKER_BACKEND=broker
PRISM_DOCKER_BROKER_URL=http://platform-docker-broker:8082
PRISM_DOCKER_BROKER_TOKEN_FILE=/run/secrets/platform/challenge_token
PRISM_PLATFORM_EVAL_IMAGE=ghcr.io/platformnetwork/prism-evaluator:latest
PRISM_PLATFORM_EVAL_GPU_COUNT=1
```

Useful limits:

```bash
PRISM_PLATFORM_EVAL_TIMEOUT_SECONDS=900
PRISM_PLATFORM_EVAL_CPUS=2
PRISM_PLATFORM_EVAL_MEMORY=8g
PRISM_PLATFORM_EVAL_PIDS_LIMIT=512
PRISM_DOCKER_NETWORK=none
```

## Review Configuration

Recommended production review settings:

```bash
PRISM_LLM_REVIEW_ENABLED=true
PRISM_LLM_REVIEW_REQUIRED=true
PRISM_PLAGIARISM_ENABLED=true
```

If using Chutes-compatible review:

```bash
PRISM_CHUTES_BASE_URL=https://llm.chutes.ai/v1
PRISM_CHUTES_MODEL=<model-name>
PRISM_CHUTES_API_KEY_FILE=/run/secrets/chutes_api_key
```

## Component Reward Configuration

```bash
PRISM_COMPONENT_REWARDS_ENABLED=true
PRISM_ARCHITECTURE_REWARD_WEIGHT=0.65
PRISM_TRAINING_REWARD_WEIGHT=0.35
PRISM_ARCHITECTURE_IMPROVEMENT_MIN_DELTA_ABS=0.01
PRISM_TRAINING_IMPROVEMENT_MIN_DELTA_ABS=0.02
PRISM_TRAINING_IMPROVEMENT_Z_SCORE=1.0
```

Architecture and training weights should usually sum to `1.0`.

## Semantic Attribution Configuration

The component agent protects ownership from useless diffs and low-confidence matches:

```bash
PRISM_COMPONENT_AGENT_ENABLED=true
PRISM_COMPONENT_AGENT_REQUIRED=false
PRISM_COMPONENT_AGENT_MIN_CONFIDENCE=0.72
PRISM_COMPONENT_AGENT_TRANSFER_CONFIDENCE=0.86
PRISM_COMPONENT_AGENT_SAME_THRESHOLD=0.82
PRISM_COMPONENT_AGENT_HOLD_THRESHOLD=0.55
PRISM_COMPONENT_AGENT_CANDIDATE_TOP_K=5
PRISM_COMPONENT_AGENT_MERMAID_ENABLED=true
PRISM_COMPONENT_HOLD_LOW_CONFIDENCE=true
```

Recommended transfer gates:

```bash
PRISM_ARCHITECTURE_TRANSFER_MIN_DELTA_ABS=0.08
PRISM_ARCHITECTURE_TRANSFER_MIN_DELTA_REL=0.05
PRISM_TRAINING_TRANSFER_MIN_DELTA_ABS=0.05
PRISM_TRAINING_TRANSFER_MIN_DELTA_REL=0.03
```

Low-confidence holds can be inspected and resolved with:

```bash
curl -H "Authorization: Bearer dev-secret" \
  http://localhost:8000/internal/v1/component-review/holds
```

## Scaling-Signal Configuration

Evaluator images should report stability and scaling metrics when available:

- `loss_smoothness`
- `loss_spike_count`
- `grad_norm_mean`
- `grad_norm_max`
- `activation_spike_rate`
- `scaling_consistency`
- `depth_scaling_score`
- `sequence_scaling_score`
- `batch_scaling_score`
- `nan_count`
- `overflow_count`

These metrics should complement `q_arch` and `q_recipe`; do not rely on final loss or a single benchmark alone.

## Running Locally

```bash
PRISM_SHARED_TOKEN=dev-secret \
PRISM_DATABASE_URL=sqlite+aiosqlite:///./prism.sqlite3 \
.venv/bin/uvicorn prism_challenge.app:app --host 0.0.0.0 --port 8000
```

## Platform Deployment

In a Platform deployment, PRISM should be registered as a challenge image and reached by the master over the internal challenge network. Public miner traffic should go through the Platform proxy, which verifies signatures and forwards to PRISM.

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
```

Internal weights require the shared token:

```bash
curl -H "Authorization: Bearer dev-secret" \
  -H "X-Platform-Challenge-Slug: prism" \
  http://localhost:8000/internal/v1/get_weights
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `invalid internal token` | Shared token mismatch between Platform and PRISM |
| submission rejected before container | Static safety or contract validation failed |
| evaluation failed | Broker, image, GPU, timeout, or container error |
| empty weights | No completed component-scored submissions yet |
| training variant not rewarded | Improvement did not pass dynamic threshold |
| submission held | Component agent confidence was too low; resolve the review hold |
