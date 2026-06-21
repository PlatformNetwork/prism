# Operator Guide

This guide covers local validation and production-oriented configuration for running PRISM as a
Platform challenge.

## Installation

```bash
git clone https://github.com/PlatformNetwork/prism.git
cd prism
uv sync --frozen --extra dev
```

## Local Validation

```bash
.venv/bin/ruff check src
.venv/bin/mypy src/prism_challenge/evaluator
.venv/bin/python -m pytest tests -q
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

Production evaluation uses the Platform Docker broker with the augmented evaluator image:

```bash
PRISM_DOCKER_ENABLED=true
PRISM_DOCKER_BACKEND=broker
PRISM_DOCKER_BROKER_URL=http://platform-docker-broker:8082
PRISM_DOCKER_BROKER_TOKEN_FILE=/run/secrets/platform/challenge_token
PRISM_PLATFORM_EVAL_IMAGE=ghcr.io/platformnetwork/prism-evaluator:augmented
PRISM_PLATFORM_EVAL_GPU_COUNT=1
PRISM_DOCKER_NETWORK=none
```

The scored run is single-node and uses `torchrun --standalone --nnodes=1 --nproc-per-node=1`. The
evaluator image must ship `sentencepiece` and an offline tiktoken gpt2 cache so reference tokenizers
load with no network.

## Locked FineWeb-Edu Data Plane

The broker bind-mounts the locked FineWeb-Edu data read-only into the eval container, which runs with
`network=none`:

```bash
PRISM_PLATFORM_EVAL_DATA_DIR=/data/fineweb-edu/train       # miner-visible, read-only
PRISM_PLATFORM_EVAL_VAL_DATA_DIR=/data/fineweb-edu/val     # secret; scorer-only, never mounted into eval
PRISM_PLATFORM_EVAL_REFERENCE_TOKENIZER_DIR=/opt/reference-tokenizers
```

`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set inside the eval container. The `val`/`test`
splits are secret and must never be exposed to a miner script.

## Compute Budget

The score is compute-normalized; wall-clock is only a safety cap, enforced in layers:

```bash
PRISM_PLATFORM_EVAL_BUDGET_SECONDS=1200            # graceful stop; score the partial stream
PRISM_PLATFORM_EVAL_WATCHDOG_GRACE_SECONDS=120     # hard watchdog above the graceful budget
PRISM_PLATFORM_EVAL_TIMEOUT_SECONDS=1800           # outer docker/broker backstop
PRISM_PLATFORM_EVAL_ARTIFACTS_QUOTA_BYTES=2147483648
```

## LLM Hard Gate Configuration

The OpenRouter LLM hard gate is enabled by default and reviews both scripts before any GPU work:

```bash
PRISM_LLM_REVIEW_ENABLED=true
PRISM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PRISM_OPENROUTER_MODEL=openai/gpt-4o
PRISM_OPENROUTER_API_KEY_FILE=/run/secrets/openrouter_api_key
```

A `reject` from the gate is terminal. The eval container carries no OpenRouter key (the gate runs
host-side before the container is launched).

## Multi-GPU Static Contract

```bash
PRISM_DISTRIBUTED_CONTRACT_POLICY=reject     # reject | flag | off
PRISM_PLATFORM_EVAL_MAX_GPU_COUNT=8
```

`reject` (the default) hard-rejects a non-distributed `training.py`; `flag` advances but logs; `off`
skips the check.

## Duplicate Review

```bash
PRISM_PLAGIARISM_ENABLED=true
```

An exact-source-hash duplicate is rejected, and a borderline-similarity quarantine is folded into a
terminal rejection at ingress. There is no operator hold-resolution surface.

## Running Locally

```bash
PRISM_SHARED_TOKEN=dev-secret \
PRISM_DATABASE_URL=sqlite+aiosqlite:///./prism.sqlite3 \
.venv/bin/uvicorn prism_challenge.app:app --host 0.0.0.0 --port 8000
```

## Platform Deployment

In a Platform deployment, PRISM registers as a challenge image reached by the master over the internal
challenge network. Public miner traffic goes through the Platform proxy, which verifies signatures and
forwards to PRISM. Weights are exposed only via `get_weights` and are always dry-run.

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
| submission rejected before container | Static sandbox, two-script contract, param cap, distributed contract, or LLM hard-gate reject |
| submission held | LLM review quarantine (transient error or ambiguous verdict) |
| evaluation failed | Broker, image, GPU, timeout, missing locked data, or container error |
| empty weights | No completed, scored submissions yet |
| `missing_locked_data` | The read-only FineWeb-Edu train mount is absent or empty on the GPU node |
