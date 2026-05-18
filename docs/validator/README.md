# Validator Guide

## Purpose

PRISM lets validators operate a neural architecture search challenge for Platform. Validators accept
miner submissions, run safety review and isolated evaluations, attribute architecture and training
ownership, and expose final raw weights to Platform.

## Responsibilities

Validators are responsible for:

- accepting only signed miner submissions;
- enforcing replay protection and size limits;
- keeping evaluation isolated from the validator host;
- configuring benchmark budgets and component reward thresholds;
- reviewing low-confidence architecture or training transfers;
- protecting shared Platform, broker, and review-provider tokens;
- monitoring scoring, holds, failures, and exported weights.

## Evaluation Lifecycle

1. A miner submits a signed bundle.
2. PRISM validates hotkey, timestamp, nonce, and submission size.
3. The bundle is parsed and reviewed for safety.
4. Architecture and training fingerprints are computed.
5. Proxy evaluation measures learning, recipe, stability, and scaling signals.
6. Component attribution assigns architecture and training ownership.
7. Final scores are persisted with metrics and audit information.
8. Platform reads the exported hotkey weights.

## Runtime Configuration

PRISM settings use the `PRISM_` prefix and accept compatible `CHALLENGE_` values when orchestrated
by Platform.

| Setting | Purpose |
| --- | --- |
| `PRISM_DATABASE_URL` | Persistent storage location. |
| `PRISM_SHARED_TOKEN` | Shared token for Platform internal calls. |
| `PRISM_SHARED_TOKEN_FILE` | File containing the shared token. |
| `PRISM_PUBLIC_SUBMISSIONS_ENABLED` | Enables public miner submissions. |
| `PRISM_SIGNATURE_TTL_SECONDS` | Replay-protection timestamp window. |
| `PRISM_EPOCH_SECONDS` | PRISM scoring epoch length. |
| `PRISM_MAX_CODE_BYTES` | Maximum submission size. |
| `PRISM_MAX_PARAMETERS` | Maximum accepted model parameters. |
| `PRISM_MAX_LAYERS` | Maximum accepted layer count. |
| `PRISM_SEQUENCE_LENGTH` | Default proxy sequence length. |
| `PRISM_FINEWEB_SAMPLE_COUNT` | FineWeb-style sample count for proxy evaluation. |
| `PRISM_EXECUTION_BACKEND` | Evaluation backend mode. |
| `PRISM_DOCKER_ENABLED` | Enables isolated container execution. |
| `PRISM_DOCKER_BACKEND` | Local or broker-backed execution. |
| `PRISM_DOCKER_BROKER_URL` | Broker URL for remote execution. |
| `PRISM_PLATFORM_EVAL_IMAGE` | Evaluation image. |
| `PRISM_PLATFORM_EVAL_GPU_COUNT` | Requested GPU count for evaluation jobs. |

Use secret files for shared tokens, broker tokens, and review-provider keys.

## Component Reward Configuration

Architecture and training rewards can be split:

```bash
PRISM_COMPONENT_REWARDS_ENABLED=true
PRISM_ARCHITECTURE_REWARD_WEIGHT=0.65
PRISM_TRAINING_REWARD_WEIGHT=0.35
```

Improvement thresholds prevent tiny metric noise from stealing ownership:

```bash
PRISM_ARCHITECTURE_IMPROVEMENT_MIN_DELTA_ABS=0.01
PRISM_TRAINING_IMPROVEMENT_MIN_DELTA_ABS=0.02
PRISM_TRAINING_IMPROVEMENT_Z_SCORE=1.0
```

Transfer gates should be stricter than ordinary improvement gates:

```bash
PRISM_ARCHITECTURE_TRANSFER_MIN_DELTA_ABS=0.08
PRISM_ARCHITECTURE_TRANSFER_MIN_DELTA_REL=0.05
PRISM_TRAINING_TRANSFER_MIN_DELTA_ABS=0.05
PRISM_TRAINING_TRANSFER_MIN_DELTA_REL=0.03
```

## Public Miner Surface

Miners and dashboards use:

```http
POST /v1/submissions
GET /v1/submissions/{submission_id}
GET /v1/leaderboard
GET /v1/architectures
GET /v1/training-variants
GET /v1/epochs/current
```

## Platform Contract

Health check:

```http
GET /health
```

Version and capability check:

```http
GET /version
```

Weight request:

```http
GET /internal/v1/get_weights
Authorization: Bearer <shared-token>
X-Platform-Challenge-Slug: prism
```

Example response:

```json
{
  "challenge_slug": "prism",
  "epoch": 1760000000,
  "weights": {
    "5Abc...": 0.91
  }
}
```

## Review And Holds

PRISM can use static checks, plagiarism checks, and optional semantic review. Low-confidence
component matches can be held instead of immediately transferring ownership.

Operators should review holds when:

- the new submission is very similar to an existing family;
- the metric improvement is near the threshold;
- the semantic reviewer cannot confidently decide same, novel, or transfer;
- the submission appears to improve training while changing architecture structure.

## Operator Checklist

Before accepting submissions:

1. Configure persistent storage.
2. Configure shared Platform token delivery.
3. Configure broker and evaluation image settings.
4. Set submission size, parameter, layer, and sequence limits.
5. Decide whether semantic review is advisory or required.
6. Submit a known-safe test bundle.
7. Confirm leaderboard, architectures, training variants, current epoch, and weights.

During operation:

- monitor failed evaluations and held component transfers;
- keep epoch settings stable during active rounds;
- inspect anomalous architecture or training transfers;
- back up persistent state;
- rotate shared, broker, and review tokens if exposed;
- keep untrusted code isolated from host resources.

## Security Checklist

- Require hotkey signatures unless running a controlled local test.
- Keep replay windows short.
- Reject oversized submissions.
- Keep network access blocked for evaluation unless explicitly required.
- Keep broker and Platform tokens out of logs.
- Treat optional LLM review output as sensitive audit data.
- Use low-confidence holds for ambiguous ownership changes.
