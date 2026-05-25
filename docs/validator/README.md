# Validator Guide

## Purpose

PRISM lets validators operate a neural architecture search challenge for Platform. Validators accept miner submissions, run safety review and isolated evaluations, attribute architecture and training ownership, and expose final raw weights to Platform.

## Responsibilities

Validators are responsible for:

* accepting only signed miner submissions
* enforcing replay protection and size limits
* keeping evaluation isolated from the validator host
* configuring benchmark budgets and component reward thresholds
* reviewing low-confidence architecture matches, architecture variants, and training current-best changes
* protecting shared Platform, broker, and review-provider tokens
* monitoring scoring, holds, quarantine, failures, and exported weights

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

PRISM settings use the `PRISM_` prefix and accept compatible `CHALLENGE_` values when orchestrated by Platform. These settings bootstrap the process and provide fallback defaults. Runtime challenge policy is SQL-first.

Precedence is:

```text
SQL active value → env/Pydantic default → schema default
```

SQL runtime config can override supported policy values. Official runtime config fails closed when an active SQL value is invalid. Explicit local non-scoring paths may fall back to defaults.

## SQL Runtime Config Keys

The implemented SQL policy keys are:

| Key | Policy area |
| --- | --- |
| `reward_pools` | Architecture and training reward pool split. Defaults to 60% and 40%. |
| `score_weights` | Final score blend and official architecture/training component formulas. Defaults to 70%/30% final blend. |
| `benchmark_weights` | Benchmark sanity mix across MMLU, GSM8K, MATH, HumanEval, ARC-Challenge, Needle, IFEval, and TruthfulQA. |
| `duplicate_thresholds` | Source, graph, quarantine, and static reject thresholds. |
| `llm_review_policy` | LLM enabled, required, confidence, timeout, and evidence-required rejection policy. |
| `gpu_policy` | Maximum GPU count, actual fixed official GPU count, GPU type, fixed-profile flag, and smoke autosplit flag. |
| `dataset_configs` | FineWeb-Edu sample count, revision, splits, and network fallback policy. |
| `execution_mode_targets` | `local_cpu_smoke`, `gpu_proxy_eval`, and `full_scale_eval` targets. |
| `artifact_limits` | Code and artifact size limits plus required manifest name. |
| `sandbox_limits` | Docker, CPU, memory, PID, timeout, network, and read-only limits. |
| `diagnostics_thresholds` | Activation, gradient, attention, and representation health thresholds. |
| `loss_comparability_policy` | Comparable-loss requirements and redistribution behavior. |

Runtime config rows are audited with `config_key`, `value_json`, `schema_version`, `updated_by`, `updated_at`, `effective_from`, and `enabled`. Active rows are selected by key using enabled rows whose `effective_from` time has arrived, then the newest `effective_from`, `updated_at`, and row id marker wins.

## Environment Settings

| Setting | Purpose |
| --- | --- |
| `PRISM_DATABASE_URL` | Persistent storage location. |
| `PRISM_SHARED_TOKEN` | Shared token for Platform internal calls. Prefer file delivery. |
| `PRISM_SHARED_TOKEN_FILE` | File containing the shared token. |
| `PRISM_PUBLIC_SUBMISSIONS_ENABLED` | Enables public miner submissions. |
| `PRISM_SIGNATURE_TTL_SECONDS` | Replay-protection timestamp window. |
| `PRISM_EPOCH_SECONDS` | PRISM scoring epoch length. |
| `PRISM_MAX_CODE_BYTES` | Maximum submission size. |
| `PRISM_MAX_PARAMETERS` | Maximum accepted model parameters. |
| `PRISM_MAX_LAYERS` | Maximum accepted layer count. |
| `PRISM_SEQUENCE_LENGTH` | Default proxy sequence length. |
| `PRISM_FINEWEB_SAMPLE_COUNT` | FineWeb-style sample count for local defaults. |
| `PRISM_EXECUTION_BACKEND` | Evaluation backend mode. |
| `PRISM_DOCKER_ENABLED` | Enables isolated container execution. |
| `PRISM_DOCKER_BACKEND` | Local or broker-backed execution. |
| `PRISM_DOCKER_BROKER_URL` | Broker URL for remote execution. |
| `PRISM_PLATFORM_EVAL_IMAGE` | Evaluation image. |
| `PRISM_PLATFORM_EVAL_MAX_GPU_COUNT` | Maximum official GPU count. Default and hard max are 8. |
| `PRISM_PLATFORM_EVAL_GPU_COUNT` | Fixed official GPU count. Default is 1. |

Use secret files for shared tokens, broker tokens, and review-provider keys. Do not store real secrets in YAML examples or docs.

## Component Reward Configuration

Default component reward pools are 60% architecture and 40% training. SQL `reward_pools` can override those shares when the values sum to 1.0.

Official score formula defaults are documented in [Scoring and Rewards](../scoring.md). SQL `score_weights` can override supported final score blend values, and runtime validation checks weight sums and benchmark sanity caps.

## FineWeb-Edu and Evaluator Modes

| Mode | Operator use | Official score eligible |
| --- | --- | --- |
| `local_cpu_smoke` | Local CPU wiring check against the tiny fixture. | No |
| `gpu_proxy_eval` | Official proxy contract against configured `sample-10BT` shards. | Yes |
| `full_scale_eval` | Official full-scale contract against configured `sample-100BT` shards. | Yes |

Local smoke verification:

```bash
pytest tests/test_local_cpu_smoke_eval.py -q
```

That command does not run official full-scale training. CI does not run 10B or 100B token training.

## GPU Policy

Official scoring uses a fixed GPU resource profile from `gpu_policy`. The maximum is 8 GPUs. Defaults request 1 GPU from a max of 8. Autosplit is allowed only for non-scoring or development paths such as smoke evaluation.

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

## Review and Holds

PRISM can use static checks, duplicate checks, and optional semantic review. LLM review is evidence-gated: `submit_mermaid` must happen before `submit_verdict`, and deterministic evidence is required for rejection. Suspicion-only results go to quarantine or hold instead of immediate rejection.

Operators should review holds when:

* the new submission is very similar to an existing family
* the metric improvement is near the threshold
* the semantic reviewer cannot confidently decide whether architecture work is novel, matching, variant-linked, or invalid
* the submission appears to improve training while changing architecture structure

## Operator Checklist

Before accepting submissions:

1. Configure persistent storage.
2. Configure shared Platform token delivery through files or another secret manager.
3. Configure broker and evaluation image settings.
4. Set submission size, parameter, layer, and sequence limits.
5. Decide whether semantic and LLM review are advisory or required.
6. Run `pytest tests/test_config.py -q` and `pytest tests/test_local_cpu_smoke_eval.py -q` locally.
7. Submit a known-safe test bundle.
8. Confirm leaderboard, architectures, training variants, current epoch, and weights.

During operation:

* monitor failed evaluations, quarantine, held component reviews, and training current-best changes
* keep epoch settings stable during active rounds
* inspect anomalous architecture matches, architecture variants, and training current-best changes
* back up persistent state
* rotate shared, broker, and review tokens if exposed
* keep untrusted code isolated from host resources

## Security Checklist

* Require hotkey signatures unless running a controlled local test.
* Keep replay windows short.
* Reject oversized submissions.
* Keep network access blocked for evaluation unless explicitly required.
* Keep broker and Platform tokens out of logs.
* Treat LLM review output as sensitive audit data.
* Use low-confidence holds and quarantine for ambiguous ownership changes.
