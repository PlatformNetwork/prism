# Validator Guide

## Purpose

PRISM lets validators operate an "ability to learn" challenge for BASE. Validators accept signed
miner submissions, run the static sandbox and the OpenRouter LLM hard gate, re-execute the miner's
training loop under a forced random init on locked data, compute the prequential bits-per-byte score
themselves, and expose normalized dry-run weights to BASE.

## Responsibilities

Validators are responsible for:

* accepting only signed miner submissions and enforcing replay protection and size limits;
* keeping evaluation isolated from the validator host (broker-backed containers, `network=none`);
* keeping the locked `val`/`test` splits secret and never exposing them to a miner script;
* forcing the seed and deterministic flags so runs reproduce;
* computing the score from the challenge-owned capture, never trusting miner-reported numbers;
* protecting shared BASE, broker, and OpenRouter tokens;
* monitoring scoring, rejections, failures, quarantine, and exported weights.

## Evaluation Lifecycle

1. A miner submits a signed two-script bundle.
2. PRISM validates the hotkey, timestamp, nonce, and submission size.
3. The bundle is resolved into the two-script contract and inspected by the AST sandbox.
4. The forced-seed `build_model` instantiation enforces the 150M parameter cap.
5. The multi-GPU static contract and single-node bound are checked.
6. The OpenRouter LLM hard gate reviews both scripts; a `reject` is terminal before any GPU work.
7. The challenge re-executes `training.py` under a forced random init on the locked train split and
   captures the single-pass online loss itself.
8. PRISM computes the prequential bits-per-byte score, the held-out delta tie-breaker, and the
   anti-memorization gap, and writes `prism_run_manifest.v2.json`.
9. Scores persist; the leaderboard ranks by `final_score`; BASE reads normalized dry-run weights.

## Runtime Configuration

PRISM settings use the `PRISM_` prefix and accept compatible `CHALLENGE_` values for the fields that
declare them. These settings bootstrap the process and provide fallback defaults. Runtime challenge
policy is SQL-first.

Precedence is:

```text
SQL active value → env/Pydantic default → schema default
```

SQL runtime config can override supported policy values. Official runtime config fails closed when an
active SQL value is invalid.

## SQL Runtime Config Keys

The validated SQL policy rows are:

| Key | Policy area |
| --- | --- |
| `reward_pools` | Validated weight-split row (compat). Live weights are normalized per hotkey from the bits-per-byte `final_score`. |
| `score_weights` | Validated score-weight row (compat). The live primary score is challenge-computed prequential bits-per-byte. |
| `benchmark_weights` | Validated benchmark-mix row (compat); not part of the live bits-per-byte score. |
| `duplicate_thresholds` | Source/graph/quarantine/static-reject duplicate thresholds. |
| `llm_review_policy` | OpenRouter LLM hard-gate enable/required, base URL, model, confidence, timeout, evidence policy. |
| `gpu_policy` | Maximum GPU count, actual fixed GPU count, GPU type, and the fixed-profile flag. |
| `dataset_configs` | Locked FineWeb-Edu sample count, frozen revision, and split names. |
| `execution_mode_targets` | The `gpu_proxy_eval` and `full_scale_eval` token/GPU targets. |
| `artifact_limits` | Code and artifact size limits plus the required `prism_run_manifest.v2.json` name. |
| `sandbox_limits` | Docker, CPU, memory, PID, timeout, network, and read-only limits. |
| `diagnostics_thresholds` | Activation, gradient, attention, and representation health thresholds. |
| `loss_comparability_policy` | Comparable-loss requirements and byte-normalized fallback. |

Runtime config rows are audited with `config_key`, `value_json`, `schema_version`, `updated_by`,
`updated_at`, `effective_from`, and `enabled`. Active rows are selected by key, using enabled rows
whose `effective_from` has arrived, with the newest `effective_from`, `updated_at`, and row id
winning.

## Environment Settings

| Setting | Purpose |
| --- | --- |
| `PRISM_DATABASE_URL` | Persistent SQLite storage location. |
| `PRISM_SHARED_TOKEN` / `PRISM_SHARED_TOKEN_FILE` | Shared token for BASE internal calls (prefer file delivery). |
| `PRISM_PUBLIC_SUBMISSIONS_ENABLED` | Enables the direct public miner submission route. |
| `PRISM_SIGNATURE_TTL_SECONDS` | Replay-protection timestamp window. |
| `PRISM_EPOCH_SECONDS` | Scoring epoch length. |
| `PRISM_MAX_CODE_BYTES` | Maximum submission size. |
| `PRISM_MAX_PARAMETERS` | Hard parameter cap (default 150M). |
| `PRISM_BASE_EVAL_IMAGE` | CI-published `prism-evaluator` image (ships sentencepiece + offline tiktoken). |
| `PRISM_BASE_EVAL_DATA_DIR` | Read-only locked FineWeb-Edu **train** mount. |
| `PRISM_BASE_EVAL_VAL_DATA_DIR` | Secret held-out **val** split (scorer-only; never mounted into the eval container). |
| `PRISM_BASE_EVAL_MAX_GPU_COUNT` | Maximum GPU count (default and hard max 8). |
| `PRISM_BASE_EVAL_GPU_COUNT` | Scored GPU count (default 1; the `nproc=1` path). |
| `PRISM_DISTRIBUTED_CONTRACT_POLICY` | `reject` / `flag` / `off` for the multi-GPU static contract. |
| `PRISM_LLM_REVIEW_ENABLED` | Enables the OpenRouter LLM hard gate (default on). |
| `PRISM_OPENROUTER_MODEL` | LLM model (default `anthropic/claude-opus-4.8`). |

Use secret files for the shared token, broker token, and OpenRouter key. Do not store real secrets in
YAML examples or docs.

## FineWeb-Edu And Execution Modes

| Mode | Operator use | Dataset target |
| --- | --- | --- |
| `gpu_proxy_eval` | Default official scored re-execution. | FineWeb-Edu `sample-10BT` locked shards. |
| `full_scale_eval` | Larger official scored re-execution. | FineWeb-Edu `sample-10BT` then `sample-100BT` phases. |

Both modes are score-eligible and run on the locked, read-only FineWeb-Edu data with `network=none`.
The retired local-CPU smoke mode is gone.

## GPU Policy

Official scoring uses a fixed GPU profile from `gpu_policy`. The maximum is 8 GPUs, and the scored run
uses 1 GPU (`torchrun --standalone --nnodes=1 --nproc-per-node=1`). PRISM is single-node only.

## Public Miner Surface

```http
POST /v1/submissions
GET /v1/submissions/{submission_id}
GET /v1/leaderboard
GET /v1/architectures
GET /v1/training-variants
GET /v1/epochs/current
```

## BASE Contract

```http
GET /health
GET /version
GET /internal/v1/get_weights
Authorization: Bearer <shared-token>
X-Base-Challenge-Slug: prism
```

`get_weights` returns one normalized weight per hotkey (best submission per hotkey). Weights are
always dry-run and are never written on-chain.

## Review And Quarantine

PRISM uses the static AST sandbox, the forced-seed parameter cap, the multi-GPU static contract, the
OpenRouter LLM hard gate, and a deterministic duplicate check. A `reject` from any static gate or the
LLM gate is terminal before any GPU work. A borderline duplicate is folded into a terminal rejection
at ingress: there is no operator hold-resolution surface (the v1-NAS component-review and ownership
machinery was decommissioned).

## Operator Checklist

Before accepting submissions:

1. Configure persistent SQLite storage.
2. Configure shared-token delivery through files or a secret manager.
3. Configure the broker, the CI-published evaluator image, and the read-only locked-data mounts.
4. Set submission size and parameter limits.
5. Provide the OpenRouter key (the LLM hard gate is on by default).
6. Run `pytest tests/test_config.py -q` and the scoring/harness suites locally.
7. Submit a known-safe two-script bundle and confirm the leaderboard and `get_weights`.

During operation:

* monitor rejected, failed, quarantined, and completed submissions separately;
* keep epoch settings stable during active rounds;
* keep the `val`/`test` splits secret and the eval container on `network=none`;
* keep broker, BASE, and OpenRouter tokens out of logs;
* confirm no on-chain weight-setter exists (weights stay dry-run).

## Security Checklist

* Require hotkey signatures unless running a controlled local test.
* Keep replay windows short and reject oversized submissions.
* Keep the eval container on `network=none` and the rootfs read-only except `artifacts_dir`.
* Keep the locked `val`/`test` splits out of any miner-visible path, fixture, or log.
* Treat LLM review output as sensitive audit data.
