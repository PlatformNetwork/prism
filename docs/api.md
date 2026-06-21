# API Reference

PRISM exposes public challenge routes and internal Platform routes.

## Public Routes

### `GET /health`

Returns challenge health metadata.

### `GET /version`

Returns challenge version, API version, SDK version, and capabilities.

### `POST /v1/submissions`

Submit a two-script bundle directly to PRISM. In production, miner submissions usually enter through
the Platform proxy.

Request:

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

The direct public route uses miner authentication headers. Platform bridge uploads use the internal
route instead.

### `GET /v1/submissions/history`

Returns daily submission counts over a window.

### `GET /v1/submissions/{submission_id}`

Returns status and score fields:

```json
{
  "id": "...",
  "hotkey": "...",
  "epoch_id": 123,
  "status": "completed",
  "code_hash": "...",
  "created_at": "...",
  "error": null,
  "final_score": 0.72,
  "anti_cheat_multiplier": 1.0
}
```

`final_score` is the challenge-computed prequential bits-per-byte score (a lower bpb yields a higher
`final_score`). The response also carries `q_arch`, `q_recipe`, `diversity_bonus`, and `penalty` as
legacy fields retained for response-schema stability; the live scoring path populates `final_score`.

`status` can be `pending`, `running`, `completed`, `failed`, `rejected`, or `held`. Rejected
submissions failed a static gate, the two-script contract, the LLM hard gate, or duplicate review.
Held submissions are quarantined by the LLM review.

### `GET /v1/leaderboard`

Returns submissions ranked by `final_score` for the current epoch (earliest-commit-wins on a tie, one
entry per hotkey).

### `GET /v1/architectures`

Legacy family-listing endpoint retained for API compatibility.

### `GET /v1/training-variants`

Legacy variant-listing endpoint retained for API compatibility. Optional query parameters:
`architecture_id`, `limit`.

### `GET /v1/epochs/current`

Returns the current epoch id and epoch length.

### `GET /v1/epochs`

Returns recent epochs.

### `GET /v1/health/eval-jobs`

Returns recent eval-job health entries (id, submission id, level, status, attempts).

### `GET /v1/gpu/status`

Returns a GPU-lease summary (total GPUs, active leases, by status, by tier).

## Internal Platform Routes

All internal routes require:

```text
Authorization: Bearer <shared-token>
```

### `GET /internal/v1/get_weights`

Standard Platform challenge contract. Returns normalized, dry-run hotkey weights (one per hotkey, from
that hotkey's best `final_score`). Weights are never written on-chain.

### `POST /internal/v1/bridge/submissions`

Receives Platform-verified submissions.

Headers:

```text
Authorization: Bearer <shared-token>
X-Platform-Verified-Hotkey: <hotkey>
X-Submission-Filename: project.zip
Content-Type: application/zip
```

The body can be raw ZIP bytes or JSON matching `SubmissionCreate`.

### `POST /internal/v1/worker/process-next`

Claims and processes one pending submission through the full pipeline: static gates, the OpenRouter
LLM hard gate, the forced-init re-execution, and prequential bits-per-byte scoring.
