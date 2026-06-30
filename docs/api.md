# API Reference

PRISM exposes public challenge routes and internal BASE routes.

## Public Routes

### `GET /health`

Returns challenge health metadata.

### `GET /version`

Returns challenge version, API version, SDK version, and capabilities.

### `POST /v1/submissions`

Submit a two-script bundle directly to PRISM. In production, miner submissions usually enter through
the BASE proxy.

Request:

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

The direct public route uses miner authentication headers. BASE bridge uploads use the internal
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

Architecture-lab leaderboard grouped by architecture family, ranked by `best_final_score`
descending. Optional query parameter `epoch_id` scopes to architectures with a completed submission
in that epoch; omitting it resolves to the most-recent non-empty epoch.

```json
{
  "epoch_id": 42,
  "architectures": [
    {
      "rank": 1,
      "architecture_id": "...",
      "arch_hash": "...",
      "name": "Rotary MoE v3",
      "owner_hotkey": "...",
      "best_final_score": 1.2345,
      "best_submission_id": "...",
      "variant_count": 3,
      "submission_count": 7,
      "updated_at": "..."
    }
  ]
}
```

`name` is the miner-declared, deterministically moderated architecture name (may be `null`).

### `GET /v1/architectures/{architecture_id}`

Returns one architecture's lab detail (name, owner, best score/submission, variant and submission
counts, `first_seen_at`, `updated_at`). `404` if the architecture does not exist.

### `GET /v1/architectures/{architecture_id}/variants`

Returns the architecture's training-script variants (best first). `404` if the architecture does
not exist; an empty `variants` array is valid.

### `GET /v1/architectures/{architecture_id}/report`

Returns the cached LLM auto-report, generated lazily and non-blockingly through the OpenRouter
gateway and grounded only in measured facts. `report.status` is `ready`, `pending`, or
`unavailable`; `content` may be `null` when not `ready`. `404` if the architecture does not exist.

### `GET /v1/submissions/{submission_id}/curve`

Returns the persisted loss curve (`online_loss` + `covered_bytes_cumulative`, downsampled to at most
500 points with the first and last samples preserved), the prequential bits-per-byte scalars, and
the reconciled compute profile (`estimated_flops`, `gpu_hours`, peak VRAM/RSS, wall-clock). `404` if
the submission has no persisted curve.

### `GET /v1/epochs/current`

Returns the current epoch id and epoch length.

### `GET /v1/epochs`

Returns recent epochs.

### `GET /v1/health/eval-jobs`

Returns recent eval-job health entries (id, submission id, level, status, attempts).

### `GET /v1/gpu/status`

Returns a GPU-lease summary (total GPUs, active leases, by status, by tier).

## Internal BASE Routes

All internal routes require:

```text
Authorization: Bearer <shared-token>
```

### `GET /internal/v1/get_weights`

Standard BASE challenge contract. Returns normalized, dry-run hotkey weights (one per hotkey, from
that hotkey's best `final_score`). Weights are never written on-chain.

### `POST /internal/v1/bridge/submissions`

Receives BASE-verified submissions.

Headers:

```text
Authorization: Bearer <shared-token>
X-Base-Verified-Hotkey: <hotkey>
X-Submission-Filename: project.zip
Content-Type: application/zip
```

The body can be raw ZIP bytes or JSON matching `SubmissionCreate`.

### `POST /internal/v1/worker/process-next`

Claims and processes one pending submission through the full pipeline: static gates, the OpenRouter
LLM hard gate, the forced-init re-execution, and prequential bits-per-byte scoring.
