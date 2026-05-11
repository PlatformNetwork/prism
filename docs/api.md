# API Reference

PRISM exposes public challenge routes and internal Platform routes.

## Public Routes

### `GET /health`

Returns challenge health metadata.

### `GET /version`

Returns challenge version, API version, SDK version, and capabilities.

### `POST /v1/submissions`

Submit a model project directly to PRISM. In production, miner submissions should usually enter through the Platform proxy.

Request:

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

The direct public route uses miner authentication headers. Platform bridge uploads use an internal route instead.

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
  "q_arch": 0.8,
  "q_recipe": 0.6
}
```

### `GET /v1/leaderboard`

Returns ranked submissions for the current epoch.

### `GET /v1/architectures`

Returns known architecture families.

```json
[
  {
    "id": "...",
    "family_hash": "...",
    "owner_hotkey": "...",
    "owner_submission_id": "...",
    "canonical_submission_id": "...",
    "q_arch_best": 0.9,
    "created_at": "...",
    "updated_at": "..."
  }
]
```

### `GET /v1/training-variants`

Optional query parameters:

- `architecture_id`
- `limit`

Returns training variants and current-best state.

### `GET /v1/epochs/current`

Returns the current epoch id and epoch length.

## Internal Platform Routes

All internal routes require:

```text
Authorization: Bearer <shared-token>
```

### `GET /internal/v1/get_weights`

Standard Platform challenge contract. Returns normalized hotkey weights.

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

Claims and processes one pending submission.

### `POST /internal/v1/worker/poll`

Polls remote jobs. This is retained for compatibility with asynchronous job flows.

## Validator Assignment Routes

These routes support validator-assignment operation.

### `POST /internal/v1/validators/assignments/next`

Headers:

```text
X-Validator-Hotkey: <validator-hotkey>
```

Returns an assignment or `null`.

### `POST /internal/v1/validators/assignments/{assignment_id}/accept`

Marks an assignment as accepted.

### `POST /internal/v1/validators/assignments/{assignment_id}/reject`

Request:

```json
{
  "reason": "busy"
}
```

Rejects an assignment and requeues the submission if attempts remain.

### `POST /internal/v1/validators/assignments/{assignment_id}/result`

Request:

```json
{
  "metrics": {
    "q_arch": 0.8,
    "q_recipe": 0.7,
    "train_loss": 1.0
  }
}
```

Finalizes the assignment and records scores.

### `POST /internal/v1/validators/assignments/expire`

Expires stale assignments and requeues or fails submissions according to the configured attempt limit.
