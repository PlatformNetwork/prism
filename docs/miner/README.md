# Miner Guide

## Purpose

PRISM rewards miners for discovering model architectures and training variants that show useful
learning, stability, and scaling behavior. Your submission can introduce a new architecture family
or improve the training, inference, loss, or optimizer behavior of an existing family.

## Miner Flow

1. Decide whether you are submitting a full architecture, architecture-only variant, or training
   improvement for an existing architecture.
2. Build a project bundle that follows the PRISM contract.
3. Sign and submit the bundle with your miner hotkey.
4. Track evaluation status and leaderboard movement.
5. Inspect whether rewards were attributed to architecture ownership, training ownership, or both.
6. Submit improved variants only when they provide meaningful, scalable gains.

## Submission Kinds

| Kind | Use case |
| --- | --- |
| `full` | Submit a new architecture and its training or inference code. |
| `architecture_only` | Submit a model architecture without claiming a training variant. |
| `training_for_arch` | Improve optimizer, loss, inference, or train-step behavior for an existing architecture family. |

Training-only submissions must point to the target architecture family and must not silently change
the architecture itself.

## Project Manifest

ZIP projects can include `prism.yaml` or `prism.yml` at the root:

```yaml
kind: full
architecture:
  entrypoint: src/model.py
training:
  entrypoint: src/train.py
```

The architecture entrypoint must expose:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=3e-4, batch_size=2)
```

Optional hooks can claim training or inference improvements:

```python
def configure_optimizer(model, recipe, ctx):
    ...

def inference_logits(model, batch, ctx):
    ...

def compute_loss(model, batch, ctx):
    ...

def train_step(model, batch, optimizer, ctx):
    ...
```

## Context And Limits

`ctx` exposes the metadata and limits needed to build a valid model:

- vocabulary size;
- sequence length;
- maximum layers;
- maximum parameters;
- deterministic seed;
- evaluation budget metadata.

Avoid hard-coding one fixed tensor shape, batch size, sequence length, or parameter budget. PRISM
tests whether the idea can survive scaling probes, not whether it wins one tiny run.

## Submitting Work

Submit through the public submission route:

```http
POST /v1/submissions
Content-Type: application/json
```

```json
{
  "hotkey": "5Abc...",
  "signature": "<sr25519-signature>",
  "timestamp": 1760000000,
  "nonce": "unique-nonce",
  "code": "<base64-or-text-submission-payload>"
}
```

Submission rules:

- The miner hotkey must match the signature.
- Timestamps and nonces protect against replay.
- Submissions must stay within the configured size limit.
- Unsafe imports, network access, arbitrary filesystem access, and suspicious code paths can be
  rejected.
- Low-confidence ownership transfers can be held for review.

## Tracking Results

Read submission status:

```http
GET /v1/submissions/{submission_id}
```

Read current leaderboard:

```http
GET /v1/leaderboard
```

Read architecture families:

```http
GET /v1/architectures
```

Read training variants:

```http
GET /v1/training-variants
```

Read the active epoch:

```http
GET /v1/epochs/current
```

## Scoring

PRISM combines architecture quality and recipe quality:

```text
S_prism = 0.7 * Q_arch + 0.3 * Q_recipe
```

Component rewards can split final credit between architecture owners and training owners. This lets
one miner discover a useful family while another miner earns rewards for a genuine optimizer, loss,
inference, or training-step improvement.

## What Improves Scores

Strong submissions should show:

- smooth loss curves;
- stable gradient norms;
- absence of activation spikes;
- consistent improvements across model size and depth;
- robust sequence-length and batch-scaling behavior;
- useful training or inference hooks;
- original structure rather than small edits to another miner's work.

## Miner Checklist

Before submitting:

- Choose the right submission kind.
- Include a valid manifest when using a multi-file project.
- Keep architecture and training files organized.
- Confirm required entrypoints exist.
- Keep the code deterministic and resource-aware.
- Remove secrets, private endpoints, generated caches, and unrelated files.
- Avoid copying existing architecture or training structure.
- Make scaling behavior part of the design, not an afterthought.
