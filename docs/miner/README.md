# Miner Guide

## Purpose

PRISM rewards miners for discovering model architectures and training variants that show useful learning, stability, and scaling behavior. Your submission can introduce a new architecture family, improve training for an existing family, or do both in one full submission.

## Miner Flow

1. Decide whether you are submitting `full`, `architecture_only`, or `training_for_arch` work.
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

Training-only submissions must point to the target architecture family and must not silently change the architecture itself.

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

## Artifact Manifest

Evaluators write `prism_run_manifest.v1.json`. It includes `architecture_graph.json`, `architecture_metadata.v1.json`, run logs, optional metrics artifacts, dataset fingerprints, GPU counts, diagnostics, loss comparability fields, benchmark metadata, and score eligibility flags.

Do not try to make Mermaid text the canonical architecture identity. PRISM derives Mermaid from the canonical graph for review and display.

## Context And Limits

`ctx` exposes the metadata and limits needed to build a valid model:

* vocabulary size
* sequence length
* maximum layers
* maximum parameters
* deterministic seed
* evaluation budget metadata

Avoid hard-coding one tensor shape, batch size, sequence length, or parameter budget. PRISM tests whether the idea can survive scaling probes, not whether it wins one tiny run.

## Local Smoke Check

Validators can run a local CPU smoke check for wiring:

```bash
pytest tests/test_local_cpu_smoke_eval.py -q
```

This command uses a tiny FineWeb-Edu fixture and sets `validation.score_eligible=false`. It does not run official full-scale training and does not create an official score.

## Submitting Work

Submit through the public submission route when public submissions are enabled:

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

* The miner hotkey must match the signature.
* Timestamps and nonces protect against replay.
* Submissions must stay within the configured size limit.
* Unsafe imports, network access, arbitrary filesystem access, and suspicious code paths can be rejected.
* Suspicion without deterministic evidence can be quarantined or held for review.

## Scoring

Defaults are:

| Score or pool | Default |
| --- | ---: |
| Final score blend | 70% architecture, 30% recipe |
| Reward pools | 60% architecture ownership, 40% training ownership |

SQL runtime config can override supported policy values. Official architecture scoring uses reference-recipe architecture signals, not raw final loss from a miner recipe. Training scoring uses architecture-normalized heldout improvement for the target family.

## What Improves Scores

Strong submissions should show:

* smooth loss curves
* stable gradient norms
* absence of activation spikes
* consistent improvements across model size and depth
* stable sequence-length and batch-scaling behavior
* useful training or inference hooks
* original structure rather than small edits to another miner's work

## Miner Checklist

Before submitting:

* Choose the right submission kind.
* Include a valid manifest when using a multi-file project.
* Keep architecture and training files organized.
* Confirm required entrypoints exist.
* Keep the code deterministic and resource-aware.
* Remove secrets, private endpoints, generated caches, and unrelated files.
* Avoid copying existing architecture or training structure.
* Make scaling behavior part of the design.
