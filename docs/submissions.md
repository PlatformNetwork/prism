# Submission Format

PRISM accepts Python submissions as a single `.py` file or as a multi-file `.zip` project. ZIP projects are preferred because they let miners mark which files belong to architecture discovery and which files belong to training or inference improvements.

PRISM runs two competitions from the same submission surface:

1. Architecture discovery, for the first useful architecture family and later canonical architecture versions.
2. Training and recipe improvement, for optimizer, loss, inference, and train-step improvements on an existing architecture family.

## Project Manifest

A ZIP project may include `prism.yaml` or `prism.yml` at the project root.

```yaml
kind: full
architecture:
  entrypoint: src/model.py
  files:
    - src/layers.py
training:
  entrypoint: src/train.py
  files:
    - src/losses.py
```

## Project Kinds

| Kind | Use case | Competition effect |
| --- | --- | --- |
| `full` | Submit a new architecture with training or inference code. | Can create or update an architecture family and can create a training variant. |
| `architecture_only` | Submit architecture code without claiming a training variant. | Architecture competition only. Training ownership is not claimed. |
| `training_for_arch` | Submit training or inference code for an existing architecture family. | Training competition only for the target architecture family. |

Training submissions must specify the target architecture:

```yaml
kind: training_for_arch
architecture_id: 7ec2c3a8-example
architecture:
  entrypoint: src/model.py
training:
  entrypoint: src/train.py
```

The architecture code must match the target architecture family. A training-only submission cannot silently change the model family.

## Required Python Contract

The architecture entrypoint must expose:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=3e-4, batch_size=2)
```

`ctx` is a `PrismContext` with fields such as:

* `vocab_size`
* `sequence_length`
* `max_layers`
* `max_parameters`
* `seed`

## First-Class Optional Hooks

Miners can customize optimization, inference, loss computation, and training behavior with optional hooks. PRISM records whether each hook exists, whether the evaluator used it, and which files contributed to the training fingerprint.

```python
def configure_optimizer(model, recipe, ctx):
    ...

def inference_logits(model, batch, ctx):
    ...

def infer(model, batch, ctx):
    ...

def compute_loss(model, batch, ctx):
    ...

def train_step(model, batch, optimizer, ctx):
    ...
```

| Hook | Purpose | Attribution |
| --- | --- | --- |
| `configure_optimizer` | Optimizer, parameter groups, schedules, clipping wrappers. | Training owner |
| `inference_logits` | Preferred inference path returning logits. | Training or inference owner |
| `infer` | Fallback inference path when `inference_logits` is absent. | Training or inference owner |
| `compute_loss` | Custom loss, auxiliary losses, regularization. | Training owner |
| `train_step` | Fully custom update step. | Training owner |

If both `inference_logits` and `infer` exist, `inference_logits` takes precedence.

## Artifact Manifest

Official and smoke evaluators write `prism_run_manifest.v1.json`. The manifest is the scoring contract for artifacts and metrics, not a free-form log.

Required artifact references include:

| Manifest field | Purpose |
| --- | --- |
| `artifacts.architecture_graph` | Canonical `architecture_graph.json` used for architecture identity. |
| `artifacts.architecture_metadata` | Source-free metadata about the accepted architecture version. |
| `artifacts.run_log` | Evaluator log artifact. |
| `artifacts.metrics` | Optional metrics artifact when the evaluator writes one. |

The manifest also carries `mode`, model IDs, dataset fingerprints, GPU counts, diagnostics, loss comparability metadata, benchmark metadata, and validation flags. `local_cpu_smoke` manifests set `validation.score_eligible=false`, so they can validate wiring but cannot produce an official score.

## Scaling Metadata

Submissions should be written so the same code can be evaluated across multiple proxy regimes:

* smaller and larger parameter counts
* shallow and deep variants
* short and long sequence lengths
* small and large global batches
* multiple seeds

Avoid hard-coding one tensor shape, batch size, context length, or parameter budget. PRISM needs architecture and training code that can be probed for scaling behavior, not just code that wins one tiny run.

## Example Multi-File Project

```text
project.zip
  prism.yaml
  src/
    model.py
    layers.py
    train.py
    losses.py
```

`prism.yaml`:

```yaml
kind: full
architecture:
  entrypoint: src/model.py
  files:
    - src/layers.py
training:
  entrypoint: src/train.py
  files:
    - src/losses.py
```

`src/model.py`:

```python
import torch
from train import recipe

class TinyBlock(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 64)
        self.proj = torch.nn.Linear(64, vocab_size)

    def forward(self, tokens):
        return self.proj(self.emb(tokens))

def build_model(ctx):
    return TinyBlock(ctx.vocab_size)

def get_recipe(ctx):
    return recipe(ctx)
```

`src/train.py`:

```python
from prism_challenge.evaluator.interface import TrainingRecipe

def recipe(ctx):
    return TrainingRecipe(learning_rate=3e-4, batch_size=2)
```

## ZIP Safety Rules

ZIP submissions are extracted defensively:

* no path traversal
* no symlinks
* limited file count
* limited total bytes
* only approved text or code suffixes

Unsupported or unsafe archives are rejected before evaluation.
