# Submission Format

PRISM accepts Python submissions as either a single `.py` file or a multi-file `.zip` project. Multi-file ZIP projects are the preferred format because they let miners separate architecture, optimizer setup, loss computation, training-step logic, and inference code cleanly.

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

| Kind | Use case |
| --- | --- |
| `full` | Submit a new architecture and its training/inference code |
| `architecture_only` | Submit architecture code without claiming a training variant |
| `training_for_arch` | Submit training or inference code for an existing architecture family |

Training improvements must specify the target architecture:

```yaml
kind: training_for_arch
architecture_id: 7ec2c3a8-...
architecture:
  entrypoint: src/model.py
training:
  entrypoint: src/train.py
```

The architecture code must match the target architecture family. This prevents a training-only submission from silently changing the model family.

## Required Python Contract

The architecture entrypoint must expose:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=3e-4, batch_size=2)
```

`ctx` is a `PrismContext` with fields such as:

- `vocab_size`
- `sequence_length`
- `max_layers`
- `max_parameters`
- `seed`

## First-Class Optional Hooks

Miners can customize optimization, inference, loss computation, and training behavior with optional hooks. These hooks are not treated as incidental helper functions: PRISM records whether they are present, whether the evaluator used them, and which files contributed to the training/inference fingerprint.

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

Hook semantics:

| Hook | Purpose | Attribution |
| --- | --- | --- |
| `configure_optimizer` | Custom optimizer, parameter groups, schedules, clipping wrappers | Training owner |
| `inference_logits` | Preferred inference path returning logits | Training/inference owner |
| `infer` | Fallback inference path when `inference_logits` is absent | Training/inference owner |
| `compute_loss` | Custom loss, auxiliary losses, regularization | Training owner |
| `train_step` | Fully custom update step | Training owner |

If both `inference_logits` and `infer` exist, `inference_logits` takes precedence. These hooks allow miners to propose training and inference improvements without necessarily introducing a new architecture family.

The evaluator may emit hook metrics such as:

```json
{
  "hook.configure_optimizer.present": 1.0,
  "hook.configure_optimizer.used": 1.0,
  "hook.inference_logits.present": 1.0,
  "hook.inference_logits.used": 1.0,
  "hook.compute_loss.used": 1.0,
  "hook.train_step.used": 1.0
}
```

## Scaling Metadata

Submissions should be written so the same code can be evaluated across multiple proxy regimes:

- smaller and larger parameter counts;
- shallow and deep variants;
- short and long sequence lengths;
- small and large global batches;
- multiple seeds.

Avoid hard-coding one fixed tensor shape, batch size, context length, or parameter budget. PRISM needs architecture and training code that can be probed for scaling behavior, not just code that wins one tiny run.

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

- no path traversal;
- no symlinks;
- limited file count;
- limited total bytes;
- only approved text/code suffixes.

Unsupported or unsafe archives are rejected before evaluation.
