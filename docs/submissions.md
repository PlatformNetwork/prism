# Submission Format

PRISM accepts Python submissions as either a single `.py` file or a multi-file `.zip` project. Multi-file ZIP projects are the preferred format because they let miners separate architecture, training, and inference code cleanly.

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

## Optional Hooks

Miners can customize optimization and inference with optional functions:

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

These hooks allow miners to propose training and inference improvements without necessarily introducing a new architecture family.

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
