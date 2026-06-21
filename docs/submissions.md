# Submission Format

PRISM accepts submissions as a **two-script** bundle: a `.zip` archive (or a directory snapshot) that
contains a model `architecture.py` and a training `training.py`. PRISM fixes the FineWeb-Edu dataset
and the evaluation protocol; it does not fix the model search space beyond the Python contract, the
AST sandbox, the 150M parameter cap, and the resource limits.

The miner owns the model and the training loop. The challenge owns the dataset and the scoring: it
re-executes the loop under a forced random init and computes the metric itself.
A single combined module no longer satisfies the contract.

## The Two-Script Contract

A bundle must contain two **distinct** scripts.

`architecture.py` exposes a model factory:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)
```

`build_model(ctx)` must return a `torch.nn.Module`. The module may use any valid PyTorch structure
that stays inside the AST sandbox, the 150M parameter cap, and the resource limits. It must not read
data, open files, touch the network, or reference the dataset.

`training.py` exposes the miner-owned training loop:

```python
def train(ctx):
    model = ctx.build_model()
    # build the optimizer/schedule, read the locked train split from ctx.data_dir,
    # tokenize, run the loop, handle multi-GPU, write only under ctx.artifacts_dir.
    ...
```

`train(ctx)` owns the optimizer, the schedule, the dataloading from the read-only locked train split,
the tokenization, the multi-GPU strategy, and the loop. It reports progress only through the
challenge-provided logging handle for observability, never as the basis of the score.

An optional `prism.yaml` may declare the entrypoints and the chosen tokenizer:

```yaml
architecture:
  entrypoint: architecture.py
training:
  entrypoint: training.py
tokenizer: gpt2
```

When `prism.yaml` is absent, PRISM uses the default entrypoints (`architecture.py` and `training.py`)
and the default symbols (`build_model` and `train`). When it is present, declared entrypoints are
honored exactly, with no silent fallback. The architecture and training entrypoints must be two
distinct files: **the single-module re-export idiom no longer satisfies the contract**.

## PrismContext

Both scripts receive a `PrismContext`. Key fields and methods:

| Field / method | Meaning |
| --- | --- |
| `vocab_size`, `max_seq_len` | Token-id geometry for the model |
| `max_params` | Hard parameter cap (150M) |
| `seed` | The forced seed (challenge-controlled; the miner cannot change it) |
| `data_dir` | Read-only path to the locked FineWeb-Edu **train** split |
| `artifacts_dir` | The only writable path (rank-0 writes) |
| `device`, `world_size`, `rank`, `local_rank` | Distributed launch geometry |
| `token_budget`, `step_budget` | Compute budget for the run |
| `build_model()` | Helper that builds the model from `architecture.py` |
| `reference_tokenizer(name)` | Loads a pre-staged offline tokenizer (`"gpt2"` or `"llama"`); never touches the network |

The miner does **not** control: the dataset content or splits, the seed/init (forced by the harness),
the scoring, or the held-out evaluation.

## The Challenge Owns The Data And The Score

PRISM re-executes the miner's `training.py` under a forced random init and a fixed seed, and records
the online loss stream itself. The validator owns the seed, the data order, and the metric. Any value
the miner reports is ignored, and the challenge authors the run manifest. See
[Architecture](architecture.md) for the forced-init re-execution flow and
[Scoring and rewards](scoring.md) for the bits-per-byte math.

## Locked FineWeb-Edu Data Plane

The dataset is a pinned FineWeb-Edu subset, split into fixed, disjoint parts:

- `train/` raw text shards, exposed **read-only** at `ctx.data_dir`;
- `val/` and `test/` held-out raw text that is **secret** and **never exposed** to the miner script,
  read only by the challenge scorer.

Delivery is a read-only bind mount on the GPU node. The eval container runs with `network=none`,
`HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, so there is **no network** during training. The
miner reads raw text from `ctx.data_dir` and tokenizes it with its own tokenizer or a pre-staged
reference; it must fail closed if the locked data is missing rather than fabricating data.

## Multi-GPU Contract

The miner's `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>`, exposing `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK`. PRISM is **single-node** only: runs use 1-8 GPUs on a single node, and the official
scored run uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path, since one
physical GPU exists). Requests above 8 GPUs or for multiple nodes are rejected.

A correct `training.py` must:

- `init_process_group` (nccl on GPU) and `set_device(local_rank)`;
- wrap the model with DDP or FSDP and shard data per-rank;
- do rank-0-only logging and artifact writes;
- all-reduce any reported metrics, then `barrier()` and `destroy_process_group()` on exit;
- also work correctly at `world_size=1`.

Multi-GPU correctness is validated off the single physical GPU with a static contract check and a
**gloo** multi-rank functional test (world size 2 and 4 on CPU) that asserts the loss decreases and
parameters stay byte-identical across ranks. True 8-GPU scaling is an accepted, unverifiable
limitation on a one-GPU node.

## Artifact Manifest

The challenge runner writes a challenge-authored `prism_run_manifest.v2.json` from the captured online
loss stream. The manifest is the scoring contract: it records the prequential bits-per-byte score
block, the held-out delta and anti-memorization gap, the compute block (the leased `gpu_count`, world
size, device, and the realized model parameter count), the run provenance, and the data coverage in
bytes.

Any manifest the miner writes is discarded, and any metric the miner reports is ignored: PRISM scores
only the values it computed itself.

## Minimal Example

```text
project.zip
  architecture.py
  training.py
  prism.yaml        # optional
```

`architecture.py`:

```python
import torch

class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.linear = torch.nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.linear(self.embedding(tokens))

def build_model(ctx):
    return TinyModel(ctx.vocab_size)
```

`training.py`:

```python
from architecture import build_model

def train(ctx):
    model = build_model(ctx)
    # construct the optimizer/schedule, read ctx.data_dir, tokenize, run the loop,
    # handle multi-GPU, and write only under ctx.artifacts_dir.
    ...
```

The container resolves `architecture.py::build_model` and `training.py::train`, forces the seed,
launches torchrun, and captures the online loss itself.

For a complete, runnable two-script bundle, see the
[tiny ~1M-parameter example](../examples/tiny-1m/README.md).

## ZIP Safety Rules

ZIP submissions are extracted defensively:

* no path traversal
* no symlinks
* limited file count
* limited total bytes
* only approved text or code suffixes

Unsupported or unsafe archives are rejected before evaluation.
