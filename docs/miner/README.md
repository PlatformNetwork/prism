# Miner Guide

## Purpose

PRISM rewards miners whose models learn fast from scratch. You submit two scripts, a model
`architecture.py` and a custom `training.py` loop; the challenge re-executes your loop under a forced
random initialization on locked FineWeb-Edu data and scores it with a prequential bits-per-byte
metric. PRISM fixes the dataset and the evaluation, not your model search space.

## Miner Flow

1. Build a two-script bundle that follows the PRISM contract.
2. Sign and submit the bundle with your miner hotkey.
3. PRISM runs the static sandbox and the OpenRouter LLM hard gate.
4. The validator re-executes your `training.py` under a forced random init on the locked train split.
5. The challenge computes your prequential bits-per-byte score and the held-out delta tie-breaker.
6. Track your leaderboard rank; better learners earn more normalized, dry-run weight.

## The Two-Script Contract

A bundle is a `.zip` (or directory) with two distinct scripts. An optional `prism.yaml` can declare
the entrypoints and the chosen tokenizer:

```yaml
architecture:
  entrypoint: architecture.py
training:
  entrypoint: training.py
tokenizer: gpt2
```

`architecture.py` exposes the model factory:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)
```

`build_model(ctx)` can return any valid `torch.nn.Module` that fits the AST sandbox, the 150M
parameter cap, and the resource limits. It must not read data, open files, touch the network, or
reference the dataset.

`training.py` exposes the training loop you own:

```python
from architecture import build_model

def train(ctx):
    model = build_model(ctx)
    # build the optimizer/schedule, read the locked train split from ctx.data_dir,
    # tokenize, run the loop, handle multi-GPU, and write only under ctx.artifacts_dir.
    ...
```

`train(ctx)` owns the optimizer, the schedule, the dataloading from the read-only locked train split,
the tokenization, the multi-GPU strategy, and the loop. The single-module re-export idiom no longer
satisfies the contract: architecture and training must be two distinct files.

## Context And Limits

`ctx` is a `PrismContext` that supplies the metadata and limits you need:

* `vocab_size` and `max_seq_len` for token-id geometry;
* `max_params` (150M cap);
* `seed`, the forced seed you cannot change;
* `data_dir`, the read-only path to the locked FineWeb-Edu **train** split;
* `artifacts_dir`, the only writable path;
* `world_size`, `rank`, `local_rank`, `device` for the distributed launch;
* `token_budget` / `step_budget` for the compute budget;
* `ctx.build_model()` and `ctx.reference_tokenizer("gpt2" | "llama")` (offline, no network).

PRISM supplies and controls the dataset. You provide model code and a training loop, not your own
data. Read raw text from `ctx.data_dir` and tokenize with your own tokenizer or a pre-staged
reference. Fail closed if the locked data is missing rather than fabricating data.

## Locked Data, No Network

The train split is exposed read-only at `ctx.data_dir`; the `val`/`test` splits are secret and never
exposed to your script. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is no network during training. Do not try to download data,
tokenizers, or weights at runtime.

## Multi-GPU

Your `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>` and exposes `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK`. PRISM is single-node: runs use 1-8 GPUs, and the official scored run uses
`torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path). A correct loop calls
`init_process_group`, wraps the model with DDP or FSDP, shards data per-rank, does rank-0-only
writes, all-reduces metrics, and tears down the process group, and it must also work at
`world_size=1`. Correctness is validated with a static contract check and a gloo multi-rank test.

## The Challenge Computes The Score

PRISM re-executes your loop under a forced random init, captures the single-pass online loss itself,
and writes a challenge-authored `prism_run_manifest.v2.json`. Any value you report and any manifest
you write are ignored. The score is the prequential bits-per-byte: the area under the from-scratch
online loss curve, normalized by the raw UTF-8 bytes consumed, with a held-out delta-over-random-init
tie-breaker. A smuggled pretrained model produces an anomalous step-0 loss and is zeroed; an excessive
train-vs-held-out gap is penalized as memorization.

## Submitting Work

Submit through the public submission route when public submissions are enabled, or through the
Platform proxy in production:

```http
POST /v1/submissions
Content-Type: application/json
```

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

Submission rules:

* The miner hotkey must match the signature; timestamps and nonces protect against replay.
* Submissions must stay within the configured size limit.
* Unsafe imports, network access, arbitrary filesystem access, deserialization escapes, and the
  single-module idiom are rejected at static review, before any GPU work.
* A `reject` from the OpenRouter LLM hard gate is terminal.

## What Improves Your Score

* A model that genuinely drives its from-scratch loss down fast (lower bits-per-byte is better).
* A training loop that uses the compute budget efficiently (the score is compute-normalized, never
  wall-clock).
* A larger held-out delta-over-random-init on the secret val split (the near-tie tie-breaker).
* A small train-vs-held-out gap (a large gap is penalized as memorization).
* Correct, DDP-safe, rank-aware distributed behavior.

## Miner Checklist

Before submitting:

* Ship two distinct scripts: `architecture.py` with `build_model(ctx)` and `training.py` with `train(ctx)`.
* Keep `build_model` pure: no data, files, or network.
* Read only `ctx.data_dir`; write only `ctx.artifacts_dir`.
* Stay under the 150M parameter cap and inside the AST sandbox.
* Make the loop deterministic under the forced seed and correct at `world_size=1`.
* Remove secrets, private endpoints, generated caches, and unrelated files.
