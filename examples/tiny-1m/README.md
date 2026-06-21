# Tiny ~1M-Parameter Two-Script Example

A minimal, valid PRISM v2 submission: a weight-tied ~1.05M-parameter decoder transformer split into
the two-script contract.

## Layout

```text
examples/tiny-1m/
  prism.yaml         # declares the architecture + training entrypoints and the tokenizer
  architecture.py    # exposes build_model(ctx); defines the model only
  training.py        # exposes train(ctx); the miner-owned loop
```

- `architecture.py` exposes `build_model(ctx)` and is pure: it never reads data, opens files, or
  touches the network.
- `training.py` exposes `train(ctx)`: it forces the seed, builds the model via `architecture.py`,
  reads the read-only locked train split from `ctx.data_dir`, tokenizes with the pre-staged gpt2
  reference tokenizer (offline), runs a single-node multi-GPU-safe loop, and writes only under
  `ctx.artifacts_dir`.

## How It Is Scored

The challenge re-executes `train(ctx)` under a forced random initialization on the locked FineWeb-Edu
train split, captures the single-pass online (predict-then-train) loss itself, and computes the
prequential bits-per-byte score with a held-out delta tie-breaker. Any value this submission reports
and any manifest it writes are ignored; the challenge authors `prism_run_manifest.v2.json`.

## Submit

Submit as a `.zip` bundle of this directory through the public route (when enabled) or the Platform
proxy. See [docs/submissions.md](../../docs/submissions.md) and the
[miner guide](../../docs/miner/README.md).
