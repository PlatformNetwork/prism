# Scaling Evaluation

PRISM scores a single-GPU forced-init re-execution, but the submission contract is built to scale: the
miner owns multi-GPU execution, and the challenge keeps the score compute-normalized so hardware does
not change the ranking. This document covers the execution modes, the single-node multi-GPU contract,
the compute budget, and how multi-GPU correctness is validated on one physical GPU.

## Execution Modes

PRISM uses the challenge-authored `prism_run_manifest.v2.json` contract across the official modes:

| Mode | Purpose | Dataset target |
| --- | --- | --- |
| `gpu_proxy_eval` | Default official scored re-execution. | FineWeb-Edu `sample-10BT` locked shards. |
| `full_scale_eval` | Larger official scored re-execution. | FineWeb-Edu `sample-10BT` then `sample-100BT` phases. |

Both modes are score-eligible and run on the locked FineWeb-Edu data, mounted read-only, with
`network=none`, `HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` so there is no network during
training. The retired local-CPU smoke mode no longer exists.

## Single-Node Multi-GPU Contract

The miner's `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>` and exposes `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK`. PRISM is **single-node** only: runs use 1-8 GPUs on one node, and the official scored
run uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path, since one physical
GPU exists). Requests above 8 GPUs or for multiple nodes are rejected.

A correct `training.py`:

- calls `init_process_group` (nccl on GPU) and `set_device(local_rank)`;
- wraps the model with DDP or FSDP and shards data per-rank;
- does rank-0-only logging and artifact writes;
- all-reduces any reported metrics, then `barrier()` and `destroy_process_group()` on exit;
- also works correctly at `world_size=1`.

## Validating Multi-GPU On One GPU

True 8-GPU scaling is an accepted, unverifiable limitation on a one-GPU node. Correctness is validated
in three ways:

1. **Static contract**: the AST contract check verifies the training script uses the distributed
   primitives and a rank-0 write guard, and enforces the single-node bound.
2. **gloo multi-rank functional test**: a CPU **gloo** run at world size 2 and 4 asserts the loss
   decreases and that parameters stay byte-identical across ranks (DDP gradient-sync correctness).
3. **Advisory NCCL `nproc=2`**: an indicative run time-sharing the single GPU, advisory only.

## Compute Budget, Not Wall-Clock

The score is compute-normalized by tokens (and optionally FLOPs), never wall-clock. Wall-clock is only
a safety cap on the run, enforced in layers:

1. a graceful budget at which the runner stops the single-pass loop and scores the partial captured
   stream;
2. a hard watchdog that terminates a loop hanging outside the instrumented iterator;
3. an outer docker/broker timeout set strictly above the graceful budget plus the watchdog grace.

Because the score is compute-normalized, a faster or larger GPU configuration does not change the
ranking; it only changes how much of the budget the run can use.

## Compute Block In The Manifest

The challenge records a typed, observability-only compute block in `prism_run_manifest.v2.json`: the
GPUs actually leased for the scored run (`gpu_count`, which is 1 for the scored `nproc=1` path), the
launch shape (`world_size`, `nproc_per_node`, `device`), and the realized parameter count of the model
the runner actually trained. The block is recorded for observability only; the bits-per-byte
`final_score` never reads `gpu_count`, so there is no GPU-count reward and no multi-GPU scaling bonus.

## Reference Studies

| Area | Study | Operational lesson |
| --- | --- | --- |
| Loss vs compute | Kaplan et al., 2020, *Scaling Laws for Neural Language Models* | Compare comparable loss trajectories, not one final checkpoint. |
| Compute-optimal scaling | Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* | Normalize by tokens/compute so under- or over-training does not skew the score. |
| Large-batch dynamics | McCandlish et al., 2018, *An Empirical Model of Large-Batch Training* | Scaling the batch across ranks must preserve a stable, descending loss. |
| Dataset provenance | Penedo et al., 2024, *The FineWeb Datasets* | Freeze the FineWeb-Edu revision and shards for reproducible official runs. |
