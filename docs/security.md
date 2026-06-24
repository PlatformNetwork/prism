# Security Model

PRISM evaluates untrusted miner code. Its security model assumes submissions may be malicious and
layers identity verification, a static AST sandbox, an OpenRouter LLM hard gate, a duplicate check,
and a forced-init re-execution that makes the common cheats inert rather than merely detected.

## Identity and Upload Security

Miner-facing uploads are handled by BASE, which verifies hotkey identity, signatures, timestamps,
nonces, request freshness, and challenge routing before forwarding the payload to PRISM:

```text
POST /internal/v1/bridge/submissions
```

PRISM trusts the verified hotkey header only on authenticated internal requests.

## Internal Authentication

Internal endpoints require the shared BASE challenge token:

```text
Authorization: Bearer <shared-token>
```

The token is read from `PRISM_SHARED_TOKEN`, `CHALLENGE_SHARED_TOKEN`, or a configured secret file.
Production configs should use secret files instead of inline values.

## Static Sandbox

Before any GPU work, PRISM runs the static gates over both submission scripts, in order:

1. **AST hard-blocks** over `architecture.py` and `training.py`: no `os`, `sys`, `subprocess`,
   `socket`, network clients, `pickle`/`torch.load` of untrusted paths, `ctypes`, dynamic
   `importlib`, `eval`/`exec`/`compile`, attribute escapes (`__globals__`, `__reduce__`,
   `__class__` walking), or filesystem writes outside `artifacts_dir`.
2. **Forced-seed parameter cap**: the challenge instantiates `build_model(ctx)` under the forced seed
   in a bounded child process and rejects a model over the 150M parameter cap (counting realized,
   first-forward shapes) before any GPU work.
3. **Multi-GPU static contract**: the training script must use the distributed primitives and a
   rank-0 write guard, and a `gpu_count > 8` or multi-node request is rejected.

A rejection at any static gate is terminal and skips the LLM review entirely.

## Forced-Init Re-Execution (Anti-Cheat Core)

The challenge re-executes the miner's `training.py` under a **forced random init** with a fixed,
challenge-controlled seed and deterministic flags set **before** any miner code runs. It feeds the
model fresh, single-pass batches from the locked train split in a challenge-controlled order and
records the online loss itself. This neutralizes the three cheat classes:

- **No pretrained weights**: forced random init makes smuggled weights inert; an impossibly low
  step-0 loss is flagged as an anomaly and zeroes the score; the container runs `network=none` and
  the sandbox blocks IO/network/deserialization escapes.
- **No metric manipulation**: the challenge computes the metric from the loss stream it captured, so
  any miner-reported number and any miner-written manifest are ignored. The fixed seed and data order
  make runs reproducible (same submission + same seed + same data yields the same score within
  tolerance).
- **No memorization**: the `val`/`test` splits are secret and **never exposed** to the miner script;
  an excessive train-vs-held-out gap penalizes the score.

## OpenRouter LLM Hard Gate

After the static gates pass, a strong LLM reviews both scripts as a **hard gate**. The reviewer runs
on OpenRouter with `openai/gpt-4o` by default, using the key from the Docker secret mounted at
`/run/secrets/openrouter_api_key`. It checks architecture-to-training coherence, cheating and
obfuscation (smuggled weights, hidden network, dead/no-op loops, metric gaming), and dangerous
operations the static sandbox might miss.

The verdict is parsed as structured JSON. A `reject` is terminal: the pipeline stops **before any GPU
work**, and the submission ends `rejected`. A transient error or an ambiguous result fails closed to
a held quarantine rather than silently allowing. The gate is enabled by default; only a
configuration-disabled gate is skipped.

## Locked Data, No Network

The dataset is a pinned FineWeb-Edu subset. The train split is mounted **read-only** at `ctx.data_dir`
and is the only data the miner script can see; the `val`/`test` splits are secret. The eval container
runs with `network=none`, `HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, so there is **no network**
during training. The miner cannot download data, tokenizers, or weights at runtime.

## Duplicate Review

PRISM stores source snapshots and runs a deterministic duplicate check. An exact-source-hash
duplicate is rejected, and a borderline-similarity quarantine is folded into a terminal rejection at
ingress (there is no operator hold-resolution surface; the v1-NAS component-review and ownership
machinery was decommissioned).

## ZIP Hardening

ZIP extraction rejects symlinks, path traversal, unsafe paths, unsupported file types, and excessive
file counts or total bytes, before code review begins.

## Execution Isolation

PRISM does not execute submitted code inside the API or worker process. The scored run happens in a
broker-backed container that is non-root, has a read-only rootfs except `artifacts_dir`, uses
`network=none` and `no-new-privileges`, and is bounded by CPU, memory, PID, and wall-clock caps. The
host-side static instantiation and held-out scoring run in bounded child processes and use
`weights_only=True` for any deserialization.

## Dry-Run Weights

Weights are normalized per hotkey from the bits-per-byte `final_score` and exposed only via
`get_weights`. They are always **dry-run** and are never written on-chain.

## Reference Studies

| Area | Study | PRISM implication |
| --- | --- | --- |
| Supply-chain attacks | Gu, Dolan-Gavitt, and Garg, 2017/2019, *BadNets* | Treat submitted code and artifacts as adversarial even when metrics look normal. |
| Untrusted deserialization | Common pickle/`torch.load` RCE guidance | Load any host-side artifact with `weights_only=True` and only the challenge-recorded path. |
| Dataset provenance | Penedo et al., 2024, *The FineWeb Datasets* | Pin the data revision and shard hashes; keep held-out splits secret. |

## Operational Guidance

* Use real secret files in production, not inline tokens.
* Keep public submissions disabled when PRISM is deployed only behind BASE.
* Keep the eval container on `network=none` and the rootfs read-only except `artifacts_dir`.
* Keep the OpenRouter LLM hard gate enabled for production operation.
* Monitor rejected, held, failed, and completed submissions separately.
