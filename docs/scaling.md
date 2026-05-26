# Scaling Evaluation

PRISM is built around a simple idea: a small-model challenge is useful only if it rewards signals that are likely to survive scale. The challenge should not overfit to one benchmark, one seed, one short run, or final perplexity alone.

## FineWeb-Edu Modes

PRISM uses the same `prism_run_manifest.v1.json` artifact contract across three evaluator modes.

| Mode | Purpose | Official score eligible | Dataset contract |
| --- | --- | --- | --- |
| `local_cpu_smoke` | Local CPU wiring check using the tiny fixture at `tests/fixtures/tiny_fineweb_fixture.jsonl`. | No | `local_cpu_smoke_fixture` with train, validation, and test records. |
| `gpu_proxy_eval` | Offline official proxy evaluation contract. | Yes | FineWeb-Edu `sample-10BT`, configured local shards, contamination report metadata. |
| `full_scale_eval` | Offline official full-scale evaluation contract. | Yes | FineWeb-Edu `sample-100BT`, configured local shards, contamination report metadata. |

CI does not run 10B or 100B token training. The local CPU smoke path validates evaluator wiring only and cannot produce an official score.

Run the local smoke verification with:

```bash
pytest tests/test_local_cpu_smoke_eval.py -q
```

That command runs a tiny CPU fixture path. It does not run official full-scale training.

## Loss Comparability

Raw final loss is not a cross-architecture ranking signal. Official architecture scoring needs fixed-tokenizer or byte-normalized standardized loss metadata, plus `loss_comparable=true` for the main track. Architecture-baseline-only loss can help compare recipes inside one architecture family, but it does not rank architecture families against each other.

Scientific scaling-law work explains this rule. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) models language-model cross-entropy as a function of parameters, dataset size, and compute. **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) shows that a model can look weak simply because it is under-trained for its parameter count, or look strong because the token budget is better matched. PRISM therefore compares normalized loss trajectories and scaling slopes, not raw final loss in isolation.

When tokenizers differ, `loss/token` can be misleading because token boundaries change the unit being scored. Official comparisons should use a fixed tokenizer or byte-normalized negative log-likelihood. That policy follows the general dataset and reporting guidance in **Datasheets for Datasets** (Gebru et al., 2018/2021) and the FineWeb documentation practices in **The FineWeb Datasets** (Penedo et al., 2024).

## GPU Resource Policy

Official score-eligible GPU runs use a fixed runtime profile. The default policy is:

| Field | Default |
| --- | --- |
| `gpu_policy.max_gpu_count` | 8 |
| `gpu_policy.actual_gpu_count` | 1 |
| `gpu_policy.official_fixed_profile` | `true` |
| `gpu_policy.allocation_policy` | `fixed_official_profile` |
| `gpu_policy.autosplit_allowed_for_smoke` | `true` |

Official scoring uses the fixed profile from SQL runtime config or its defaults. Autosplit is allowed only for non-scoring or development paths such as smoke runs. It is not an official scoring shortcut.

### Broker GPU Contract

PRISM passes the actual lease size to the Platform broker as `gpu_count`. `gpu_count=None` or an omitted `gpu_count` means CPU-only execution and must not create a Kubernetes GPU resource limit. A positive integer requests that many GPUs. Invalid values such as `0`, negatives, booleans, strings, or floats fail validation before placement.

Platform owns `gpu_resource_name`; PRISM does not pass it. In Kubernetes broker mode, Platform maps a positive `gpu_count` to `resources.limits['nvidia.com/gpu']` by default, or to the configured Platform-owned resource name. Prism GPU environment variables, labels, payload metadata, and device IDs are observability and backward-compatibility metadata only. Device IDs are not Kubernetes placement semantics, and this contract is not an arbitrary TPU, AMD, or custom accelerator abstraction.

## Single-Node Torchrun And DDP

PRISM's distributed v1 scope is single-node only. Runs with 1-8 GPUs use single-node torchrun with one process per GPU, including `torchrun --standalone --nnodes=1 --nproc-per-node=1` for a 1 GPU run. Requests above 8 GPUs are rejected. This documents command and environment support, not proof that every submission succeeds on 8 GPUs.

PRISM DDP-wraps default training before running the default loop. Rank 0 writes shared checkpoint and manifest artifacts, including `prism_run_manifest.v1.json`; other ranks participate in training and synchronization without writing those shared artifacts. Custom `train_step` implementations that bypass the default loop must be DDP-safe and rank-aware.

PRISM does not support multi-node distributed training in v1.

## Bad Scaling Predictors

The following signals are weak predictors of frontier-scale performance when used alone:

| Weak signal | Why it is risky |
| --- | --- |
| Early MMLU-style benchmark score | Too noisy and too dependent on tiny-run artifacts. |
| Subjective chat quality | Easy to overfit and hard to compare deterministically. |
| Final perplexity only | Hides instability, spikes, and bad extrapolation. |
| Single seed | Cannot separate improvement from luck. |
| Very short training without extrapolation | Rewards tricks that fail after longer training. |

These signals may be logged, but they should not dominate reward decisions.

## Strong Scaling Predictors

PRISM prioritizes signals that describe the training trajectory and stability envelope.

These predictors are strongest when measured at multiple token budgets. **Deep Learning Scaling is Predictable, Empirically** (Hestness et al., 2017) supports power-law learning trends, while **Broken Neural Scaling Laws** (Caballero et al., 2023) shows that curves can change regime. For that reason, PRISM treats slope as a high-value signal only when the manifest includes enough checkpoints to detect instability, plateaus, and broken extrapolation.

### Smooth Loss Curve

Good architectures and training recipes should produce smooth loss curves:

* no recurring oscillation
* no repeated divergence and recovery pattern
* no late-training instability
* no hidden loss spikes masked by final loss

### Stable Gradient Norms

Gradient norms should remain stable as training progresses and as batch size changes. Silent gradient explosion is a strong negative signal even when final loss looks acceptable.

### No Activation Spikes

Activation spikes are critical for models that may scale beyond 10B parameters. A small model can hide activation problems that later become catastrophic.

### Coherent Scaling Across Sizes

The strongest sign is consistent improvement across multiple proxy sizes. For example:

```text
125M: +2%
350M: +2%
1B:   +2%
```

This is more valuable than a large gain at one size and a regression elsewhere.

## Required Scaling Probes

| Probe | What it checks |
| --- | --- |
| Depth scaling | Stability under deeper and narrower, shallower and wider, and higher layer-count variants. |
| Sequence scaling | Attention stability, positional behavior, KV-cache behavior, memory, and latency under longer context. |
| Batch scaling | Loss spikes, NaNs, overflow, gradient noise scale, optimizer instability, and clipping frequency. |

PRISM's long-term goal is to produce a global view of each architecture and training recipe: not just whether it wins a small benchmark, but whether its learning dynamics suggest that it can scale.

## Scientific Reference Profiles

| Profile | Study | Operational lesson |
| --- | --- | --- |
| Loss vs compute | Kaplan et al., 2020, *Scaling Laws for Neural Language Models* | Fit comparable loss curves rather than using one final checkpoint. |
| Compute-optimal scaling | Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* | Compare models near matched parameter-token-compute regimes. |
| Broken extrapolation | Caballero et al., 2023, *Broken Neural Scaling Laws* | Penalize unreliable slopes and curve-regime changes. |
| Data-constrained scaling | Muennighoff et al., 2023, *Scaling Data-Constrained Language Models* | Track repeated data and flattening marginal returns. |
| Repeated data risk | Hernandez et al., 2022, *Scaling Laws and Interpretability of Learning from Repeated Data* | Watch train/validation gaps and repeated-sample overfitting. |
| Batch and optimizer dynamics | McCandlish et al., 2018, *An Empirical Model of Large-Batch Training* | Use gradient-noise scale and batch sensitivity as recipe diagnostics. |
| FineWeb-Edu validity | Penedo et al., 2024, *The FineWeb Datasets* | Freeze shards, revisions, and contamination metadata for official modes. |
