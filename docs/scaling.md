# Scaling Evaluation

PRISM is built around a simple idea: a small-model challenge is useful only if it rewards signals that are likely to survive scale. The challenge should not overfit to one benchmark, one seed, one short run, or final perplexity alone.

## Bad Scaling Predictors

The following signals are weak predictors of frontier-scale performance when used alone:

| Weak signal | Why it is risky |
| --- | --- |
| Early MMLU-style benchmark score | Too noisy and too dependent on tiny-run artifacts |
| Subjective chat quality | Easy to overfit and hard to compare deterministically |
| Final perplexity only | Hides instability, spikes, and bad extrapolation |
| Single seed | Cannot separate improvement from luck |
| Very short training without extrapolation | Rewards tricks that fail after longer training |

These signals may be logged, but they should not dominate reward decisions.

## Strong Scaling Predictors

PRISM should prioritize signals that describe the training trajectory and stability envelope.

### 1. Smooth Loss Curve

Good architectures and training recipes should produce smooth loss curves:

- no recurring oscillation;
- no repeated divergence/recovery pattern;
- no late-training instability;
- no hidden loss spikes masked by final loss.

### 2. Stable Gradient Norms

Gradient norms should remain stable as training progresses and as batch size changes.

Signals to track:

- mean gradient norm;
- max gradient norm;
- gradient-norm variance;
- gradient clipping frequency;
- gradient noise scale.

Silent gradient explosion is a strong negative signal even when final loss looks acceptable.

### 3. No Activation Spikes

Activation spikes are critical for models that may scale beyond 10B parameters. A small model can hide activation problems that later become catastrophic.

Signals to track:

- activation max / RMS by layer;
- spike frequency;
- residual-stream drift;
- normalization instability;
- overflow or NaN events.

### 4. Coherent Scaling Across Sizes

The strongest sign is consistent improvement across multiple proxy sizes. For example:

```text
125M: +2%
350M: +2%
1B:   +2%
```

This is far more valuable than a large gain at one size and a regression elsewhere.

## Common Failure at Larger Scale

Many architectures look better at 1B but become unstable at 30B+. Common reasons:

- activation variance explodes;
- residual streams drift;
- MoE routing collapses;
- KV cache behavior degrades;
- normalization no longer scales;
- optimizer behavior changes with batch size;
- longer context exposes attention or memory-path failures.

PRISM’s evaluation policy should reduce this risk by probing the architecture and training code along multiple scaling axes.

## Required Scaling Probes

### A. Depth Scaling

Test the same approximate compute budget with:

- deeper and narrower variants;
- shallower and wider variants;
- increasing layer count under controlled parameter limits.

If the submission breaks when depth increases, it is a poor scaling candidate even if the small shallow model is strong.

### B. Sequence Scaling

Evaluate context behavior across increasing sequence lengths:

```text
2k -> 8k -> 32k
```

Even with limited training, this can reveal:

- attention instability;
- KV-cache degradation;
- positional encoding failure;
- memory or latency cliffs.

### C. Batch Scaling

Increase global batch progressively and track:

- loss spikes;
- NaNs;
- overflow;
- gradient noise scale;
- optimizer instability;
- clipping frequency.

Stable batch scaling is a strong sign that a recipe may survive larger training runs.

## Recommended Metric Families

Evaluator containers can report these metrics when available:

```json
{
  "loss_smoothness": 0.94,
  "loss_spike_count": 0,
  "grad_norm_mean": 1.8,
  "grad_norm_max": 4.2,
  "activation_spike_rate": 0.0,
  "scaling_consistency": 0.91,
  "depth_scaling_score": 0.88,
  "sequence_scaling_score": 0.84,
  "batch_scaling_score": 0.9,
  "nan_count": 0,
  "overflow_count": 0
}
```

## Scoring Guidance

Good PRISM rewards should prefer:

- modest but consistent gains across scales;
- stable loss and gradient behavior;
- low activation-spike risk;
- improvements that remain visible under depth, sequence, and batch probes.

They should penalize:

- one-off gains at a single model size;
- final-loss-only improvements with unstable curves;
- high variance across seeds;
- improvements that vanish when batch, depth, or context length changes.

## Evaluation Loop

```mermaid
flowchart LR
    Sub[Submission] --> Size[Size Probes]
    Size --> Depth[Depth Probes]
    Depth --> Seq[Sequence Probes]
    Seq --> Batch[Batch Probes]
    Batch --> Stable[Stability Metrics]
    Stable --> Agent[Semantic Review]
    Agent --> Reward[Reward Decision]
```

PRISM’s long-term goal is to produce a global view of each architecture and training recipe: not just whether it wins a small benchmark, but whether its learning dynamics suggest that it can scale.
