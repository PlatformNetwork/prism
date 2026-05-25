# Scoring and Rewards

PRISM rewards two kinds of contributions:

1. Architecture discovery, meaning useful architecture families and canonical architecture versions.
2. Training and recipe improvement, meaning better optimizer, loss, inference, and train-step behavior for a known architecture family.

This split makes PRISM a decentralized NAS system rather than a single leaderboard for monolithic submissions.

## Default Reward Pools

The default component reward pools are:

| Pool | Default share |
| --- | ---: |
| Architecture ownership | 60% |
| Training ownership | 40% |

These are runtime policy defaults. SQL runtime config can override supported policy values. The active SQL row is read first, then env or Pydantic defaults, then schema defaults where a schema field has its own default.

## Default Final Score Blend

The default leaderboard compatibility score is:

```text
S_prism = 70% * Q_arch + 30% * Q_recipe
```

The SQL `score_weights` runtime config can override `final_architecture_weight` and `final_recipe_weight` as long as they sum to 1.0. Official component formulas still require the implemented component names and default percentages below.

## Architecture Official Formula

Architecture scoring is separate from miner-submitted training recipes. Official architecture evaluation uses manifest-backed reference-recipe architecture signals so a custom recipe cannot become the cross-architecture ranking signal.

| Component | Default share | Manifest source |
| --- | ---: | --- |
| `learning_scaling_dynamics` | 35% | `loss.relative_loss_reduction` and `metrics.learning_speed_slope` |
| `standardized_lm_quality` | 20% | `loss.standardized_eval_loss` |
| `compute_efficiency` | 15% | `metrics.estimated_flops` |
| `parameter_efficiency` | 10% | `metrics.parameter_count` |
| `diagnostics_health` | 10% | `metrics.diagnostics` |
| `robustness_stability` | 5% | `metrics.loss_vs_tokens` |
| `benchmark_sanity` | 5% | `metrics.benchmark_scores` |

Raw final loss is not a cross-architecture ranking signal. Architecture official scoring requires fixed-tokenizer or byte-normalized standardized loss metadata. A manifest that reports only architecture-baseline loss is not enough for official architecture ranking.

### Scientific basis

This formula follows the main reproducibility lessons from NAS and scaling-law literature:

* **NAS-Bench-101** (Ying, Klein, Real, Christiansen, Murphy, Hutter, 2019) and **NAS-Bench-201** (Dong and Yang, 2020) show that NAS comparisons need fixed search spaces, canonical architecture identity, repeated evaluation, and controlled protocols. PRISM therefore separates architecture identity from source text and keeps architecture scoring separate from miner-specific training recipes.
* **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) and **Training Compute-Optimal Large Language Models / Chinchilla** (Hoffmann et al., 2022) show that language-model loss depends jointly on parameters, tokens, and compute. PRISM therefore gives the largest architecture share to learning and scaling dynamics instead of raw final loss.
* **Deep Learning Scaling is Predictable, Empirically** (Hestness et al., 2017) supports fitting learning curves across budgets, while **Broken Neural Scaling Laws** (Caballero et al., 2023) warns that extrapolation can fail. PRISM logs loss-vs-token trajectories and keeps benchmark sanity capped rather than allowing a single extrapolation or benchmark to dominate.
* **Green AI** (Schwartz, Dodge, Smith, and Etzioni, 2019) and **MLPerf Training Benchmark** (Mattson et al., 2020) motivate explicit compute-efficiency reporting. PRISM therefore scores compute efficiency and requires fixed official GPU profiles for comparable official runs.

## Training Official Formula

Training scoring ranks improvements within a target architecture family.

| Component | Default share | Manifest source |
| --- | ---: | --- |
| `architecture_normalized_heldout_improvement` | 30% | `loss.architecture_normalized_heldout_improvement` |
| `learning_stability_dynamics` | 25% | `metrics.loss_vs_tokens` and `metrics.learning_speed_slope` |
| `benchmark_sanity` | 15% | `metrics.benchmark_scores` |
| `compute_efficiency` | 10% | `metrics.estimated_flops` |
| `reproducibility_stability` | 10% | `metrics.benchmark_noise_metadata` |
| `robustness_failure_behavior` | 5% | `validation` and `metrics.diagnostics` |
| `artifact_completeness` | 5% | `artifacts` |

Raw final loss is metadata only for training scoring. The primary loss signal is architecture-normalized heldout improvement.

Training scoring follows the hyperparameter-optimization and NAS reproducibility findings of **Random Search and Reproducibility for Neural Architecture Search** (Li and Talwalkar, 2019): optimizer and recipe improvements must be evaluated within a controlled architecture context so recipe quality is not mistaken for architecture ownership. Gradient and batch behavior are informed by **An Empirical Model of Large-Batch Training** (McCandlish et al., 2018), which motivates tracking learning stability, batch efficiency, and gradient-noise behavior rather than only final checkpoint loss.

## Benchmark Sanity

Benchmark sanity is a secondary score input. The implemented benchmark weights are:

| Benchmark key | Default share inside benchmark sanity |
| --- | ---: |
| `mmlu` | 20% |
| `gsm8k` | 15% |
| `math` | 15% |
| `humaneval` | 15% |
| `arc_challenge` | 10% |
| `needle` | 10% |
| `ifeval` | 10% |
| `truthfulqa` | 5% |

Benchmark sanity is capped by the formula share. The implemented runtime validation rejects formula configs where `benchmark_sanity` exceeds 15% or becomes the primary architecture or training signal. Defaults keep the cap at 5% for architecture and 15% for training.

The benchmark set maps to public evaluation studies: **GSM8K** (Cobbe et al., 2021), **MATH** (Hendrycks et al., 2021), **ARC** (Clark et al., 2018), **HumanEval** (Chen et al., 2021), **MMLU** (Hendrycks et al., 2020), **IFEval** (Zhou et al., 2023), **TruthfulQA** (Lin, Hilton, and Evans, 2021), and long-context probes inspired by **Lost in the Middle** (Liu et al., 2023) and **RULER** (Hsieh et al., 2024). These benchmarks are useful for downstream sanity checks, but they are capped because public benchmarks can be prompt-sensitive, scale-sensitive, and contaminated.

## Reference Studies

| Area | Study | PRISM implication |
| --- | --- | --- |
| NAS reproducibility | Ying et al., 2019, *NAS-Bench-101* | Use canonical architecture identity and controlled evaluation. |
| NAS reproducibility | Dong and Yang, 2020, *NAS-Bench-201* | Keep architecture and recipe effects separated. |
| NAS baselines | Li and Talwalkar, 2019, *Random Search and Reproducibility for Neural Architecture Search* | Require strong baselines and avoid over-crediting search noise. |
| Scaling laws | Kaplan et al., 2020, *Scaling Laws for Neural Language Models* | Score loss trajectories across parameters, tokens, and compute. |
| Compute optimality | Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* | Avoid ranking by raw loss under under-trained or over-trained regimes. |
| Broken scaling | Caballero et al., 2023, *Broken Neural Scaling Laws* | Treat extrapolation as evidence with uncertainty, not as certainty. |
| Compute reporting | Schwartz et al., 2019, *Green AI* | Include compute efficiency and total evaluation cost. |
| Batch dynamics | McCandlish et al., 2018, *An Empirical Model of Large-Batch Training* | Track gradient-noise and batch-efficiency signals for recipes. |

## Architecture Ownership

PRISM computes canonical architecture identity from `architecture_graph.json` and its SHA-256 hash, plus source-free architecture metadata. Mermaid text is derived for review and display. It is not canonical architecture identity.

The first accepted submission for a new architecture family creates an `architecture_families` record with immutable first-discovery ownership. Later accepted architecture versions may update the canonical submission, best architecture score, and metadata, or be linked as variants, but they never transfer `owner_hotkey` or the first-discovery owner slot.

## Training Variant Ownership

Training and inference code are fingerprinted separately from architecture code. For each architecture family, PRISM tracks training variants and the current best training script version.

Training ownership includes code in:

* `configure_optimizer`
* `inference_logits` or `infer`
* `compute_loss`
* `train_step`
* training helper files declared in `prism.yaml`

A training variant can win the training pool for its target architecture, but it cannot redirect the architecture pool. The training pool is split across architectures using canonical architecture scores, then awarded within each architecture to the current best training owner.

## Dynamic Thresholds

Improvement thresholds prevent tiny metric noise from stealing ownership.

| Setting | Purpose |
| --- | --- |
| `architecture_improvement_min_delta_abs` | Minimum absolute architecture-score improvement. |
| `architecture_improvement_min_delta_rel` | Minimum relative architecture-score improvement. |
| `training_improvement_min_delta_abs` | Minimum absolute training-score improvement. |
| `training_improvement_min_delta_rel` | Minimum relative training-score improvement. |
| `training_improvement_z_score` | Required noise-adjusted improvement margin. |
| `training_metric_default_std` | Default standard deviation when repeats do not report one. |

Architecture improvement thresholds only decide whether canonical version and best-score metadata update; they do not move first-discovery ownership. Training current-best changes use separate training thresholds.

## SQL Override Policy

These percentages are defaults, not hard-coded operator promises. SQL runtime config can override supported policy values for `reward_pools`, `score_weights`, and `benchmark_weights`, subject to validation. Official runtime config fails closed when SQL values are invalid. Explicit local non-scoring paths may fall back to defaults.
