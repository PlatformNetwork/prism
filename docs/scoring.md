# Scoring and Rewards

PRISM scores a single thing: a model's ability to learn from scratch, measured as online compression.
The primary metric is a **prequential bits-per-byte (bpb)** score that the challenge computes itself
from a forced-init re-execution. A held-out delta-over-random-init breaks near-ties, and an
anti-memorization gap penalizes overfitting. Lower bits-per-byte is better.

## Primary Metric: Prequential Bits-Per-Byte

During the forced-init re-execution, the challenge feeds the model fresh, single-pass batches from the
locked train split and records the model's loss on each new batch **before** the optimizer updates on
it. Because the data is single-pass, this online (predict-then-train) loss is the prequential
code-length by construction.

The challenge integrates that code-length over the whole run and normalizes it by the raw UTF-8 bytes
of text covered:

```text
bpb = (sum over consumed tokens of -log2 p(token)) / total_bytes_covered
```

Because the denominator is bytes, the metric is **tokenizer-agnostic**: a miner can bring any
tokenizer and the score still compares like for like. Because it integrates the whole loss curve, a
single good checkpoint cannot game it. Because each token is scored before being trained on, there is
no held-out leakage by construction. And because the validator forces random init, smuggled
pretrained weights are inert.

`final_score` is a documented monotone-decreasing transform of bpb, so a **lower** bpb yields a
**better** (higher) `final_score`, and the leaderboard's `ORDER BY final_score DESC` ranks better
learners first:

```text
final_score = 1 / (1 + bpb)        # before tie-break, penalty, and anti-cheat multiplier
```

## Compute Normalization, Not Wall-Clock

The score is **compute-normalized**: it is reported and normalized by tokens consumed (and,
optionally, estimated FLOPs), never by wall-clock time. A faster GPU or more GPUs cannot buy a better
score; wall-clock is only a safety cap on the run. This keeps scores fair across the 1-to-8 GPU range
even though the scored run uses one physical GPU.

## Tie-Breaker: Held-Out Delta Over Random Init

When two submissions are near-equal on bpb, the challenge breaks the tie with the held-out delta on
the secret `val` split:

```text
heldout_delta = bpb(random-init twin on val) - bpb(trained model on val)
```

A larger improvement over the random-init twin is better. The held-out delta is folded into
`final_score` as a **bounded** tie-break term: it can only reorder submissions whose bpb is within a
small epsilon of each other, so a strictly lower bpb is never ranked worse on the primary axis. When
no secret val split is scored for a run, the run is graded on bpb alone with no tie-break.

## Anti-Memorization Gap

The challenge also measures the train-vs-held-out gap (the converged train bpb against the held-out
val bpb on the same byte basis). An excessive gap flags memorization and multiplies a penalty into
`final_score`, so a memorizer ranks below an equivalent non-memorizing learner. The gap comparison is
basis-consistent so a benign learner is not falsely flagged, while the final bpb denominator stays
UTF-8 bytes.

## Anomaly Zeroing

A step-0 / smuggled-weights anomaly (an impossibly low initial loss under forced random init) drives
the anti-cheat multiplier to zero, so an anomalously good bpb is flagged and zeroed rather than
rewarded. A degenerate run (zero coverage, non-finite, or out-of-band bpb) is failed rather than
scored, so it never collapses into a fabricated score that ranks.

## Leaderboard And Tie-Break Ordering

The leaderboard ranks by `final_score` (so by bpb and the folded-in held-out delta). When two
submissions are still equal, the final deterministic tie-break is **earliest-commit-wins**, then
submission id, producing a total, reproducible order. Each hotkey appears at most once: the best
submission per hotkey survives, so a worse same-hotkey submission never supersedes a better one.

## Weights

`get_weights` converts completed scores into normalized BASE weights: one weight per hotkey, taken
from that hotkey's best `final_score`, normalized to sum to 1.0. Weights are always **dry-run** and
are never written on-chain.

## The Challenge Is The Source Of Truth

Every number above is recomputed by the challenge from the challenge-authored
`prism_run_manifest.v2.json`. Miner-reported metrics and miner-written manifests are ignored. The
legacy raw-loss term and the v1-NAS architecture/training ownership pools are retired from the score.

## Reference Studies

| Area | Study | PRISM implication |
| --- | --- | --- |
| Prequential / online coding | Dawid, 1984, *Present Position and Potential Developments: Some Personal Views: Statistical Theory: The Prequential Approach* | Score the integrated online (predict-then-train) loss, not a final checkpoint. |
| Minimum description length | Rissanen, 1978, *Modeling by Shortest Data Description* | Treat compression (code-length) as the learning signal. |
| Scaling laws | Kaplan et al., 2020, *Scaling Laws for Neural Language Models* | Compare loss trajectories under matched compute, not raw final loss. |
| Compute-optimal scaling | Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models* | Normalize by tokens/compute so under- or over-trained regimes do not skew ranking. |
| Dataset provenance | Penedo et al., 2024, *The FineWeb Datasets* | Freeze the data revision and shards for reproducible official runs. |
