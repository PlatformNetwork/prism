from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .schemas import ComputeBlock

# --- Prequential bits-per-byte primary scoring (architecture.md section 5) -------------------
# The authoritative score is the prequential / online compression metric in bits-per-byte: the
# AREA UNDER the from-scratch online loss curve (integrated over the whole single-pass run),
# normalized by the number of raw UTF-8 BYTES covered (tokenizer-agnostic by construction). It is
# always recomputed from the CHALLENGE-OWNED prism_run_manifest.v2 capture; miner-reported numbers
# are ignored. The legacy raw-loss ``standardized_lm_quality`` term is retired from the score.
NATS_TO_BITS = 1.0 / math.log(2.0)
# A finite/positive sanity band for bits-per-byte; anything outside it is treated as a degenerate
# (non-scorable) run rather than silently ranked.
BPB_SANE_MAX = 64.0

# --- Held-out delta tie-breaker + anti-memorization gap (architecture.md sections 5, 6) ----------
# The held-out delta-over-random-init (``bpb(random-init twin on val) - bpb(trained on val)``) is a
# NEAR-TIE TIE-BREAKER: a larger improvement ranks better, but it must NEVER override the primary
# bpb axis. The tie-break term folded into ``final_score`` is therefore bounded (see
# ``_heldout_tie_break``) so it can only reorder submissions whose bpb is within
# ``HELDOUT_DELTA_BPB_EPSILON`` of each other (VAL-SCORE-001 / VAL-SCORE-019); beyond that band the
# bpb ordering always wins. The train-vs-held-out gap flags memorization: an excessive gap penalizes
# the score so a memorizer ranks below an equivalent non-memorizing run.
HELDOUT_DELTA_TIE_BREAK_WEIGHT = 1e-3
# bpb band within which the held-out delta is allowed to break a near-tie. Submissions whose bpb
# differs by more than this keep strict bpb order regardless of delta.
HELDOUT_DELTA_BPB_EPSILON = 5e-3
MEMORIZATION_GAP_THRESHOLD_BPB = 1.0
MEMORIZATION_PENALTY_FACTOR = 0.5

class ScoreValidationError(ValueError):
    def __init__(self, reasons: list[str] | tuple[str, ...]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class PrequentialBpbScore:
    """Challenge-computed prequential bits-per-byte primary score (architecture.md section 5).

    ``bpb`` is the prequential code-length integrated over the WHOLE single-pass online-loss curve
    divided by the raw UTF-8 BYTES covered (tokenizer-agnostic). ``final_score`` is a documented
    monotone transform where a SMALLER bpb yields a BETTER (larger) final_score, so the existing
    leaderboard ``ORDER BY final_score DESC`` ranks better learners higher. A step-0 / smuggled-
    weights anomaly drives the anti-cheat multiplier to zero so an anomalously-low bpb is flagged
    rather than rewarded.
    """

    bpb: float
    final_score: float
    covered_bytes: int
    sum_neg_log2_likelihood_bits: float
    cumulative_codelength_bits: float
    tokens_consumed: int
    online_loss_samples: int
    step0_loss: float | None
    anti_cheat_multiplier: float
    anomaly: bool
    flags: tuple[str, ...]
    # Held-out delta tie-breaker + anti-memorization gap (architecture.md sections 5, 6). These are
    # ``None`` when no secret val split was scored for the run (held-out simply skipped).
    heldout_delta: float | None = None
    val_bpb_trained: float | None = None
    val_bpb_random_init: float | None = None
    train_heldout_gap: float | None = None
    memorization_flag: bool = False
    memorization_penalty: float = 1.0
    # The CONVERGED (final-checkpoint) train bpb used as the gap's train reference, and which
    # reference produced the gap ("converged" vs the curve-averaged "prequential" fallback).
    train_bpb_converged: float | None = None
    gap_basis: str | None = None

    def metrics_payload(self) -> dict[str, Any]:
        """Flat metrics for the ``scores`` row (challenge-computed; no raw-loss term)."""
        payload: dict[str, Any] = {
            "prequential_bpb": self.bpb,
            "bits_per_byte": self.bpb,
            "final_score": self.final_score,
            "total_bytes_covered": float(self.covered_bytes),
            "covered_bytes": float(self.covered_bytes),
            "sum_neg_log2_likelihood_bits": self.sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": self.cumulative_codelength_bits,
            "tokens_consumed": float(self.tokens_consumed),
            "online_loss_samples": float(self.online_loss_samples),
            "anti_cheat_multiplier": self.anti_cheat_multiplier,
            "step0_anomaly": float(self.anomaly),
            "memorization_flag": float(self.memorization_flag),
            "memorization_penalty": self.memorization_penalty,
        }
        if self.heldout_delta is not None:
            payload["heldout_delta"] = self.heldout_delta
            payload["held_out_delta"] = self.heldout_delta
        if self.val_bpb_trained is not None:
            payload["val_bpb_trained"] = self.val_bpb_trained
        if self.val_bpb_random_init is not None:
            payload["val_bpb_random_init"] = self.val_bpb_random_init
        if self.train_heldout_gap is not None:
            payload["train_heldout_gap"] = self.train_heldout_gap
            payload["train_val_gap"] = self.train_heldout_gap
        if self.train_bpb_converged is not None:
            payload["train_bpb_converged"] = self.train_bpb_converged
        if self.gap_basis is not None:
            payload["gap_basis"] = self.gap_basis
        return payload

    def manifest_score_block(self) -> dict[str, Any]:
        """Challenge-authored ``score`` block merged into prism_run_manifest.v2.json."""
        block: dict[str, Any] = {
            "schema": "prism_score.v2",
            "primary_metric": "prequential_bpb",
            "prequential_bpb": self.bpb,
            "bits_per_byte": self.bpb,
            "final_score": self.final_score,
            "lower_is_better": True,
            "covered_bytes": self.covered_bytes,
            "total_bytes_covered": self.covered_bytes,
            "sum_neg_log2_likelihood_bits": self.sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": self.cumulative_codelength_bits,
            "tokens_consumed": self.tokens_consumed,
            "compute_normalization": "tokens_bytes",
            "wall_clock_term": False,
            "anti_cheat_multiplier": self.anti_cheat_multiplier,
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "tie_breaker": "heldout_delta",
            "memorization_flag": self.memorization_flag,
            "memorization_penalty": self.memorization_penalty,
            "miner_reported_ignored": True,
        }
        if self.heldout_delta is not None:
            block["heldout_delta"] = self.heldout_delta
            block["held_out_delta"] = self.heldout_delta
        if self.val_bpb_trained is not None:
            block["val_bpb_trained"] = self.val_bpb_trained
        if self.val_bpb_random_init is not None:
            block["val_bpb_random_init"] = self.val_bpb_random_init
        if self.train_heldout_gap is not None:
            block["train_heldout_gap"] = self.train_heldout_gap
        if self.train_bpb_converged is not None:
            block["train_bpb_converged"] = self.train_bpb_converged
        if self.gap_basis is not None:
            block["gap_basis"] = self.gap_basis
        return block


def bpb_to_final_score(bpb: float) -> float:
    """Monotone-decreasing transform of bits-per-byte: lower bpb -> higher (better) final_score."""
    return 1.0 / (1.0 + max(0.0, float(bpb)))


def build_compute_block(
    *,
    gpu_count: int,
    world_size: int,
    nproc_per_node: int,
    device: str,
    max_gpu_count: int | None = None,
    model_params: int | None = None,
) -> dict[str, Any]:
    """Build the typed, observability-only ``compute`` block for the v2 manifest.

    ``gpu_count`` records the GPUs actually LEASED for the scored run (``== 1`` for the scored
    ``nproc=1`` path). The block is RECORDED for observability and VAL-GPU-005; it is NEVER read by
    ``score_prequential_bpb`` (``final_score`` derives only from compute-normalized learning
    metrics, so there is no GPU-count reward and no scaling bonus). ``model_params`` records the
    realized parameter count of the model the runner actually trained/scored so the cap can be
    shown to bind the scored model (VAL-CHEAT-022). Validated through the typed
    :class:`~prism_challenge.evaluator.schemas.ComputeBlock` so the launch shape is well-formed.
    """
    block = ComputeBlock(
        gpu_count=gpu_count,
        world_size=world_size,
        nproc_per_node=nproc_per_node,
        device=device,
        max_gpu_count=max_gpu_count,
        model_params=model_params,
    )
    return block.model_dump(by_alias=True, exclude_none=True)


# --- Deterministic leaderboard ordering + final tie-break (architecture.md section 5) ------------
# ``final_score`` already folds the primary prequential bpb and the held-out-delta tie-breaker into
# one monotone number (lower bpb / larger delta => larger final_score, so ORDER BY final_score DESC
# ranks better learners first). When two submissions are near-equal on BOTH axes their final_score
# is (near-)equal; the FINAL deterministic tie-break is EARLIEST-COMMIT-WINS (then submission id) so
# the leaderboard order is a TOTAL, reproducible order. The tie epsilon stays far below
# ``HELDOUT_DELTA_TIE_BREAK_WEIGHT`` so a genuine held-out-delta difference still orders ahead of
# the commit-time tie-break (the delta tie-break is never collapsed).
LEADERBOARD_TIE_EPSILON = 1e-9


@dataclass(frozen=True)
class LeaderboardRow:
    """A scored, completed submission competing for a leaderboard rank."""

    submission_id: str
    hotkey: str
    final_score: float
    accepted_at: str


def leaderboard_rank_key(row: LeaderboardRow) -> tuple[int, str, str]:
    """Total, deterministic leaderboard sort key (ascending => better rank first).

    ``final_score`` is quantized onto an epsilon grid so near-equal scores share a bucket; the
    higher bucket ranks first (``-bucket``), and a same-bucket tie is resolved by earliest commit
    (``accepted_at`` ascending) then ``submission_id`` ascending.
    """
    score = float(row.final_score)
    bucket = math.floor(score / LEADERBOARD_TIE_EPSILON + 0.5) if math.isfinite(score) else 0
    return (-bucket, row.accepted_at, row.submission_id)


def rank_leaderboard(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    """Order leaderboard rows by bpb/learning with the deterministic earliest-commit tie-break."""
    return sorted(rows, key=leaderboard_rank_key)


def score_prequential_bpb(
    manifest: Mapping[str, Any], *, sane_max: float = BPB_SANE_MAX
) -> PrequentialBpbScore:
    """Compute the prequential bits-per-byte score from the challenge-owned v2 manifest.

    ``bpb = (sum over consumed tokens of -log2 p(token)) / total_bytes_covered`` where the
    numerator is the token-weighted online (predict-then-train) negative log-likelihood the
    challenge captured itself. Raises ``ScoreValidationError`` for a degenerate (zero-coverage,
    non-finite, or out-of-band) run so it never collapses into a fabricated/0-that-ranks score.
    """
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ScoreValidationError(["v2 manifest is missing a metrics block"])
    covered_bytes = _manifest_covered_bytes(manifest, metrics)
    if covered_bytes <= 0:
        raise ScoreValidationError(["prequential scoring requires covered_bytes > 0"])
    sum_nll_nats = float(metrics.get("sum_neg_log_likelihood_nats", 0.0))
    online_loss = metrics.get("online_loss")
    online_samples = len(online_loss) if isinstance(online_loss, list) else 0
    if online_samples == 0:
        raise ScoreValidationError(["prequential scoring requires a captured online-loss stream"])
    cumulative_codelength_bits = sum_nll_nats * NATS_TO_BITS
    bpb = cumulative_codelength_bits / covered_bytes
    flags: list[str] = []
    if not math.isfinite(bpb):
        raise ScoreValidationError(["prequential bpb is not finite"])
    if bpb <= 0.0:
        raise ScoreValidationError(["prequential bpb must be positive"])
    if bpb > sane_max:
        flags.append("bpb_out_of_band")
    anti_cheat = manifest.get("anti_cheat")
    anti_cheat = anti_cheat if isinstance(anti_cheat, Mapping) else {}
    step0_anomaly = bool(anti_cheat.get("step0_anomaly", False))
    if step0_anomaly:
        flags.append("step0_anomaly")
    if bool(anti_cheat.get("nan_inf_detected", False)):
        flags.append("nan_inf_detected")
    anti_cheat_multiplier = 0.0 if step0_anomaly else 1.0
    heldout = _read_heldout(manifest, metrics, anti_cheat, train_bpb=bpb)
    if heldout.memorization_flag:
        flags.append("memorization_gap")
    # The held-out delta refines ranking only as a NEAR-TIE tie-breaker (bounded additive term,
    # monotone in the delta) so a strictly lower bpb is NEVER ranked worse on the primary axis even
    # at high bpb where ``base`` compresses; an excessive train-vs-held-out gap multiplies in a
    # memorization penalty so a memorizer ranks below an equivalent non-memorizing run.
    base = bpb_to_final_score(bpb)
    tie_break = _heldout_tie_break(bpb, heldout.delta, heldout.penalty)
    final_score_value = (base * heldout.penalty + tie_break) * anti_cheat_multiplier
    step0_loss = metrics.get("step0_loss")
    return PrequentialBpbScore(
        bpb=bpb,
        final_score=final_score_value,
        covered_bytes=covered_bytes,
        sum_neg_log2_likelihood_bits=cumulative_codelength_bits,
        cumulative_codelength_bits=cumulative_codelength_bits,
        tokens_consumed=int(metrics.get("predicted_tokens", metrics.get("tokens_seen", 0)) or 0),
        online_loss_samples=online_samples,
        step0_loss=float(step0_loss) if isinstance(step0_loss, int | float) else None,
        anti_cheat_multiplier=anti_cheat_multiplier,
        anomaly=step0_anomaly,
        flags=tuple(flags),
        heldout_delta=heldout.delta,
        val_bpb_trained=heldout.val_bpb_trained,
        val_bpb_random_init=heldout.val_bpb_random_init,
        train_heldout_gap=heldout.gap,
        memorization_flag=heldout.memorization_flag,
        memorization_penalty=heldout.penalty,
        train_bpb_converged=heldout.train_bpb_converged,
        gap_basis=heldout.gap_basis,
    )


@dataclass(frozen=True)
class _HeldoutView:
    delta: float | None
    val_bpb_trained: float | None
    val_bpb_random_init: float | None
    gap: float | None
    memorization_flag: bool
    penalty: float
    train_bpb_converged: float | None = None
    gap_basis: str | None = None


def _read_heldout(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    anti_cheat: Mapping[str, Any],
    *,
    train_bpb: float,
) -> _HeldoutView:
    """Read the host-computed held-out delta + anti-memorization gap from the v2 manifest.

    The held-out delta + gap are populated host-side (``evaluator/heldout.py``) into the metrics /
    score blocks. When absent (no secret val split scored) the run is graded on prequential bpb
    alone with no tie-break and no penalty.
    """
    score_block = manifest.get("score")
    score_block = score_block if isinstance(score_block, Mapping) else {}
    delta = _coerce_float(_first_present(metrics, score_block, ("heldout_delta", "held_out_delta")))
    val_trained = _coerce_float(_first_present(metrics, score_block, ("val_bpb_trained",)))
    val_random = _coerce_float(_first_present(metrics, score_block, ("val_bpb_random_init",)))
    # The CONVERGED (final-checkpoint) train reference is measured byte-level on the SAME trained
    # model as the held-out val bpb, so the gap is like-for-like by construction. When the manifest
    # carries it, the gap bypasses the tokenizer-basis gating below (which only guards the inflated,
    # potentially cross-basis prequential fallback) and reflects the converged model -- closing the
    # false-negative hole where the curve-averaged prequential reference shrinks the gap.
    train_converged = _coerce_float(_first_present(metrics, score_block, ("train_bpb_converged",)))
    gap_basis = _coerce_str(_first_present(metrics, score_block, ("gap_basis",)))
    converged = gap_basis == "converged" or train_converged is not None
    # The anti-memorization GAP compares train bpb against val bpb; for the prequential fallback it
    # is only meaningful when both were measured on the SAME tokenizer basis. The host measures val
    # bpb on raw UTF-8 bytes, so a native-tokenizer train basis would inflate the "gap" and
    # false-flag a benign learner (VAL-SCORE-009 / VAL-SCORE-004). Absent basis info => comparable
    # (backward compatible with manifests that recorded no basis). A converged reference is always
    # comparable (byte-level on both sides).
    train_basis = _coerce_str(_first_present(metrics, score_block, ("train_bpb_basis",)))
    val_basis = _coerce_str(_first_present(metrics, score_block, ("val_bpb_basis",)))
    bases_comparable = converged or (
        train_basis is None or val_basis is None or train_basis == val_basis
    )
    gap = _coerce_float(
        _first_present(metrics, score_block, ("train_heldout_gap", "train_val_gap"))
    )
    if not bases_comparable:
        gap = None
    elif gap is None and val_trained is not None:
        reference = train_converged if train_converged is not None else train_bpb
        if reference is not None and math.isfinite(reference):
            gap = val_trained - reference
    explicit_flag = bool(
        metrics.get("memorization_flag")
        or score_block.get("memorization_flag")
        or anti_cheat.get("memorization_flag")
    )
    memorization_flag = bases_comparable and (
        explicit_flag or (gap is not None and gap > MEMORIZATION_GAP_THRESHOLD_BPB)
    )
    penalty = MEMORIZATION_PENALTY_FACTOR if memorization_flag else 1.0
    return _HeldoutView(
        delta=delta,
        val_bpb_trained=val_trained,
        val_bpb_random_init=val_random,
        gap=gap,
        memorization_flag=memorization_flag,
        penalty=penalty,
        train_bpb_converged=train_converged,
        gap_basis=("converged" if converged else gap_basis),
    )


def _first_present(
    primary: Mapping[str, Any], secondary: Mapping[str, Any], keys: tuple[str, ...]
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _heldout_tie_break(bpb: float, delta: float | None, penalty: float) -> float:
    """Held-out-delta tie-break bounded so it can ONLY reorder a NEAR-TIE on bpb.

    The term is capped at half the (penalty-scaled) final_score resolution across a
    ``HELDOUT_DELTA_BPB_EPSILON`` bpb band (and never above ``HELDOUT_DELTA_TIE_BREAK_WEIGHT``). The
    resolution shrinks monotonically as bpb grows (``base`` is convex), so two submissions whose bpb
    differs by more than the epsilon keep strict bpb order regardless of delta (VAL-SCORE-001 /
    VAL-SCORE-019); within the band the larger delta wins (VAL-SCORE-008).
    """
    if delta is None:
        return 0.0
    resolution = bpb_to_final_score(bpb) - bpb_to_final_score(bpb + HELDOUT_DELTA_BPB_EPSILON)
    weight = max(0.0, penalty)
    cap = min(HELDOUT_DELTA_TIE_BREAK_WEIGHT * weight, 0.5 * weight * max(0.0, resolution))
    return cap * math.tanh(delta)


def _manifest_covered_bytes(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> int:
    for source in (metrics, manifest.get("data")):
        if isinstance(source, Mapping):
            value = source.get("covered_bytes")
            if isinstance(value, int | float) and not isinstance(value, bool):
                return int(value)
    return 0
