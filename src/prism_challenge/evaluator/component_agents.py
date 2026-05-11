from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .component_signatures import ComponentSemanticSignature, semantic_similarity

ARCH_ACTIONS = {"new", "existing", "transfer", "hold", "reject"}
TRAINING_ACTIONS = {"new", "existing", "transfer", "hold", "reject", "none"}


@dataclass(frozen=True)
class ComponentOwnershipDecision:
    architecture_action: str
    architecture_confidence: float
    training_action: str
    training_confidence: float
    matched_architecture_id: str | None = None
    matched_training_variant_id: str | None = None
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def held(self) -> bool:
        return self.architecture_action == "hold" or self.training_action == "hold"

    @property
    def rejected(self) -> bool:
        return self.architecture_action == "reject" or self.training_action == "reject"


class SemanticOwnershipAgent:
    def __init__(
        self,
        *,
        min_confidence: float,
        same_threshold: float,
        hold_threshold: float,
    ) -> None:
        self.min_confidence = min_confidence
        self.same_threshold = same_threshold
        self.hold_threshold = hold_threshold

    def decide(
        self,
        *,
        signature: ComponentSemanticSignature,
        architecture_candidates: list[dict[str, Any]],
        training_candidates: list[dict[str, Any]],
        requested_architecture_id: str | None,
    ) -> ComponentOwnershipDecision:
        architecture = self._architecture_decision(
            signature=signature,
            candidates=architecture_candidates,
            requested_architecture_id=requested_architecture_id,
        )
        training = self._training_decision(
            signature=signature,
            candidates=training_candidates,
            has_architecture=architecture["action"] != "hold",
        )
        reason = f"architecture={architecture['reason']}; training={training['reason']}"
        return ComponentOwnershipDecision(
            architecture_action=str(architecture["action"]),
            architecture_confidence=float(architecture["confidence"]),
            training_action=str(training["action"]),
            training_confidence=float(training["confidence"]),
            matched_architecture_id=architecture.get("id"),
            matched_training_variant_id=training.get("id"),
            reason=reason,
            raw={"architecture": architecture, "training": training},
        )

    def _architecture_decision(
        self,
        *,
        signature: ComponentSemanticSignature,
        candidates: list[dict[str, Any]],
        requested_architecture_id: str | None,
    ) -> dict[str, Any]:
        if not candidates:
            return {
                "action": "new",
                "confidence": 1.0,
                "id": None,
                "reason": "no comparable architecture family exists",
            }
        exact = next(
            (
                item
                for item in candidates
                if str(item.get("family_hash")) == signature.family_hash
                or str(item.get("arch_fingerprint")) == signature.arch_fingerprint
            ),
            None,
        )
        if exact is not None:
            return {
                "action": "existing",
                "confidence": 1.0,
                "id": str(exact["id"]),
                "reason": "exact architecture fingerprint match",
            }
        if requested_architecture_id:
            requested = next(
                (item for item in candidates if str(item.get("id")) == requested_architecture_id),
                None,
            )
            if requested is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "id": requested_architecture_id,
                    "reason": "requested architecture was not found",
                }
            score = _candidate_similarity(signature.architecture_graph, requested)
            if score >= self.same_threshold:
                return {
                    "action": "existing",
                    "confidence": score,
                    "id": requested_architecture_id,
                    "reason": "requested architecture semantically matches",
                }
            return {
                "action": "hold",
                "confidence": score,
                "id": requested_architecture_id,
                "reason": "requested architecture does not semantically match",
            }
        best = _best_candidate(signature.architecture_graph, candidates)
        if best is None:
            return {
                "action": "new",
                "confidence": 1.0,
                "id": None,
                "reason": "no semantic architecture match",
            }
        best_row, score = best
        if score >= self.same_threshold:
            return {
                "action": "existing",
                "confidence": score,
                "id": str(best_row["id"]),
                "reason": "semantic architecture match; likely derivative",
            }
        if score >= self.hold_threshold:
            return {
                "action": "hold",
                "confidence": score,
                "id": str(best_row["id"]),
                "reason": "ambiguous architecture similarity",
            }
        return {
            "action": "new",
            "confidence": 1.0 - score,
            "id": None,
            "reason": "architecture is semantically distinct",
        }

    def _training_decision(
        self,
        *,
        signature: ComponentSemanticSignature,
        candidates: list[dict[str, Any]],
        has_architecture: bool,
    ) -> dict[str, Any]:
        if signature.project_kind == "architecture_only":
            return {"action": "none", "confidence": 1.0, "id": None, "reason": "architecture only"}
        if not has_architecture:
            return {
                "action": "hold",
                "confidence": 0.0,
                "id": None,
                "reason": "architecture decision must resolve first",
            }
        if not candidates:
            return {
                "action": "new",
                "confidence": 1.0,
                "id": None,
                "reason": "no comparable training variant exists",
            }
        exact = next(
            (
                item
                for item in candidates
                if str(item.get("training_hash")) == signature.training_hash
            ),
            None,
        )
        if exact is not None:
            return {
                "action": "existing",
                "confidence": 1.0,
                "id": str(exact["id"]),
                "reason": "exact training fingerprint match",
            }
        best = _best_candidate(signature.training_graph, candidates)
        if best is None:
            return {
                "action": "new",
                "confidence": 1.0,
                "id": None,
                "reason": "no semantic training match",
            }
        best_row, score = best
        if score >= self.same_threshold:
            return {
                "action": "existing",
                "confidence": score,
                "id": str(best_row["id"]),
                "reason": "semantic training match; likely useless derivative",
            }
        if score >= self.hold_threshold:
            return {
                "action": "hold",
                "confidence": score,
                "id": str(best_row["id"]),
                "reason": "ambiguous training similarity",
            }
        return {
            "action": "new",
            "confidence": 1.0 - score,
            "id": None,
            "reason": "training code is semantically distinct",
        }


def validate_decision(decision: ComponentOwnershipDecision) -> ComponentOwnershipDecision:
    if decision.architecture_action not in ARCH_ACTIONS:
        raise ValueError(f"invalid architecture action: {decision.architecture_action}")
    if decision.training_action not in TRAINING_ACTIONS:
        raise ValueError(f"invalid training action: {decision.training_action}")
    return decision


def _best_candidate(
    graph: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], float] | None:
    scored = [(item, _candidate_similarity(graph, item)) for item in candidates]
    scored = [item for item in scored if item[1] > 0]
    return max(scored, key=lambda item: item[1]) if scored else None


def _candidate_similarity(graph: dict[str, Any], candidate: dict[str, Any]) -> float:
    candidate_graph = candidate.get("architecture_graph") or candidate.get("training_graph") or {}
    if not isinstance(candidate_graph, dict):
        return 0.0
    return semantic_similarity(graph, candidate_graph)
