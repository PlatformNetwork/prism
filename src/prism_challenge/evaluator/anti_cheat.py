from __future__ import annotations

import ast
from dataclasses import dataclass

from .sandbox import inspect_code


@dataclass(frozen=True)
class CheatFinding:
    kind: str
    severity: float
    details: str


@dataclass(frozen=True)
class AntiCheatResult:
    multiplier: float
    diversity_bonus: float
    findings: list[CheatFinding]


def ast_similarity(left: str, right: str) -> float:
    left_nodes = [type(node).__name__ for node in ast.walk(ast.parse(left))]
    right_nodes = [type(node).__name__ for node in ast.walk(ast.parse(right))]
    if not left_nodes or not right_nodes:
        return 0.0
    left_set = set(left_nodes)
    right_set = set(right_nodes)
    return len(left_set & right_set) / len(left_set | right_set)


def jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / len(left | right))


def evaluate_anti_cheat(code: str, previous_codes: list[str]) -> AntiCheatResult:
    report = inspect_code(code)
    findings: list[CheatFinding] = []
    max_similarity = 0.0
    distances: list[float] = []
    for previous in previous_codes:
        similarity = ast_similarity(code, previous)
        max_similarity = max(max_similarity, similarity)
        previous_report = inspect_code(previous)
        distances.append(jaccard_distance(report.ast_fingerprint, previous_report.ast_fingerprint))
    if max_similarity >= 0.96:
        findings.append(CheatFinding("ast_similarity", 0.8, "near-duplicate submission"))
    avg_distance = sum(distances) / len(distances) if distances else 1.0
    diversity_bonus = min(0.05, max(0.0, avg_distance * 0.05))
    multiplier = max(0.0, 1.0 - sum(f.severity for f in findings))
    return AntiCheatResult(multiplier, diversity_bonus, findings)
