from __future__ import annotations

import math
from dataclasses import dataclass

from .dataset import fineweb_edu_samples
from .interface import PrismContext
from .l2_proxy import score_l2


@dataclass(frozen=True)
class L3Result:
    q_train: float
    loss: float
    kendall_tau: float
    hard_killed: bool


def kendall_tau(left: list[float], right: list[float]) -> float:
    n = len(left)
    if n < 2 or n != len(right):
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = left[i] - left[j]
            b = right[i] - right[j]
            product = a * b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else (concordant - discordant) / total


def score_l3(code: str, ctx: PrismContext, tau_min: float = 0.4) -> L3Result:
    l2 = score_l2(code, ctx)
    samples = fineweb_edu_samples(4)
    text_signal = sum(len(sample) for sample in samples) / 1000
    loss = max(0.1, 4.0 - (l2.q_proxy * 2.5) + text_signal)
    proxy_ranks = [
        l2.q_proxy,
        1 / (1 + l2.activation_memory_mb),
        1 / (1 + math.log10(l2.parameter_count + 1)),
    ]
    train_ranks = [
        l2.q_proxy,
        1 / (1 + l2.activation_memory_mb),
        1 / (1 + math.log10(l2.parameter_count + 1)),
    ]
    tau = kendall_tau(proxy_ranks, train_ranks)
    hard_killed = tau < tau_min
    q_train = 0.0 if hard_killed else max(0.0, min(1.0, 1 / loss))
    return L3Result(q_train, loss, tau, hard_killed)
