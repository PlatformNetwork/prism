from __future__ import annotations

FALLBACK_FINEWEB_EDU = [
    "Students learn best when examples connect abstract rules to concrete practice.",
    "A neural network maps tokens into vectors and composes them through layers.",
    "Scientific evaluation requires held out data and reproducible measurements.",
    "Efficient architectures reduce compute while preserving generalization quality.",
]


def fineweb_edu_samples(sample_count: int) -> list[str]:
    samples: list[str] = []
    while len(samples) < sample_count:
        samples.extend(FALLBACK_FINEWEB_EDU)
    return samples[:sample_count]
