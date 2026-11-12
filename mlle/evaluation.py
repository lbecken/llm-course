"""
Evaluation arithmetic (Module 10): tokenizer-independent loss, and honest uncertainty.

    bits_per_byte         cross-entropy normalised by UTF-8 bytes (nanochat's main loss metric)
    accuracy_stderr       binomial standard error of an accuracy
    bootstrap_ci          confidence interval of a mean by resampling
    paired_bootstrap      is system A better than system B on the SAME items?
    mean_std              summary of a list of seeds
"""

from __future__ import annotations

import math
import random


def bits_per_byte(total_nats: float, total_bytes: int) -> float:
    """Sum of token losses (nats) divided by the number of UTF-8 bytes they encode, in bits.
    Same formula as nanochat/loss_eval.py: bpb = total_nats / (ln 2 * total_bytes)."""
    return total_nats / (math.log(2) * total_bytes)


def accuracy_stderr(accuracy: float, n: int) -> float:
    """Standard error of an accuracy measured on n independent items."""
    return math.sqrt(accuracy * (1 - accuracy) / n)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (n - 1 in the denominator)."""
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(values: list[float], num_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper): percentile bootstrap confidence interval for the mean of `values`."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(num_resamples))
    lo = means[int(alpha / 2 * num_resamples)]
    hi = means[int((1 - alpha / 2) * num_resamples) - 1]
    return sum(values) / n, lo, hi


def paired_bootstrap(a: list[float], b: list[float], num_resamples: int = 2000, seed: int = 0) -> dict:
    """Compare two systems scored on the SAME items (a[i], b[i] are per-item scores).

    Resamples items with replacement and reports how often A's mean exceeds B's. Pairing
    removes the variance due to item difficulty, which usually dominates: two systems can
    differ reliably even when their individual confidence intervals overlap."""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    wins = 0
    resampled = []
    for _ in range(num_resamples):
        d = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        resampled.append(d)
        wins += d > 0
    resampled.sort()
    return {
        "mean_diff": sum(diffs) / n,
        "ci_low": resampled[int(0.025 * num_resamples)],
        "ci_high": resampled[int(0.975 * num_resamples) - 1],
        "p_a_better": wins / num_resamples,
    }
