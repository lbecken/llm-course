"""
Small algorithmic tasks with exact answers.

`lookup_batch`   — "attention as a soft dictionary": find the value stored under a key.
`AdditionTask`   — character-level integer addition, used for fine-tuning, RL with
                   verifiable rewards, tool use and reasoning experiments (Parts IV-V).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# ----------------------------------------------------------------------------
# Lookup task (Module 05)
#
# A sequence is a shuffled set of (key, value) PAIR tokens followed by one QUERY token:
#
#     pair(k3,v5) pair(k0,v2) pair(k6,v1) ... query(k0)   ->  answer: v2
#
# Token ids:   pair(k, v) = k * num_values + v           (num_keys * num_values ids)
#              query(k)   = num_keys * num_values + k    (num_keys ids)
# The label is the value v stored under the queried key, a class in [0, num_values).
#
# One attention head solves this exactly: the query token's QUERY vector must match the
# KEY vector of the pair with the same key, and the pair's VALUE vector must carry v.
# No position information is needed, because the answer does not depend on order.


@dataclass
class LookupSpec:
    num_keys: int = 8
    num_values: int = 8
    num_pairs: int = 5  # how many distinct keys appear in each sequence

    @property
    def vocab_size(self) -> int:
        return self.num_keys * self.num_values + self.num_keys

    @property
    def seq_len(self) -> int:
        return self.num_pairs + 1


def lookup_batch(spec: LookupSpec, batch_size: int, generator: torch.Generator | None = None):
    """Returns (tokens, labels, target_pos).

    tokens:     (B, num_pairs + 1) int64 — pairs then the query token
    labels:     (B,) int64 — the value to retrieve
    target_pos: (B,) int64 — index of the pair that holds the answer (for inspecting attention)
    """
    g = generator
    B, P = batch_size, spec.num_pairs
    # choose P distinct keys per row via random permutations
    keys = torch.stack([torch.randperm(spec.num_keys, generator=g)[:P] for _ in range(B)])  # (B, P)
    values = torch.randint(0, spec.num_values, (B, P), generator=g)  # (B, P)
    target_pos = torch.randint(0, P, (B,), generator=g)  # (B,)
    rows = torch.arange(B)
    query_key = keys[rows, target_pos]
    labels = values[rows, target_pos]
    pair_tokens = keys * spec.num_values + values
    query_tokens = spec.num_keys * spec.num_values + query_key
    tokens = torch.cat([pair_tokens, query_tokens[:, None]], dim=1)
    return tokens, labels, target_pos


# ----------------------------------------------------------------------------
# Addition task (Parts IV-V)


@dataclass
class AdditionProblem:
    a: int
    b: int

    @property
    def question(self) -> str:
        return f"{self.a}+{self.b}="

    @property
    def answer(self) -> str:
        return str(self.a + self.b)


class AdditionTask:
    """Deterministic stream of addition problems with disjoint train / test splits.

    The split is by hash of (a, b), so a test problem never appears in training: any
    test accuracy above chance is generalisation, not memorisation."""

    def __init__(self, max_digits: int = 3, test_fraction: float = 0.1, seed: int = 0):
        self.max_value = 10**max_digits - 1
        self.test_fraction = test_fraction
        self.seed = seed

    def _is_test(self, a: int, b: int) -> bool:
        h = (a * 1_000_003 + b * 998_244_353 + self.seed * 7919) % 1000
        return h < int(self.test_fraction * 1000)

    def sample(self, rng: random.Random, split: str = "train") -> AdditionProblem:
        assert split in ("train", "test")
        while True:
            # sample the number of digits first so that short numbers are not rare
            a = rng.randint(0, 10 ** rng.randint(1, len(str(self.max_value))) - 1)
            b = rng.randint(0, 10 ** rng.randint(1, len(str(self.max_value))) - 1)
            if self._is_test(a, b) == (split == "test"):
                return AdditionProblem(a, b)

    def problems(self, n: int, split: str, seed: int = 0) -> list[AdditionProblem]:
        rng = random.Random(f"{split}-{seed}-{self.seed}")
        return [self.sample(rng, split) for _ in range(n)]
