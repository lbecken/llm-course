"""
Count-based language models (Module 02).

A language model assigns a probability to the next token given the previous ones.
The simplest possible versions just count:

    unigram:  P(x_t)                  = count(x_t) / N
    bigram:   P(x_t | x_{t-1})        = count(x_{t-1}, x_t) / count(x_{t-1})
    n-gram:   P(x_t | x_{t-n+1..t-1}) = count(context, x_t) / count(context)

They are the baseline every neural model must beat, and they make the limits of a fixed
short context concrete: a bigram model literally cannot see two tokens back.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import torch


class CharTokenizer:
    """Character-level tokenizer: each distinct character is one token."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class NGramLM:
    """n-gram model over integer tokens with add-alpha smoothing and back-off.

    `order` is the number of tokens of context: 0 = unigram, 1 = bigram, 2 = trigram ...
    Smoothing adds `alpha` pseudo-counts to every possible next token so that unseen
    events get non-zero probability (otherwise one unseen pair makes the loss infinite).
    Back-off: if a context was never seen in training, drop its oldest token and try
    again, down to the unigram distribution. Both are classic answers to the central
    weakness of counting: most long contexts never occur in any finite corpus."""

    def __init__(self, order: int, vocab_size: int, alpha: float = 0.01):
        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.counts: dict[tuple, Counter] = defaultdict(Counter)

    def fit(self, tokens: list[int]) -> "NGramLM":
        for t in range(len(tokens)):
            for n in range(0, min(self.order, t) + 1):  # every context length up to `order`
                self.counts[tuple(tokens[t - n : t])][tokens[t]] += 1
        return self

    def num_contexts(self) -> int:
        """Number of distinct full-length contexts observed in training."""
        return sum(1 for c in self.counts if len(c) == self.order)

    def probs(self, context: list[int]) -> torch.Tensor:
        """Distribution over the next token, shape (vocab_size,)."""
        context = tuple(context[-self.order :]) if self.order > 0 else ()
        while context and context not in self.counts:
            context = context[1:]  # back off to a shorter context
        counter = self.counts.get(context, {})
        p = torch.full((self.vocab_size,), self.alpha, dtype=torch.float64)
        for tok, c in counter.items():
            p[tok] += c
        return p / p.sum()

    def cross_entropy(self, tokens: list[int]) -> float:
        """Average negative log-probability (nats per token) of `tokens` under the model."""
        total = 0.0
        for t in range(len(tokens)):
            p = self.probs(tokens[max(0, t - self.order) : t])
            total += -math.log(p[tokens[t]].item())
        return total / len(tokens)

    def generate(self, prompt: list[int], num_tokens: int, seed: int = 0) -> list[int]:
        g = torch.Generator().manual_seed(seed)
        out = list(prompt)
        for _ in range(num_tokens):
            p = self.probs(out)
            out.append(torch.multinomial(p, 1, generator=g).item())
        return out


def perplexity(cross_entropy_nats: float) -> float:
    """exp(cross-entropy): the size of a uniform distribution that is equally uncertain."""
    return math.exp(cross_entropy_nats)


class WordTokenizer:
    """Whitespace word-level tokenizer for the Pocket language: one token per word, plus
    the newline that ends each sentence. Used in Module 04, where whole-word embeddings
    make the geometry of the embedding space easy to inspect."""

    def __init__(self, text: str):
        self.words = sorted(set(self._split(text)))
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.itos = {i: w for w, i in self.stoi.items()}

    @staticmethod
    def _split(text: str) -> list[str]:
        out = []
        for line in text.split("\n"):
            if line:
                out.extend(line.split(" "))
                out.append("\n")
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in self._split(text)]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids).replace(" \n ", "\n").replace(" \n", "\n")
