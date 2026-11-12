"""
Decoding strategies (Module 10): turning a vector of logits into one chosen token.

    greedy              arg-max
    temperature         divide logits by tau before the softmax
    top-k               keep only the k most likely tokens
    top-p (nucleus)     keep the smallest set of tokens whose probability mass reaches p
    repetition penalty  make already-generated tokens less likely

nanochat's engine implements temperature and top-k (nanochat/engine.py, sample_next_token).
Top-p and the repetition penalty are course extensions.
"""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set all but the k largest logits (per row) to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the most likely tokens until their cumulative probability
    first reaches p; drop the rest. The kept set adapts to the shape of the distribution:
    one token when the model is sure, many when it is not."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # drop a token if the mass BEFORE it already reaches p (so the token that crosses p is kept)
    drop_sorted = (cumulative - probs) >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def repetition_penalty(logits: torch.Tensor, previous: torch.Tensor, penalty: float) -> torch.Tensor:
    """Divide positive logits (multiply negative ones) of tokens that already occurred by `penalty` > 1."""
    logits = logits.clone()
    for b in range(logits.size(0)):
        ids = torch.unique(previous[b])
        vals = logits[b, ids]
        logits[b, ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                previous: torch.Tensor | None = None, rep_penalty: float | None = None, generator=None) -> torch.Tensor:
    """logits: (B, V) for the next position. Returns token ids of shape (B, 1)."""
    if rep_penalty is not None and previous is not None:
        logits = repetition_penalty(logits, previous, rep_penalty)
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, generator=None, **sampling) -> torch.Tensor:
    """Generation loop with pluggable decoding (naive recomputation; sliding window of block_size)."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.config.block_size :])[:, -1, :]
        next_id = sample_next(logits, previous=idx, generator=generator, **sampling)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def distinct_n(tokens: list[int], n: int = 3) -> float:
    """Fraction of n-grams that are unique: a simple measure of repetitiveness (1.0 = no repeats)."""
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))
