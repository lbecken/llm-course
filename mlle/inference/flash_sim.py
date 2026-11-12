"""
Attention without the T x T matrix (Module 12).

`attention_tiled` computes EXACTLY the same result as ordinary causal attention, but it
never holds more than one (block_q x block_k) tile of scores at a time. This is the
algorithmic idea of FlashAttention (Dao et al., 2022), written in plain PyTorch so the
bookkeeping is visible. Real FlashAttention implements the same loop inside one GPU
kernel so that each tile lives in fast on-chip memory; here the point is only to show
that the quadratic MEMORY of attention is a property of the naive implementation, not of
the mathematics.

The trick is an "online" softmax. For one query, split its scores s into chunks
s^(1), s^(2), ...  and keep three running quantities:

    m   the largest score seen so far                       (for numerical stability)
    l   the sum of exp(score - m) over all scores seen      (the softmax normaliser)
    o   the sum of exp(score - m) * value over all seen     (the un-normalised output)

When a new chunk arrives with a larger maximum m', everything accumulated so far is
rescaled by exp(m - m') and the new chunk is added. At the end, output = o / l.
"""

from __future__ import annotations

import math

import torch


def attention_tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    block_q: int = 64, block_k: int = 64):
    """q: (..., Tq, D), k: (..., Tk, D), v: (..., Tk, Dv), with Tq == Tk when causal.
    Returns (out, stats); out equals softmax(q k^T / sqrt(D)) v up to rounding."""
    Tq, D = q.shape[-2], q.shape[-1]
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=v.dtype, device=v.device)
    tiles = 0
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[..., qs:qe, :]  # (..., bq, D)
        m = torch.full(qb.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)  # (..., bq)
        l = torch.zeros_like(m)
        o = torch.zeros(*qb.shape[:-1], v.shape[-1], dtype=v.dtype, device=v.device)  # (..., bq, Dv)
        k_end = qe if causal else Tk  # causal: queries in this block never look past position qe - 1
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = (qb @ k[..., ks:ke, :].transpose(-2, -1)) * scale  # ONE tile of scores: (..., bq, bk)
            if causal:
                rows = torch.arange(qs, qe, device=q.device)[:, None]
                cols = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            p = torch.exp(s - m_new[..., None])  # this tile's un-normalised weights
            alpha = torch.exp(m - m_new)  # rescale what was accumulated under the old maximum
            l = alpha * l + p.sum(dim=-1)
            o = alpha[..., None] * o + p @ v[..., ks:ke, :]
            m = m_new
            tiles += 1
        out[..., qs:qe, :] = o / l[..., None]
    stats = {
        "tiles": tiles,
        "peak_score_elements": block_q * block_k,  # per (batch, head)
        "naive_score_elements": Tq * Tk,
    }
    return out, stats
