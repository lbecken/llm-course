"""
Scaled dot-product attention, three times (Module 05):

    attention_loops        nested Python loops, one head, no batch: the definition
    attention              the same computation as matrix algebra, batched, with masks
    MultiHeadSelfAttention a trainable module that also returns its attention weights

tests/test_attention.py checks that all three agree with each other and with PyTorch's
optimized `F.scaled_dot_product_attention`, which is what nanochat calls on hardware
without Flash Attention 3 (nanochat/flash_attention.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_loops(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    """The definition, with nothing vectorised.

    q: (T, D) one query vector per position        k: (T, D) one key vector per position
    v: (T, Dv) one value vector per position
    Returns (out, weights): out (T, Dv), weights (T, T) with weights[i, j] = how much
    position i reads from position j."""
    T, D = q.shape
    weights = torch.zeros(T, T, dtype=q.dtype)
    out = torch.zeros(T, v.shape[1], dtype=v.dtype)
    for i in range(T):  # for every position that is reading ...
        visible = range(i + 1) if causal else range(T)  # ... causal: only itself and the past
        scores = torch.stack([(q[i] * k[j]).sum() / math.sqrt(D) for j in visible])
        probs = torch.softmax(scores, dim=0)  # non-negative, sums to 1
        for slot, j in enumerate(visible):
            weights[i, j] = probs[slot]
            out[i] += probs[slot] * v[j]  # weighted average of value vectors
    return out, weights


def causal_mask(t_query: int, t_key: int | None = None, window: int | None = None, device=None) -> torch.Tensor:
    """Boolean mask of shape (t_query, t_key); True = "may attend".

    With t_key == t_query this is the familiar lower triangle. With t_key > t_query the
    queries are taken to be the LAST t_query positions of the key sequence, which is the
    situation during cached decoding (Module 11).
    `window` additionally limits attention to the `window` previous tokens plus the token
    itself (sliding-window attention, same convention as nanochat's SDPA fallback)."""
    t_key = t_query if t_key is None else t_key
    row = (t_key - t_query) + torch.arange(t_query, device=device)[:, None]  # absolute position of each query
    col = torch.arange(t_key, device=device)[None, :]
    mask = col <= row
    if window is not None:
        mask = mask & ((row - col) <= window)
    return mask


def attention(q, k, v, mask: torch.Tensor | None = None, scale: float | None = None):
    """Batched scaled dot-product attention.

    q: (..., Tq, D)   k: (..., Tk, D)   v: (..., Tk, Dv)   mask: broadcastable to (..., Tq, Tk)
    Returns (out, weights) with out (..., Tq, Dv) and weights (..., Tq, Tk).

        scores  = q @ k^T / sqrt(D)            every query dotted with every key
        scores  = -inf where mask is False     forbidden positions get zero probability
        weights = softmax(scores, dim=-1)      each row is a probability distribution
        out     = weights @ v                  each output is an average of value vectors
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention with every tensor shape spelled out.

    Stand-alone teaching module used in Module 05. The MiniGPT model (Module 06) has its
    own attention class with the modern options (RoPE, QK-norm, GQA, KV cache)."""

    def __init__(self, n_embd: int, n_head: int, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D): heads become a batch-like dimension
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, causal: bool = True, return_weights: bool = False):
        B, T, C = x.shape
        q = self.split_heads(self.q_proj(x))  # (B, H, T, D)
        k = self.split_heads(self.k_proj(x))  # (B, H, T, D)
        v = self.split_heads(self.v_proj(x))  # (B, H, T, D)
        mask = causal_mask(T, device=x.device) if causal else None  # (T, T), broadcast over B and H
        y, weights = attention(q, k, v, mask)  # y (B, H, T, D), weights (B, H, T, T)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # put the heads back side by side
        y = self.out_proj(y)  # (B, T, C)
        return (y, weights) if return_weights else y


def sdpa_reference(q, k, v, causal: bool = True) -> torch.Tensor:
    """PyTorch's fused implementation, for equivalence tests and benchmarks."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
