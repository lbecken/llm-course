"""
Three ways to tell a Transformer where a token is (Module 04).

Attention by itself is permutation-equivariant: shuffle the input tokens and the outputs
are shuffled the same way, nothing else changes. Order has to be injected explicitly.

    learned      a trainable vector per position, ADDED to the token embedding (GPT-2)
    sinusoidal   a fixed vector of sines and cosines per position, ADDED (Vaswani et al. 2017)
    rotary/RoPE  no vector is added; instead the QUERY and KEY vectors are ROTATED by an
                 angle proportional to their position, inside every attention layer
                 (Su et al. 2021). Used by nanochat.

The defining property of RoPE, tested in tests/test_positional.py:

    <rotate(q, m), rotate(k, n)>  depends on the positions only through (m - n).
"""

from __future__ import annotations

import math

import torch


def sinusoidal_positions(seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Fixed positional encodings, shape (seq_len, dim).

    PE[t, 2i]   = sin(t / base^(2i/dim))
    PE[t, 2i+1] = cos(t / base^(2i/dim))

    Each pair of channels is a clock hand turning at its own speed: fast for small i,
    slow for large i. Together the hands give every position a distinct code."""
    assert dim % 2 == 0
    t = torch.arange(seq_len, dtype=torch.float32)[:, None]  # (T, 1)
    inv_freq = torch.exp(-math.log(base) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)  # (dim/2,)
    angles = t * inv_freq  # (T, dim/2)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles.cos()
    return pe


def rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0, offset: int = 0):
    """Rotation tables for RoPE. Returns (cos, sin), each of shape (seq_len, head_dim/2).

    Channel pair i rotates by angle  theta[t, i] = t * base^(-2i/head_dim)  at position t.
    `offset` shifts the positions; the KV cache uses it to continue where it stopped.
    nanochat uses base=100000 (nanochat/gpt.py, GPT._precompute_rotary_embeddings)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # (D/2,)
    t = torch.arange(offset, offset + seq_len, dtype=torch.float32)  # (T,)
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs of x by the position-dependent angles.

    x:        (..., T, D)   queries or keys, any leading dims (batch, heads)
    cos, sin: (T, D/2)

    The D channels are split into two halves; channel j is paired with channel j + D/2
    and each pair (x1, x2) is treated as a point in a 2-D plane:

        y1 =  x1 cos + x2 sin
        y2 = -x1 sin + x2 cos

    This is exactly nanochat's `apply_rotary_emb`. It rotates by -theta, the transpose of
    the textbook convention; the two are equivalent because only the RELATIVE rotation
    between a query and a key ever matters."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)
