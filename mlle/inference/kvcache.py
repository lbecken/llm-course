"""
A key/value cache for MiniGPT (Module 11).

During generation, position t needs the keys and values of positions 0..t in every
layer. Positions 0..t-1 were already processed at earlier steps and, because attention
is causal, NOTHING computed for them can change when a new token arrives. So their keys
and values are stored and only the newest token is pushed through the network.

Layout: one pre-allocated tensor per layer for keys and one for values,

    k[layer]: (B, n_kv_head, max_len, head_dim)      v[layer]: same

plus a single integer `pos` = number of positions filled. nanochat's KVCache
(nanochat/engine.py) is the same idea with layout (n_layers, B, T, H, D) to match
Flash Attention 3, and a per-row `cache_seqlens` tensor instead of one integer.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layer: int, batch_size: int, n_kv_head: int, head_dim: int, max_len: int,
                 device=None, dtype=torch.float32):
        shape = (n_layer, batch_size, n_kv_head, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0
        self.max_len = max_len

    @classmethod
    def for_model(cls, model, batch_size: int, max_len: int | None = None) -> "KVCache":
        c = model.config
        p = next(model.parameters())
        return cls(c.n_layer, batch_size, c.n_kv_head, c.n_embd // c.n_head,
                   max_len or c.block_size, device=p.device, dtype=p.dtype)

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Write the new keys/values at [pos, pos + T) and return views of everything so far."""
        T = k_new.shape[2]
        assert self.pos + T <= self.max_len, "KV cache is full"
        self.k[layer_idx, :, :, self.pos : self.pos + T] = k_new
        self.v[layer_idx, :, :, self.pos : self.pos + T] = v_new
        end = self.pos + T
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def advance(self, num_tokens: int) -> None:
        """Called once per forward pass, after the last layer has appended."""
        self.pos += num_tokens

    def expand_batch(self, batch_size: int) -> "KVCache":
        """Copy a batch-1 cache into a batch-N cache: prefill the prompt once, then decode
        N samples in parallel (nanochat: KVCache.prefill)."""
        assert self.k.shape[1] == 1
        new = KVCache.__new__(KVCache)
        new.k = self.k.expand(-1, batch_size, -1, -1, -1).clone()
        new.v = self.v.expand(-1, batch_size, -1, -1, -1).clone()
        new.pos, new.max_len = self.pos, self.max_len
        return new

    def nbytes(self) -> int:
        """Memory actually reserved by the cache."""
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


def kv_bytes_per_token(n_layer: int, n_kv_head: int, head_dim: int, bytes_per_value: int = 4) -> int:
    """Bytes needed to remember ONE token: keys and values, for every layer and kv-head.
    Same formula as nanochat's GPT.kv_bytes_per_token."""
    return n_layer * 2 * n_kv_head * head_dim * bytes_per_value


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                    top_k: int | None = None, generator=None) -> torch.Tensor:
    """Same outputs as MiniGPT.generate, but each decode step processes ONE token.

    Phase 1 (prefill): run the whole prompt once, filling the cache.
    Phase 2 (decode):  feed only the newest token; attention reads the past from the cache."""
    B, T0 = idx.shape
    assert T0 + max_new_tokens <= model.config.block_size, (
        "cached generation cannot run past block_size: the cache has no eviction. "
        "Use MiniGPT.generate (sliding window, no cache) for longer outputs."
    )
    cache = KVCache.for_model(model, B, max_len=T0 + max_new_tokens)
    logits = model(idx, cache=cache)[:, -1, :]  # prefill
    for _ in range(max_new_tokens):
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
        if idx.shape[1] == T0 + max_new_tokens:
            break
        logits = model(next_id, cache=cache)[:, -1, :]  # decode: T = 1
    return idx
