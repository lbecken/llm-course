"""
LoRA: low-rank adaptation (Module 14). A course extension; nanochat does not use it.

Instead of updating a weight matrix W (out x in) directly, freeze it and learn a
low-rank correction:

    W_effective = W + (alpha / r) * B @ A        A: (r, in)   B: (out, r)   r << min(in, out)

A starts random, B starts at ZERO, so the adapted model initially equals the base model.
Only A and B are trained: 2*r*n numbers per n x n matrix instead of n^2. Because the base
weights never change, the adapter can be removed (or swapped) to recover the base model
exactly, and what the model can "forget" is limited by the rank.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # the pretrained weights are frozen
        self.r, self.scale = r, alpha / r
        self.enabled = True  # set False to run the untouched base model
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged_weight(self) -> torch.Tensor:
        """The single matrix equivalent to base + adapter (what you would deploy)."""
        return self.base.weight + self.scale * (self.B @ self.A)


class EmbeddingWithNewRows(nn.Module):
    """A frozen embedding table plus trainable CORRECTIONS for a few token ids.

    Pretraining never saw the chat special tokens, so their embedding rows are untrained
    noise, and no adapter on the attention or MLP matrices can supply them. Rather than
    unfreezing the table, we learn one extra vector per new token and add it at lookup.
    The base table is never modified, so disabling the adapter restores the base model."""

    def __init__(self, base: nn.Embedding, token_ids: list[int]):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.enabled = True
        slot = torch.full((base.num_embeddings,), len(token_ids), dtype=torch.long)  # last slot = "no correction"
        slot[token_ids] = torch.arange(len(token_ids))
        self.register_buffer("slot", slot)
        self.delta = nn.Parameter(torch.zeros(len(token_ids), base.embedding_dim))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, idx):
        out = self.base(idx)
        if not self.enabled:
            return out
        table = torch.cat([self.delta, torch.zeros(1, self.delta.shape[1], dtype=self.delta.dtype, device=self.delta.device)])
        return out + table[self.slot[idx]]


class LinearWithNewRows(nn.Module):
    """The output-side counterpart: a frozen lm_head plus trainable corrections to the
    score rows of a few token ids (so that the model can learn to EMIT the new tokens)."""

    def __init__(self, base: nn.Linear, token_ids: list[int]):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.enabled = True
        self.register_buffer("rows", torch.tensor(token_ids, dtype=torch.long))
        self.delta = nn.Parameter(torch.zeros(len(token_ids), base.in_features))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        return out.index_add(-1, self.rows, x @ self.delta.T)


def add_lora(model: nn.Module, r: int = 4, alpha: float = 8.0, targets=("c_q", "c_v"), new_token_ids: list[int] | None = None) -> list[nn.Parameter]:
    """Freeze the whole model and wrap the named Linear layers of every block with LoRA.
    With `new_token_ids`, also add trainable row corrections for those tokens to the
    embedding and to the output head. Returns the list of trainable (adapter) parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for block in model.h:
        for holder in (block.attn, block.mlp):
            for name in targets:
                layer = getattr(holder, name, None)
                if isinstance(layer, nn.Linear):
                    wrapped = LoRALinear(layer, r, alpha)
                    setattr(holder, name, wrapped)
                    trainable += [wrapped.A, wrapped.B]
    if new_token_ids:
        assert model.lm_head.weight is not model.wte.weight, "new-token rows assume an untied output head"
        model.wte = EmbeddingWithNewRows(model.wte, new_token_ids)
        model.lm_head = LinearWithNewRows(model.lm_head, new_token_ids)
        trainable += [model.wte.delta, model.lm_head.delta]
    return trainable


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Switch every adapter on or off. Off = exactly the frozen base model."""
    for m in model.modules():
        if isinstance(m, (LoRALinear, EmbeddingWithNewRows, LinearWithNewRows)):
            m.enabled = enabled


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
