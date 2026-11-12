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
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged_weight(self) -> torch.Tensor:
        """The single matrix equivalent to base + adapter (what you would deploy)."""
        return self.base.weight + self.scale * (self.B @ self.A)


def add_lora(model: nn.Module, r: int = 4, alpha: float = 8.0, targets=("c_q", "c_v")) -> list[nn.Parameter]:
    """Freeze the whole model and wrap the named Linear layers of every block with LoRA.
    Returns the list of trainable (adapter) parameters."""
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
    return trainable


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
