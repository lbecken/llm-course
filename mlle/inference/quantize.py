"""
Post-training weight quantization (Module 12).

Store each weight as a small integer plus a shared scale:

    w  ~=  scale * q        q in {-(2^(b-1) - 1), ..., +(2^(b-1) - 1)}      (symmetric, b bits)

    per-tensor   one scale for the whole matrix:      scale = max|W| / qmax
    per-channel  one scale per output row:            scale_i = max|W_i| / qmax

Memory drops from 32 bits to b bits per weight (plus the scales). The price is rounding
error of up to scale / 2 per weight, so a few large weights (which inflate the scale)
hurt every other weight in their group; that is why per-channel scales are better.

This module does "fake quantization": weights are rounded and immediately converted back
to float so that the unchanged model code can be used to measure the effect on quality.
A real deployment keeps the integers and uses integer or mixed kernels. nanochat does not
quantize; its FP8 training path (Module 13) is a related idea applied during training.
"""

from __future__ import annotations

import torch


def quantize_tensor(w: torch.Tensor, bits: int = 8, per_channel: bool = True):
    """Returns (q, scale): integer codes (stored in int8 for bits <= 8) and the scale(s)."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel and w.ndim == 2:
        amax = w.abs().amax(dim=1, keepdim=True)  # one scale per output row
    else:
        amax = w.abs().max()
    scale = amax.clamp(min=1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale


@torch.no_grad()
def quantize_model_(model: torch.nn.Module, bits: int = 8, per_channel: bool = True) -> dict:
    """Round every 2-D weight of `model` IN PLACE to b-bit precision. Returns size and error statistics."""
    n_weights, sq_err, sq_norm = 0, 0.0, 0.0
    seen = set()
    for p in model.parameters():
        if p.ndim != 2 or id(p) in seen:
            continue
        seen.add(id(p))
        q, scale = quantize_tensor(p, bits, per_channel)
        w_hat = dequantize_tensor(q, scale)
        sq_err += (p - w_hat).pow(2).sum().item()
        sq_norm += p.pow(2).sum().item()
        n_weights += p.numel()
        p.copy_(w_hat)
    return {
        "bits": bits,
        "per_channel": per_channel,
        "quantized_weights": n_weights,
        "bytes_fp32": n_weights * 4,
        "bytes_quantized": n_weights * bits / 8,
        "relative_error": (sq_err / max(sq_norm, 1e-30)) ** 0.5,  # ||W - W_hat|| / ||W||
    }
