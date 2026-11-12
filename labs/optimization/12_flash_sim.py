"""
Lab 12a — Same attention, three memory footprints.   Tier A (about 30 s, CPU)

    python labs/optimization/12_flash_sim.py
    python labs/optimization/12_flash_sim.py --device cuda

Computes causal attention three ways and checks that the outputs agree:
    naive   materialise the (T, T) score matrix                    (mlle.foundations.attention)
    tiled   online softmax over blocks: one small tile at a time   (mlle.inference.flash_sim)
    fused   torch.nn.functional.scaled_dot_product_attention       (what nanochat falls back to)
Reports time and the size of the largest intermediate each needs.
Success criterion: all three agree to ~1e-5; the tiled version's largest intermediate is
independent of T. (The pure-Python tiled loop is SLOW: it demonstrates the algorithm,
not the speed. The speed comes from running the same loop inside one GPU kernel.)
"""

import argparse

import torch
import torch.nn.functional as F

from mlle.bench.harness import CudaPeakMemory, time_fn
from mlle.foundations.attention import attention, causal_mask
from mlle.inference.flash_sim import attention_tiled
from mlle.utils import pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
device = pick_device(args.device)
H, D, BLOCK = 4, 64, 128

print(f"causal attention, batch 1, {H} heads, head_dim {D}, fp32, device {device}; tile = {BLOCK} x {BLOCK}")
print(f"{'T':>6} {'max diff naive/tiled':>21} {'max diff naive/fused':>21} {'naive ms':>9} {'tiled ms':>9} {'fused ms':>9} "
      f"{'naive scores MiB':>17} {'tiled scores MiB':>17}" + ("  peak GPU MiB naive/fused" if device.type == "cuda" else ""))
for T in (256, 1024, 2048, 4096):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, T, D, device=device) for _ in range(3))
    mask = causal_mask(T, device=device)
    ref, _ = attention(q, k, v, mask)
    tiled, stats = attention_tiled(q, k, v, causal=True, block_q=BLOCK, block_k=BLOCK)
    fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    t_naive = time_fn(lambda: attention(q, k, v, mask), warmup=1, repeats=3, device=device)["median_s"]
    t_tiled = time_fn(lambda: attention_tiled(q, k, v, True, BLOCK, BLOCK), warmup=0, repeats=1, device=device)["median_s"]
    t_fused = time_fn(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=1, repeats=3, device=device)["median_s"]
    extra = ""
    if device.type == "cuda":
        with CudaPeakMemory(device) as m1:
            attention(q, k, v, mask)
        with CudaPeakMemory(device) as m2:
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        extra = f"  {m1.bytes / 2**20:8.1f} / {m2.bytes / 2**20:.1f}"
    print(f"{T:>6} {(ref - tiled).abs().max().item():>21.2e} {(ref - fused).abs().max().item():>21.2e} "
          f"{t_naive * 1e3:>9.1f} {t_tiled * 1e3:>9.1f} {t_fused * 1e3:>9.1f} "
          f"{H * stats['naive_score_elements'] * 4 / 2**20:>17.1f} {H * stats['peak_score_elements'] * 4 / 2**20:>17.2f}{extra}")
