"""
Lab 13b — Gradient accumulation and where training memory goes.
Tier A (about 30 s, CPU)

    python labs/optimization/13_grad_accum.py

1. One batch of 32 sequences versus 4 micro-batches of 8 with the loss divided by 4:
   the accumulated gradient is the same.
2. Memory accounting for a training step: weights, gradients, optimizer state,
   activations. Measured as peak process memory in separate subprocesses for several
   batch sizes, with and without activation checkpointing.
Success criterion: gradient difference ~1e-7; peak memory grows with batch size by the
activation term only; checkpointing lowers that growth and costs time.
"""

import argparse
import subprocess
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint

from mlle.bench.harness import current_rss_bytes, peak_rss_bytes, reset_peak_rss
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--worker", nargs=2, metavar=("BATCH", "MODE"), help="internal: measure one configuration")
args = ap.parse_args()

CFG = dict(block_size=256, n_layer=8, n_head=4, n_embd=256)


def build():
    torch.manual_seed(0)
    return MiniGPT(MiniGPTConfig.nanochat_style(1024, **CFG))


if args.worker:  # ---- subprocess: one configuration, report peak memory and time
    batch, mode = int(args.worker[0]), args.worker[1]
    model = build()
    if mode == "checkpoint":  # recompute each block's activations during backward instead of storing them
        for block in model.h:
            inner = block.forward
            block.forward = lambda x, cache=None, record=False, _f=inner: checkpoint(_f, x, use_reentrant=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 1024, (batch, 256))
    opt.zero_grad(set_to_none=True)
    model(x[:1], x[:1]).backward()  # tiny warm-up step: allocates gradients and optimizer state
    opt.step()
    reset_peak_rss()
    base = current_rss_bytes()
    t0 = time.perf_counter()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        model(x, x).backward()
        opt.step()
    print(f"{max(0, peak_rss_bytes() - base)} {(time.perf_counter() - t0) / 2}")
    sys.exit(0)

# %% 1. Accumulation equals one large batch
model = build()
x = torch.randint(0, 1024, (32, 64))
model.zero_grad()
model(x, x).backward()
big = [p.grad.clone() for p in model.parameters()]
model.zero_grad()
for i in range(4):
    xb = x[i * 8 : (i + 1) * 8]
    (model(xb, xb) / 4).backward()  # gradients ADD across backward() calls; dividing keeps it a mean
worst = max((g - p.grad).abs().max().item() for g, p in zip(big, model.parameters()))
scale = max(g.abs().max().item() for g in big)
print(f"1. largest gradient difference, batch 32 vs 4 x 8 accumulated: {worst:.2e} (gradient magnitudes up to {scale:.2e})")

# %% 2. Where the memory goes
model = build()
n = model.num_params()
print(f"\n2. model: {n:,} parameters")
print(f"   weights {n * 4 / 2**20:6.1f} MiB | gradients {n * 4 / 2**20:6.1f} MiB | AdamW state (m and v) {2 * n * 4 / 2**20:6.1f} MiB   <- independent of batch size")
c = model.config
per_token_acts = c.n_layer * (c.n_embd * 12 + c.n_head * c.block_size * 2)  # rough: stream, q/k/v, 4C hidden, attention rows
print(f"   activations kept for backward: roughly {per_token_acts * 4 / 2**10:.0f} KiB per token of the batch (grows with batch x sequence x layers)")
print(f"\n   measured peak process memory during a training step, above the level before it (one process per configuration):")
print(f"{'batch':>8} {'tokens':>8} {'plain MiB':>10} {'s/step':>8} {'checkpointed MiB':>17} {'s/step':>8}")
for batch in (2, 8, 32):
    row = []
    for mode in ("plain", "checkpoint"):
        out = subprocess.run([sys.executable, __file__, "--worker", str(batch), mode], capture_output=True, text=True, check=True).stdout.split()
        row += [int(out[0]) / 2**20, float(out[1])]
    print(f"{batch:>8} {batch * 256:>8} {row[0]:>10.0f} {row[1]:>8.2f} {row[2]:>17.0f} {row[3]:>8.2f}")
print("   Halving the batch and accumulating twice trades memory for nothing but wall-clock neutrality;")
print("   checkpointing trades memory for extra forward computation.")
