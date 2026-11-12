"""
Lab 11 — The KV cache: correct first, then fast.   Tier A (about 1 minute, CPU; faster on GPU)

    python labs/optimization/11_kv_cache.py
    python labs/optimization/11_kv_cache.py --device cuda

1. Correctness: logits from prefill + token-by-token decode equal a full forward pass;
   cached generation emits exactly the tokens of naive generation.
2. Memory: bytes held by the cache versus the formula.
3. Speed: naive versus cached generation as the sequence grows, and the work each does.
Success criterion: maximum logit difference below 1e-4, identical tokens, and a speed-up
that GROWS with sequence length. Records go to results/runs/kv_cache.jsonl.
"""

import argparse

import torch

from mlle.bench.harness import record, time_fn
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.inference.kvcache import KVCache, generate_cached, kv_bytes_per_token
from mlle.utils import REPO_ROOT, JsonlLogger, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--n-layer", type=int, default=4)
ap.add_argument("--n-embd", type=int, default=128)
ap.add_argument("--n-head", type=int, default=4)
args = ap.parse_args()
device = pick_device(args.device)

torch.manual_seed(0)
V, BLOCK = 512, 1024
model = MiniGPT(MiniGPTConfig.nanochat_style(V, block_size=BLOCK, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)).to(device).eval()
with torch.no_grad():
    for p in model.parameters():  # untrained zero-initialised projections would make every comparison trivial
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.1)
print(f"model: {model.num_params():,} parameters on {device}\n")

# %% 1. Correctness
idx = torch.randint(0, V, (2, 48), device=device)
with torch.no_grad():
    full = model(idx)
    cache = KVCache.for_model(model, batch_size=2, max_len=48)
    parts = [model(idx[:, :16], cache=cache)]  # prefill 16 tokens
    parts += [model(idx[:, t : t + 1], cache=cache) for t in range(16, 48)]  # decode 32 tokens one by one
    diff = (torch.cat(parts, dim=1) - full).abs().max().item()
    same = torch.equal(model.generate(idx[:, :8], 40, temperature=0.0), generate_cached(model, idx[:, :8], 40, temperature=0.0))
print(f"1. max |logit(cached) - logit(full)| = {diff:.2e}    greedy tokens identical: {same}")
assert diff < 1e-3 and same

# %% 2. Memory
c = model.config
per_token = kv_bytes_per_token(c.n_layer, c.n_kv_head, c.n_embd // c.n_head, 4)
cache = KVCache.for_model(model, batch_size=1, max_len=BLOCK)
print(f"\n2. cache layout: k and v of shape (layers={c.n_layer}, batch, kv_heads={c.n_kv_head}, max_len, head_dim={c.n_embd // c.n_head})")
print(f"   bytes per cached token = layers x 2 x kv_heads x head_dim x 4 = {per_token:,}")
print(f"   allocated for {BLOCK} tokens, batch 1: {cache.nbytes():,} bytes = {BLOCK} x {per_token:,}: {cache.nbytes() == BLOCK * per_token}")
print(f"   model weights: {sum(p.numel() for p in model.parameters()) * 4:,} bytes")

# %% 3. Speed
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "kv_cache.jsonl")
print(f"\n3. generate N new tokens after an 8-token prompt (greedy, batch 1)")
print(f"{'N':>5} {'naive s':>9} {'cached s':>9} {'speed-up':>9} {'naive tok/s':>12} {'cached tok/s':>13} {'positions processed (naive / cached)':>38}")
prompt = torch.randint(0, V, (1, 8), device=device)
for n in (32, 128, 512):
    naive = time_fn(lambda: model.generate(prompt, n, temperature=0.0), warmup=1, repeats=3, device=device)
    cached = time_fn(lambda: generate_cached(model, prompt, n, temperature=0.0), warmup=1, repeats=3, device=device)
    work_naive = sum(8 + i for i in range(n))  # step i re-processes the whole sequence so far
    work_cached = 8 + (n - 1)  # the prompt once, then one position per step
    print(f"{n:>5} {naive['median_s']:>9.3f} {cached['median_s']:>9.3f} {naive['median_s'] / cached['median_s']:>8.1f}x "
          f"{n / naive['median_s']:>12.0f} {n / cached['median_s']:>13.0f} {work_naive:>24,} / {work_cached:,}")
    log.log(**record("kv_cache", n_new=n, naive_s=naive["median_s"], cached_s=cached["median_s"], device=str(device),
                     params=model.num_params(), max_logit_diff=diff))
