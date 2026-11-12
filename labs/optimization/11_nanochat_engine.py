"""
Lab 11b — nanochat's Engine versus its naive generator.   Tier A (CPU ok). Needs `make env-cpu`
and the miniature checkpoint from `make nanochat-local-data nanochat-local-train`.

    export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local
    python labs/optimization/11_nanochat_engine.py --model-tag pocket-d4

nanochat ships the same comparison in `nanochat/engine.py` (`python -m nanochat.engine`), but
that entry point calls torch.cuda.synchronize() and therefore needs a GPU. This script does
the same thing on any device:
1. greedy tokens from GPT.generate (no cache) and from Engine.generate (KV cache) must be identical;
2. timing of both;
3. the cache's size according to nanochat's own accounting methods.
Success criterion: "Match: True".
"""

import argparse
import time

import torch

from mlle.nanochat_bridge import require_nanochat

require_nanochat()
from nanochat.checkpoint_manager import load_model  # noqa: E402
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_init  # noqa: E402
from nanochat.engine import Engine, KVCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="base")
ap.add_argument("--model-tag", default=None)
ap.add_argument("--prompt", default="the cats that chase the dog")
ap.add_argument("--max-tokens", type=int, default=64)
args = ap.parse_args()

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
sync = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
prompt = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())
max_tokens = min(args.max_tokens, model.config.sequence_len - len(prompt))

# %% 1-2. Naive generation: every step re-runs the whole sequence
sync()
t0 = time.perf_counter()
reference = list(model.generate(prompt, max_tokens=max_tokens, temperature=0.0))
sync()
t_naive = time.perf_counter() - t0

# Engine: one prefill, then one token per step through the KV cache
engine = Engine(model, tokenizer)
sync()
t0 = time.perf_counter()
cached = []
for token_column, token_masks in engine.generate(prompt, num_samples=1, max_tokens=max_tokens, temperature=0.0):
    cached.append(token_column[0])
sync()
t_engine = time.perf_counter() - t0

n = min(len(reference), len(cached))  # the Engine stops at <|bos|>/<|assistant_end|>; the naive loop does not
print(f"prompt: {args.prompt!r} ({len(prompt)} tokens), {max_tokens} new tokens, device {device}, dtype {COMPUTE_DTYPE}")
print(f"naive : {t_naive:.3f} s   {tokenizer.decode(reference[:n])!r}")
print(f"engine: {t_engine:.3f} s   {tokenizer.decode(cached[:n])!r}")
print(f"Match: {reference[:n] == cached[:n]}   speed-up: {t_naive / t_engine:.1f}x")

# %% 3. nanochat's own cache accounting
c = model.config
cache = KVCache(batch_size=1, num_heads=c.n_kv_head, seq_len=c.sequence_len, head_dim=c.n_embd // c.n_head,
                num_layers=c.n_layer, device=device, dtype=COMPUTE_DTYPE)
print(f"\nKVCache tensors: k_cache {tuple(cache.k_cache.shape)} = (layers, batch, T, kv_heads, head_dim), same for v_cache")
print(f"kv_bytes_per_token() = {model.kv_bytes_per_token():,} bytes   x sequence_len {c.sequence_len} = {model.kv_bytes_per_token() * c.sequence_len:,} bytes per row")
print(f"allocated: {(cache.k_cache.numel() + cache.v_cache.numel()) * cache.k_cache.element_size():,} bytes")
print(f"kv_read_bytes(context={c.sequence_len}) = {model.kv_read_bytes(c.sequence_len):,}  (window pattern {c.window_pattern})")
print(f"estimate_decode_flops(context={c.sequence_len}) = {model.estimate_decode_flops(c.sequence_len):.3e}   "
      f"estimate_prefill_flops({len(prompt)}) = {model.estimate_prefill_flops(len(prompt)):.3e}")
