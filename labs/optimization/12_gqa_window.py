"""
Lab 12b — Grouped-query attention and sliding windows: what do they cost in quality?
Tier A (about 3 minutes, CPU)

    python labs/optimization/12_gqa_window.py

Trains the same MiniGPT with fewer key/value heads (GQA) and with a sliding attention
window, and reports parameters, KV-cache bytes per token, validation loss, and the
agreement test (whose dependency is 20-30 characters long).
Predict first: which window size will break agreement, given that a 2-layer model can
relay information across at most two windows?
"""

import torch

from mlle.data import pocket
from mlle.foundations.experiments import agreement_accuracy
from mlle.foundations.train import TrainConfig, train_on_text
from mlle.inference.kvcache import kv_bytes_per_token

text = pocket.generate_corpus(10000, seed=0)
items = pocket.agreement_items(200, with_attractor=True)
floor = pocket.entropy_rate()["nats_per_char"]
variants = [
    ("baseline: 4 heads, 4 kv heads, full context", dict()),
    ("GQA: 4 heads, 2 kv heads", dict(n_kv_head=2)),
    ("GQA: 4 heads, 1 kv head", dict(n_kv_head=1)),
    ("window 32 (both layers)", dict(window=32)),
    ("window 16 (both layers)", dict(window=16)),
    ("window 8 (both layers)", dict(window=8)),
]
print(f"entropy floor {floor:.4f} nats/char; 2 layers, n_embd 64, 1500 steps, seed 0")
print(f"{'variant':46s} {'params':>8} {'KV bytes/token':>15} {'val loss':>9} {'agreement+attractor':>20}")
for name, kw in variants:
    cfg = dict(block_size=64, n_layer=2, n_head=4, n_embd=64)
    cfg.update(kw)
    model, tok, res = train_on_text(text, cfg, TrainConfig(steps=1500, batch_size=32, lr=1e-2, eval_every=1500), verbose=False)
    c = model.config
    kv = kv_bytes_per_token(c.n_layer, c.n_kv_head, c.n_embd // c.n_head, 4)
    print(f"{name:46s} {model.num_params():>8,} {kv:>15,} {res.final_val_loss:>9.4f} {agreement_accuracy(model, tok, items):>20.2f}")
print("\nwith a window the cache only needs the last `window` tokens per layer, whatever the sequence length")
