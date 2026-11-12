"""
Lab 09a — Parameters, FLOPs and budgets, counted exactly.   Tier A (seconds, CPU)

    python labs/pretraining/09_param_flops.py                 # MiniGPT only
    python labs/pretraining/09_param_flops.py --nanochat      # + nanochat miniseries (needs `make env-cpu`)

1. Where the parameters of a model are, and which of them cost compute.
2. FLOPs per token from the "6 per matmul parameter" rule plus the attention term, and a
   check of the rule against a measured forward+backward time.
3. With --nanochat: the same accounting for nanochat depths, straight from its own code.
Success criterion: you can predict how FLOPs per token change when width doubles, and
explain why total parameters and FLOPs can move in different directions.
"""

import argparse
import time

import torch

from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--nanochat", action="store_true")
args = ap.parse_args()


def accounting(model: MiniGPT) -> dict:
    c = model.config
    matmul = sum(p.numel() for b in model.h for p in b.parameters() if p.ndim == 2) + model.lm_head.weight.numel()
    lookup = model.wte.weight.numel() if not c.tie_weights else 0  # a tied table is counted once, as a matmul
    attn = 12 * c.n_head * (c.n_embd // c.n_head) * c.block_size * c.n_layer  # forward+backward, per token
    return {"total": model.num_params(), "matmul": matmul, "lookup": lookup, "flops_per_token": 6 * matmul + attn, "attn_share": attn / (6 * matmul + attn)}


# %% 1-2. Width and depth sweeps
print("MiniGPT, nanochat style, vocabulary 4096, block_size 256")
print(f"{'layers':>6} {'width':>6} {'total params':>13} {'matmul params':>14} {'lookup params':>14} {'FLOPs/token':>12} {'attention share':>16}")
rows = {}
for n_layer, n_embd in [(4, 128), (4, 256), (4, 512), (8, 256), (2, 256)]:
    model = MiniGPT(MiniGPTConfig.nanochat_style(4096, block_size=256, n_layer=n_layer, n_head=max(1, n_embd // 64), n_embd=n_embd))
    a = accounting(model)
    rows[(n_layer, n_embd)] = (model, a)
    print(f"{n_layer:>6} {n_embd:>6} {a['total']:>13,} {a['matmul']:>14,} {a['lookup']:>14,} {a['flops_per_token']:>12.3e} {a['attn_share']:>15.1%}")

print("\ncheck of the FLOP rule against measured time (CPU, fp32): if time is proportional to FLOPs,")
print("the last column is constant across model sizes")
print(f"{'layers':>6} {'width':>6} {'ms / fwd+bwd':>13} {'tokens/s':>10} {'achieved FLOPs/s':>17}")
for (n_layer, n_embd), (model, a) in rows.items():
    x = torch.randint(0, 4096, (4, 256))
    model(x, x).backward()  # warm-up
    t0 = time.perf_counter()
    for _ in range(3):
        model.zero_grad()
        model(x, x).backward()
    dt = (time.perf_counter() - t0) / 3
    tps = x.numel() / dt
    print(f"{n_layer:>6} {n_embd:>6} {dt * 1e3:>13.1f} {tps:>10,.0f} {tps * a['flops_per_token']:>17.3e}")

# %% 3. nanochat's own accounting
if args.nanochat:
    from mlle.nanochat_bridge import require_nanochat

    require_nanochat()
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))
    from estimate_run import build_meta

    print("\nnanochat at the pinned revision (vocab 32768, sequence 2048, window pattern SSSL)")
    print(f"{'depth':>5} {'width':>6} {'wte':>12} {'value_embeds':>13} {'lm_head':>12} {'block matrices':>15} {'total':>14} {'lookup share':>13} {'FLOPs/token':>12}")
    for depth in (4, 8, 12, 20, 26):
        m = build_meta(depth, 64, 128, 2048, 32768, "SSSL")
        c = m.num_scaling_params()
        lookup = (c["wte"] + c["value_embeds"]) / c["total"]
        print(f"{depth:>5} {m.config.n_embd:>6} {c['wte']:>12,} {c['value_embeds']:>13,} {c['lm_head']:>12,} "
              f"{c['transformer_matrices']:>15,} {c['total']:>14,} {lookup:>12.1%} {m.estimate_flops():>12.3e}")
