"""
Lab 06 — Trace one forward pass through a complete Transformer.   Tier A (seconds, CPU)

    python labs/foundations/06_transformer_trace.py
    python labs/foundations/06_transformer_trace.py --style gpt2

Prints every tensor shape from token ids to next-token probabilities, the parameter
budget by component, how the residual stream changes layer by layer, and what each
architectural switch does to the parameter count.
Success criterion: you can predict every printed shape before reading it.
"""

import argparse

import torch

from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--style", choices=["nanochat", "gpt2"], default="nanochat")
args = ap.parse_args()

text = "the cat sleeps ."
tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz .\n")
make = MiniGPTConfig.nanochat_style if args.style == "nanochat" else MiniGPTConfig.gpt2_style
torch.manual_seed(0)
cfg = make(tok.vocab_size, block_size=32, n_layer=3, n_head=4, n_embd=48)
model = MiniGPT(cfg).eval()
B, T, C, H, V = 1, len(text), cfg.n_embd, cfg.n_head, cfg.vocab_size

# %% 1. Shapes, stage by stage
idx = torch.tensor([tok.encode(text)])
print(f"style: {args.style}\ntext {text!r} -> token ids {tuple(idx.shape)} = (B={B}, T={T})")
x = model.wte(idx)
print(f"wte lookup                    -> {tuple(x.shape)} = (B, T, C={C})")
block = model.h[0]
xn = block.norm1(x)
attn = block.attn
q = attn.c_q(xn).view(B, T, H, C // H).transpose(1, 2)
print(f"norm -> c_q -> split heads    -> {tuple(q.shape)} = (B, H={H}, T, D={C // H})")
print(f"scores q @ k^T                -> {(B, H, T, T)} = (B, H, T, T)   one T x T table per head")
print(f"weights @ v, merge heads      -> {(B, T, C)}")
h = block.mlp.c_fc(xn)
print(f"MLP c_fc (expand)             -> {tuple(h.shape)} = (B, T, {cfg.mlp_ratio}C)")
print(f"MLP c_proj (contract)         -> {(B, T, C)}")
logits = model(idx, record=True)
print(f"after {cfg.n_layer} blocks + final norm   -> {(B, T, C)}")
print(f"lm_head                       -> {tuple(logits.shape)} = (B, T, V={V})")
probs = torch.softmax(logits[0, -1], dim=-1)
print(f"softmax of the LAST position  -> {tuple(probs.shape)} = (V,), sums to {probs.sum():.4f}")

# %% 2. Where the parameters are
print("\nparameters by component:")
groups = {"token embedding (wte)": model.wte.weight.numel()}
if model.wpe is not None:
    groups["position embedding (wpe)"] = model.wpe.weight.numel()
groups["attention (all layers)"] = sum(p.numel() for b in model.h for p in b.attn.parameters())
groups["MLP (all layers)"] = sum(p.numel() for b in model.h for p in b.mlp.parameters())
groups["norms"] = sum(p.numel() for n, p in model.named_parameters() if "norm" in n)
groups["lm_head"] = 0 if cfg.tie_weights else model.lm_head.weight.numel()
for name, n in groups.items():
    print(f"  {name:26s} {n:8,d}")
print(f"  {'total':26s} {model.num_params():8,d}   (sum of rows: {sum(groups.values()):,})")

# %% 3. The residual stream: each block adds a correction
stream = model.internals["residual_stream"]
print("\nresidual stream at the last position (RMS of the vector, and how much each block changed it):")
for i, s in enumerate(stream):
    rms = s[0, -1].pow(2).mean().sqrt().item()
    delta = "" if i == 0 else f"   |change| / |previous| = {(s[0, -1] - stream[i - 1][0, -1]).norm() / stream[i - 1][0, -1].norm():.3f}"
    print(f"  after {'embedding' if i == 0 else f'block {i - 1}':10s}: rms {rms:.3f}{delta}")
print("  (an untrained nanochat-style model shows change = 0: its blocks are initialised to the identity)")

# %% 4. One switch at a time
print("\nparameter count as each GPT-2 component is replaced by its nanochat counterpart:")
steps = [
    ("GPT-2 style", {}),
    ("+ no biases", dict(bias=False)),
    ("+ RMSNorm without parameters", dict(bias=False, norm="rmsnorm")),
    ("+ RoPE instead of learned positions", dict(bias=False, norm="rmsnorm", pos="rope")),
    ("+ untied output head", dict(bias=False, norm="rmsnorm", pos="rope", tie_weights=False)),
    ("+ relu^2, QK-norm, logit softcap (no parameters)", dict(bias=False, norm="rmsnorm", pos="rope", tie_weights=False, act="relu2", qk_norm=True, logit_softcap=15.0)),
]
for name, overrides in steps:
    m = MiniGPT(MiniGPTConfig.gpt2_style(tok.vocab_size, block_size=32, n_layer=3, n_head=4, n_embd=48, **overrides))
    print(f"  {name:50s} {m.num_params():7,d}")
