"""
Lab 13c — Why master weights stay in fp32.   Tier A (about 2 minutes, CPU)

    python labs/optimization/13_master_weights.py

Trains the same MiniGPT three ways. The arithmetic of every step is identical (fp32);
the only difference is the format in which the WEIGHTS are stored between steps:

    fp32   weights kept as computed
    fp16   weights rounded to 16-bit floats after every optimizer step
    bf16   weights rounded to bfloat16 after every optimizer step

and counts, at each step, the fraction of weights that the update failed to change.
Predict first: bf16 resolves about 2-3 significant digits. With a learning rate of
1e-3, how large is a typical update relative to a weight, and what happens to it?
Success criterion: you can explain the ordering of the three final losses and why a
larger learning rate narrows the gap.
"""

import argparse

import torch

from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import estimate_loss, get_batch, split_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-2])
args = ap.parse_args()

text = pocket.generate_corpus(10000, seed=0)
tok = CharTokenizer(text)
train_tokens, val_tokens = split_tokens(torch.tensor(tok.encode(text)))
STORE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def run(lr: float, store: str) -> tuple[float, float]:
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64))
    matrices = [p for p in model.parameters() if p.ndim == 2]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    unchanged, total = 0, 0
    for _ in range(args.steps):
        x, y = get_batch(train_tokens, 32, 64, gen)
        before = [p.detach().clone() for p in matrices]
        opt.zero_grad(set_to_none=True)
        model(x, y).backward()
        opt.step()
        if STORE[store] is not None:
            with torch.no_grad():
                for p in matrices:
                    p.copy_(p.to(STORE[store]).float())  # what survives storage in the low-precision format
        unchanged += sum((p == b).sum().item() for p, b in zip(matrices, before))
        total += sum(p.numel() for p in matrices)
    return estimate_loss(model, val_tokens, 32, 20), unchanged / total


print(f"MiniGPT 2 layers x 64 on Pocket, AdamW, {args.steps} steps, seed 0. All arithmetic in fp32; only weight STORAGE differs.")
print(f"{'lr':>8} {'stored as':>10} {'val loss':>9} {'weights unchanged by a step':>28}")
for lr in args.lrs:
    for store in STORE:
        loss, frac = run(lr, store)
        print(f"{lr:>8.0e} {store:>10} {loss:>9.4f} {frac:>27.1%}")
print("\nAn AdamW step moves each weight by about lr. If lr is below half the format's spacing at the")
print("weight's magnitude, the rounded weight does not move at all: the update is lost, every step.")
