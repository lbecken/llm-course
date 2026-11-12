"""
Lab 09b — A miniature compute-optimal frontier.   Tier A-B (about 8 minutes, CPU)

    python labs/pretraining/09_scaling_sweep.py
    python labs/pretraining/09_scaling_sweep.py --quick      # 2 budgets, ~2 minutes

For a FIXED compute budget F (in FLOPs) there is a choice: a small model trained on many
tokens, or a large model trained on few. This script trains MiniGPTs of several widths
at several budgets, giving each model   tokens = F / (FLOPs per token)   of training,
and reports the validation loss of every (budget, width) cell.
Hypothesis to test: the best width GROWS with the budget (the essence of compute-optimal
scaling), and for every budget both too-small and too-large models lose.
Results are appended to results/runs/scaling_sweep.jsonl.
"""

import argparse

import torch

from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import TrainConfig, split_tokens, train
from mlle.utils import REPO_ROOT, JsonlLogger, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

BLOCK, BATCH, LAYERS = 64, 32, 2
widths = [8, 16, 32, 64]
budgets = [1e10, 4e10] if args.quick else [1e10, 4e10, 1.6e11, 6.4e11]

text = pocket.generate_corpus(40000, seed=0)
tok = CharTokenizer(text)
train_tokens, val_tokens = split_tokens(torch.tensor(tok.encode(text)))
floor = pocket.entropy_rate()["nats_per_char"]
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "scaling_sweep.jsonl")


def flops_per_token(model: MiniGPT) -> float:
    c = model.config
    matmul = sum(p.numel() for b in model.h for p in b.parameters() if p.ndim == 2) + model.lm_head.weight.numel()
    return 6 * matmul + 12 * c.n_embd * c.block_size * c.n_layer


print(f"entropy floor {floor:.4f} nats/char; each cell: validation loss (optimizer steps)")
print(f"{'budget (FLOPs)':>15} " + " ".join(f"{'width ' + str(w):>16}" for w in widths) + "   best width")
for budget in budgets:
    cells, losses = [], {}
    for w in widths:
        seed_everything(args.seed)
        model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=BLOCK, n_layer=LAYERS, n_head=2, n_embd=w))
        fpt = flops_per_token(model)
        steps = max(10, int(budget / fpt / (BATCH * BLOCK)))
        result = train(model, train_tokens, val_tokens,
                       TrainConfig(steps=steps, batch_size=BATCH, lr=1e-2, eval_every=steps, seed=args.seed, warmup_steps=min(20, steps // 5)), verbose=False)
        losses[w] = result.final_val_loss
        cells.append(f"{result.final_val_loss:.4f} ({steps:>5d})")
        log.log(lab="09_scaling_sweep", budget=budget, width=w, steps=steps, params=model.num_params(), flops_per_token=fpt,
                tokens=steps * BATCH * BLOCK, val_loss=result.final_val_loss, seed=args.seed)
    best = min(losses, key=losses.get)
    print(f"{budget:>15.1e} " + " ".join(f"{c:>16}" for c in cells) + f"   {best}")
