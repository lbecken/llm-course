"""
Lab 07 — Train a small GPT, generate text, measure what it learned.   Tier A (1-2 minutes, CPU)

    python labs/foundations/07_train_minigpt.py                       # nanochat-style, AdamW
    python labs/foundations/07_train_minigpt.py --style gpt2
    python labs/foundations/07_train_minigpt.py --optimizer muon
    python labs/foundations/07_train_minigpt.py --style gpt2 --lr 0.3  # too large: collapses to the unigram solution
    python labs/foundations/07_train_minigpt.py --lr 0.3               # same rate, nanochat style: still trains
    python labs/foundations/07_train_minigpt.py --text-file notes.txt --steps 3000

This is the Part I milestone: a complete causal Transformer trained on a small corpus.
With the default synthetic corpus the result can be judged exactly: the validation loss
is compared with the entropy of the source, and number agreement is tested on minimal pairs.
Success criterion (defaults): validation loss within 0.06 nats/char of the entropy floor
and agreement accuracy with attractors > 0.95. Records go to results/runs/minigpt.jsonl.
"""

import argparse
from pathlib import Path

import torch

from mlle.data import pocket
from mlle.foundations.experiments import agreement_accuracy
from mlle.foundations.train import TrainConfig, train_on_text
from mlle.utils import REPO_ROOT, JsonlLogger, environment_report

ap = argparse.ArgumentParser()
ap.add_argument("--style", choices=["nanochat", "gpt2"], default="nanochat")
ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--n-layer", type=int, default=2)
ap.add_argument("--n-head", type=int, default=2)
ap.add_argument("--n-embd", type=int, default=64)
ap.add_argument("--block-size", type=int, default=64)
ap.add_argument("--sentences", type=int, default=20000)
ap.add_argument("--text-file", type=str, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", type=str, default="cpu")
args = ap.parse_args()

synthetic = args.text_file is None
text = pocket.generate_corpus(args.sentences, seed=0) if synthetic else Path(args.text_file).read_text(encoding="utf-8")

model, tok, result = train_on_text(
    text,
    dict(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd),
    TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, eval_every=max(1, args.steps // 6),
                seed=args.seed, device=args.device, optimizer=args.optimizer),
    style=args.style,
)
model.cpu().eval()
print(f"\n{result.num_params:,} parameters | {result.seconds:.1f} s of training | {result.tokens_per_second:,.0f} tokens/s")

record = dict(lab="07_train_minigpt", args=vars(args), final_val_loss=result.final_val_loss, history=result.history,
              tokens_per_second=result.tokens_per_second, num_params=result.num_params, env=environment_report())
if synthetic:
    floor = pocket.entropy_rate()["nats_per_char"]
    plain = agreement_accuracy(model, tok, pocket.agreement_items(200, with_attractor=False))
    attr = agreement_accuracy(model, tok, pocket.agreement_items(200, with_attractor=True))
    print(f"validation loss {result.final_val_loss:.4f} nats/char | entropy floor {floor:.4f} | gap {result.final_val_loss - floor:.4f}")
    print(f"agreement accuracy: {plain:.2f} (simple), {attr:.2f} (with an attractor of the opposite number)")
    record.update(entropy_floor=floor, agreement=plain, agreement_attractor=attr)

prompt = torch.tensor([tok.encode("\nthe " if synthetic else text[:20])])
g = torch.Generator().manual_seed(args.seed)
# 200 new tokens exceed block_size, so use the naive generator, which re-reads a sliding window of
# the last block_size tokens at every step. (The cached generator is limited to block_size: Module 11.)
sample = model.generate(prompt, 200, temperature=1.0, generator=g)
print("\n--- sample (temperature 1.0) ---")
print(tok.decode(sample[0]).strip())
JsonlLogger(REPO_ROOT / "results" / "runs" / "minigpt.jsonl").log(**record)
