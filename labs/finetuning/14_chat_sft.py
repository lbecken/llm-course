"""
Lab 14a — From completion to conversation: supervised fine-tuning.
Tier A (about 4 minutes on a 4-core CPU the first time; the base model is cached afterwards)

    python labs/finetuning/14_chat_sft.py
    python labs/finetuning/14_chat_sft.py --steps 300 --n-convs 100     # less fine-tuning data

The base model was pretrained on plain text: Pocket sentences and facts like "47+38=85".
It has never seen a conversation. This lab renders conversations with nanochat's special
tokens and loss mask, fine-tunes, and measures FOUR things for every variant:

    chat (seen)     exact answers to questions whose facts were in the pretraining text
    chat (new)      exact answers to held-out questions (never seen in any form)
    raw (new)       the pretraining-style prompt "47+38=" on held-out questions
    Pocket loss     nats/char on held-out Pocket sentences (did it forget the language?)

Variants: the base model; SFT; SFT with replay of pretraining batches; replay alone (a
control: is the gain from the conversations or from more training?); and SFT of a randomly
initialised model (is the knowledge coming from pretraining or from the conversations?).
Predict before running: which variants can answer in chat format, and which still
complete "47+38="?
"""

import argparse
import copy

import torch

from mlle.finetune.base import get_base_model, new_model, pocket_loss, pretraining_batch_fn
from mlle.finetune.data import conversations
from mlle.finetune.sft import build_sft_rows, chat_accuracy, raw_completion_accuracy, sft_train
from mlle.utils import REPO_ROOT, JsonlLogger

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--n-convs", type=int, default=3000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--n-eval", type=int, default=300)
args = ap.parse_args()

base, world = get_base_model()
tok, task = world.tok, world.task

# %% 1. What the model is trained on
example = conversations(task, 1, "train", 3)[0]
ids, mask = tok.render_conversation(example)
print("\n1. one conversation as the model sees it (mask=1: the assistant must learn to produce this token)")
print(tok.visualize(ids, mask))

convs = conversations(task, args.n_convs, "train", 7)
x, y = build_sft_rows(tok, convs, world.spec.block_size)
supervised = (y >= 0).float().mean().item()
print(f"\n   {len(convs)} conversations packed into {x.shape[0]} rows of {x.shape[1]} tokens; "
      f"{supervised:.0%} of positions are supervised, the rest are context or padding")

# %% 2. Variants
seen, new = conversations(task, args.n_eval, "train", 1), conversations(task, args.n_eval, "test", 1)
new_problems = task.problems(args.n_eval, "test", 1)
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part4" / "14_chat_sft.jsonl")


def report(name, model):
    row = dict(variant=name, chat_seen=chat_accuracy(model, tok, seen), chat_new=chat_accuracy(model, tok, new),
               raw_new=raw_completion_accuracy(model, tok, new_problems), pocket_loss=pocket_loss(model, world))
    print(f"{name:44s} {row['chat_seen']:>11.3f} {row['chat_new']:>10.3f} {row['raw_new']:>9.3f} {row['pocket_loss']:>12.4f}", flush=True)
    log.log(steps=args.steps, n_convs=args.n_convs, lr=args.lr, **row)


print(f"\n2. {args.steps} fine-tuning steps, learning rate {args.lr}, {args.n_eval} evaluation questions per column")
print(f"{'variant':44s} {'chat (seen)':>11} {'chat (new)':>10} {'raw (new)':>9} {'Pocket loss':>12}")
report("base model (no fine-tuning)", base)

m = copy.deepcopy(base)
sft_train(m, x, y, steps=args.steps, lr=args.lr)
report("SFT on conversations only", m)

m = copy.deepcopy(base)
sft_train(m, x, y, steps=args.steps, lr=args.lr, extra_batch_fn=pretraining_batch_fn(world))
report("SFT + replay of pretraining batches", m)

m = copy.deepcopy(base)
replay = pretraining_batch_fn(world, seed=1)
rows = [replay() for _ in range(64)]  # 2 048 rows of pretraining text, used as the "fine-tuning set"
sft_train(m, torch.cat([r[0] for r in rows]), torch.cat([r[1] for r in rows]), steps=args.steps, lr=args.lr)
report("control: more pretraining, no conversations", m)

m = new_model(world, seed=1)
sft_train(m, x, y, steps=args.steps, lr=args.lr)
report("control: SFT from random initialisation", m)

print("\nReading guide: 'chat (new)' can only be right if the model computes the sum; with a random")
print("start the conversations are the only source of that skill. Compare rows 2 and 3 for forgetting,")
print("rows 3 and 4 for what the conversations added, rows 2 and 5 for what pretraining contributed.")
