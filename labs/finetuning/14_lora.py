"""
Lab 14b — LoRA: fine-tuning a few thousand numbers instead of the whole model.
Tier A (about 4 minutes on a 4-core CPU; uses the cached base model of lab 14a)

    python labs/finetuning/14_lora.py

Same task as lab 14a (teach the base model the chat format), but the base weights are
FROZEN. Trainable: low-rank corrections B @ A added to chosen matrices, plus (optionally)
correction vectors for the embedding and output rows of the eight chat tokens, which
pretraining never trained.

Reported per variant: number of trainable parameters; chat accuracy on held-out
questions; Pocket loss with the adapter ON (what the adapted model does to plain text)
and with the adapter OFF (must equal the base model exactly: that is LoRA's guarantee).
Predict before running: can an adapter on the attention matrices alone learn a format
whose tokens the frozen embedding table has never trained?
LoRA is a course extension: nanochat fine-tunes all weights.
"""

import argparse
import copy

import torch

from mlle.finetune.base import get_base_model, pocket_loss
from mlle.finetune.data import conversations
from mlle.finetune.lora import LoRALinear, add_lora, count_trainable, set_lora_enabled
from mlle.finetune.sft import build_sft_rows, chat_accuracy, sft_train
from mlle.utils import REPO_ROOT, JsonlLogger

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0, help="seeds the adapter initialisation and the batch order")
args = ap.parse_args()

base, world = get_base_model()
tok, task = world.tok, world.task
x, y = build_sft_rows(tok, conversations(task, 3000, "train", 7), world.spec.block_size)
new = conversations(task, 300, "test", 1)
chat_tokens = sorted(i for name, i in tok.special.items() if name != "<|bos|>")  # BOS was trained in pretraining
QV, ALL = ("c_q", "c_v"), ("c_q", "c_k", "c_v", "c_proj", "c_fc")
total = base.num_params()
base_pocket = pocket_loss(base, world)
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part4" / "14_lora.jsonl")

print(f"\nbase model: {total:,} parameters, Pocket loss {base_pocket:.4f}; {args.steps} steps, lr {args.lr}, seed {args.seed}")
print(f"{'variant':46s} {'trainable':>10} {'% of model':>10} {'chat (new)':>10} {'Pocket, adapter on':>19} {'adapter off':>12}")

m = copy.deepcopy(base)
sft_train(m, x, y, steps=args.steps, lr=2e-3)
print(f"{'full fine-tuning (lab 14a, no replay)':46s} {total:>10,} {'100.0':>10} {chat_accuracy(m, tok, new):>10.3f} {pocket_loss(m, world):>19.4f} {'n/a':>12}", flush=True)

variants = [
    ("LoRA r=8 on q,v; token rows frozen", 8, QV, None),
    ("LoRA r=8 on q,v + chat-token rows", 8, QV, chat_tokens),
    ("LoRA r=2 on all matrices + chat-token rows", 2, ALL, chat_tokens),
    ("LoRA r=8 on all matrices + chat-token rows", 8, ALL, chat_tokens),
    ("LoRA r=32 on all matrices + chat-token rows", 32, ALL, chat_tokens),
]
for name, r, targets, token_ids in variants:
    m = copy.deepcopy(base)
    torch.manual_seed(args.seed)  # A is initialised randomly: results vary with this seed (try --seed 1, 2)
    params = add_lora(m, r=r, alpha=2 * r, targets=targets, new_token_ids=token_ids)
    sft_train(m, x, y, steps=args.steps, lr=args.lr, params=params, seed=args.seed)
    n, acc, on = count_trainable(m), chat_accuracy(m, tok, new), pocket_loss(m, world)
    set_lora_enabled(m, False)
    off = pocket_loss(m, world)
    set_lora_enabled(m, True)
    print(f"{name:46s} {n:>10,} {100 * n / total:>10.1f} {acc:>10.3f} {on:>19.4f} {off:>12.4f}", flush=True)
    log.log(variant=name, rank=r, seed=args.seed, trainable=n, chat_new=acc, pocket_on=on, pocket_off=off, steps=args.steps, lr=args.lr)
    assert abs(off - base_pocket) < 1e-6, "with the adapter off the model must be the base model"

# %% merging: the adapter can be folded into one matrix for deployment
layer = next(mod for mod in m.modules() if isinstance(mod, LoRALinear))
probe = torch.randn(4, layer.base.in_features)
diff = (layer(probe) - probe @ layer.merged_weight().T).abs().max().item()
print(f"\nmerged weight W + (alpha/r) B A reproduces the adapted layer: max difference {diff:.1e}")
print("adapter off == base model held for every variant (asserted above)")
