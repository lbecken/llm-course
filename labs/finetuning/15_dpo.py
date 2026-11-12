"""
Lab 15b — Learning from preferences: DPO, and likelihood displacement.
Tier A (about 5 minutes on a 4-core CPU; uses the cached base model)

    python labs/finetuning/15_dpo.py

Part 1: a STYLE preference. The starting model answers in two styles, "85" and
"it is 85" (it was fine-tuned on a 50/50 mixture). The preference data say: for the same
correct answer, the long style is better. No reward function, no sampling: only pairs
(prompt, preferred, rejected). Four DPO configurations are compared on

    long style     fraction of sampled answers in the preferred style   (what was optimised)
    accuracy       exact answers, greedy                                 (what was NOT optimised)
    log p          mean log-probability of the preferred and of the rejected answer

Part 2: a CORRECTNESS preference with near-miss negatives (the right sum versus a sum
that is off by 1, 2 or 10).

Predict before running: the DPO loss only looks at the DIFFERENCE between the two
log-probabilities. Name two different ways a model can increase that difference.
DPO is a course extension; nanochat does not implement it.
"""

import argparse
import copy

import torch

from mlle.finetune.base import get_base_model, get_chat_model, pocket_loss
from mlle.finetune.data import LONG_PREFIX, conversations, extract_answer
from mlle.finetune.dpo import dpo_train, make_preference_pairs, make_style_pairs, mean_logprobs
from mlle.finetune.sft import chat_accuracy, complete
from mlle.utils import REPO_ROOT, JsonlLogger

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--n-pairs", type=int, default=300)
args = ap.parse_args()

base, world = get_base_model()
tok, task = world.tok, world.task
held_out = conversations(task, 200, "test", 1)
pair_convs = conversations(task, args.n_pairs, "train", 11)
stop = {tok.encode_special("<|assistant_end|>"), tok.get_bos_token_id()}
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part4" / "15_dpo.jsonl")


def long_style_rate(model, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    texts = [tok.decode(o) for c in held_out for o in complete(model, tok.render_for_completion(c), stop, 14, temperature=1.0, num_samples=4, generator=g)]
    return sum(t.startswith(LONG_PREFIX) for t in texts) / len(texts)


def row(name, model, pairs, start=None, **extra):
    lw, ll = mean_logprobs(model, pairs)
    r = dict(variant=name, long_style=long_style_rate(model), accuracy=chat_accuracy(model, tok, held_out, max_new_tokens=14),
             logp_preferred=lw, logp_rejected=ll, pocket=pocket_loss(model, world))
    print(f"{name:46s} {r['long_style']:>11.3f} {r['accuracy']:>9.3f} {lw:>10.2f} {ll:>10.2f} {r['pocket']:>8.4f}", flush=True)
    log.log(**r, **extra)


# %% Part 1: style
start = get_chat_model(base, world, steps=600, lr=2e-3, long_fraction=0.5)  # answers in both styles
pairs = make_style_pairs(tok, pair_convs, prefer="long")
print(f"\n1. style preference: {len(pairs)} pairs (preferred 'it is N', rejected 'N'), {args.steps} DPO steps")
print(f"{'variant':46s} {'long style':>11} {'accuracy':>9} {'log p pref':>10} {'log p rej':>10} {'Pocket':>8}")
row("start (SFT on a 50/50 mixture of styles)", start, pairs)
for name, kw in [
    ("DPO beta=1, lr 3e-4", dict(beta=1.0, lr=3e-4)),
    ("DPO beta=5, lr 1e-4 (stay closer to the start)", dict(beta=5.0, lr=1e-4)),
    ("DPO beta=1, lr 3e-4 + 0.3 x SFT on preferred", dict(beta=1.0, lr=3e-4, sft_weight=0.3)),
]:
    model = copy.deepcopy(start)
    dpo_train(model, pairs, steps=args.steps, **kw)
    row(name, model, pairs, **kw)

# %% Part 2: correctness with near-miss negatives
start2 = get_chat_model(base, world)  # the under-trained short-style model of lab 15a
pairs2 = make_preference_pairs(tok, pair_convs)
print(f"\n2. correctness preference: preferred = right sum, rejected = a near miss; {args.steps} DPO steps")
print(f"{'variant':46s} {'long style':>11} {'accuracy':>9} {'log p pref':>10} {'log p rej':>10} {'Pocket':>8}")
row("start (under-trained chat model)", start2, pairs2)
for name, kw in [("DPO beta=1, lr 3e-4", dict(beta=1.0, lr=3e-4)), ("DPO beta=1, lr 3e-4 + 0.3 x SFT on preferred", dict(beta=1.0, lr=3e-4, sft_weight=0.3))]:
    model = copy.deepcopy(start2)
    dpo_train(model, pairs2, steps=args.steps, **kw)
    row(name, model, pairs2, **kw)
print("\nIf 'log p pref' FELL while the loss was being minimised, you have seen likelihood displacement:")
print("the rejected answer was pushed down, and the preferred answer, which resembles it, went down with it.")
