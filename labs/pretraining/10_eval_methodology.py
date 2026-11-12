"""
Lab 10b — How sure are you? Seeds, sample sizes and paired comparisons.
Tier A (about 2 minutes, CPU)

    python labs/pretraining/10_eval_methodology.py

1. Seed variance: the same configuration trained with 5 seeds.
2. Evaluation-set size: the standard error of an accuracy, and a bootstrap interval.
3. Two models, same test items: unpaired intervals versus a paired comparison.
4. From nats per token to bits per byte.
Success criterion: for each of two comparisons in the output you can say whether the
difference is supported by the data, and why pairing changes the answer.
"""

import math
import random

import torch

from mlle import evaluation as ev
from mlle.data import pocket
from mlle.foundations.experiments import continuation_logprob
from mlle.foundations.train import TrainConfig, train_on_text

text = pocket.generate_corpus(10000, seed=0)
items = pocket.agreement_items(300, with_attractor=True)


def train_model(seed: int, steps: int, n_embd: int = 64):
    return train_on_text(text, dict(block_size=64, n_layer=2, n_head=2, n_embd=n_embd),
                         TrainConfig(steps=steps, batch_size=32, lr=1e-2, eval_every=steps, seed=seed), verbose=False)


def per_item_correct(model, tok) -> list[float]:
    model.eval()
    out = []
    for prefix, good, bad in items:
        ctx = "\n" + prefix
        out.append(float(continuation_logprob(model, tok, ctx, good + " ") > continuation_logprob(model, tok, ctx, bad + " ")))
    return out


# %% 1. Seed variance
print("1. same configuration, five seeds (400 steps)")
runs = []
for seed in range(5):
    model, tok, res = train_model(seed, 400)
    correct = per_item_correct(model, tok)
    runs.append((res.final_val_loss, sum(correct) / len(correct), correct))
    print(f"   seed {seed}: val loss {res.final_val_loss:.4f}   agreement+attractor {runs[-1][1]:.3f}")
m_loss, s_loss = ev.mean_std([r[0] for r in runs])
m_acc, s_acc = ev.mean_std([r[1] for r in runs])
print(f"   val loss  {m_loss:.4f} +- {s_loss:.4f}   (std across seeds)")
print(f"   agreement {m_acc:.3f} +- {s_acc:.3f}   <- the behavioural metric varies far more than the loss")

# %% 2. Evaluation-set size
acc, correct = min(((r[1], r[2]) for r in runs), key=lambda t: abs(t[0] - 0.6))  # the run closest to 60 %
print(f"\n2. one of these models scored {acc:.3f} on {len(correct)} items. How precise is that number?")
for n in (30, 100, 300, 3000):
    print(f"   if measured on {n:>4} items: standard error {ev.accuracy_stderr(acc, n):.3f}  -> 95% interval about +-{1.96 * ev.accuracy_stderr(acc, n):.3f}")
mean, lo, hi = ev.bootstrap_ci(correct)
print(f"   bootstrap 95% interval on the actual 300 items: [{lo:.3f}, {hi:.3f}]")

# %% 3. Paired versus unpaired
print("\n3. model A (600 steps) versus model B (450 steps), same seed, scored on the same 300 fresh sentences")
model_a, tok, res_a = train_model(0, 600)
model_b, _, res_b = train_model(0, 450)
rng = random.Random(4242)
sentences = [pocket.sample_sentence(rng).text for _ in range(300)]


def per_sentence_nll(model) -> list[float]:
    """Negative log-probability (nats) the model assigns to each whole sentence."""
    model.eval()
    return [-continuation_logprob(model, tok, "\n", s) for s in sentences]


a, b = per_sentence_nll(model_a), per_sentence_nll(model_b)
ma, la, ha = ev.bootstrap_ci(a)
mb, lb, hb = ev.bootstrap_ci(b)
print(f"   mean nats per sentence   A: {ma:.3f}  [{la:.3f}, {ha:.3f}]      B: {mb:.3f}  [{lb:.3f}, {hb:.3f}]")
print(f"   unpaired 95% intervals overlap: {la <= hb and lb <= ha}   (sentences differ hugely in difficulty: std {ev.mean_std(a)[1]:.2f} nats)")
paired = ev.paired_bootstrap(b, a)  # positive = A has the lower loss
print(f"   paired difference B - A: {paired['mean_diff']:+.4f}  [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]   P(A better) = {paired['p_a_better']:.3f}")
better = sum(x < y for x, y in zip(a, b)) / len(a)
print(f"   A assigns the higher probability on {better:.0%} of the individual sentences")

# %% 4. Units
print(f"\n4. units: model A val loss {res_a.final_val_loss:.4f} nats/char = {res_a.final_val_loss / math.log(2):.4f} bits/char; "
      f"a character is one byte here, so that is also bits per byte (floor {pocket.entropy_rate()['bits_per_char']:.4f})")
total_nats, total_bytes = res_a.final_val_loss * 10000, 10000
print(f"   evaluation.bits_per_byte(sum of token losses, sum of token bytes) = {ev.bits_per_byte(total_nats, total_bytes):.4f}")
