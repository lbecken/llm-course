"""
Lab 12c — How many bits does a weight need?   Tier A (about 1 minute, CPU)

    python labs/optimization/12_quantize.py

Trains a MiniGPT, then rounds its weight matrices to 8, 6, 4, 3 and 2 bits (per tensor
and per output channel) and measures size, weight error, validation loss and agreement.
Success criterion: 8-bit quantization changes the loss by less than 0.002 nats; you can
identify the bit-width at which quality collapses, and the benefit of per-channel scales.
"""

import copy

import torch

from mlle.data import pocket
from mlle.foundations.experiments import agreement_accuracy
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import TrainConfig, estimate_loss, split_tokens, train_on_text
from mlle.inference.quantize import quantize_model_

text = pocket.generate_corpus(20000, seed=0)
model, tok, res = train_on_text(text, dict(block_size=64, n_layer=2, n_head=2, n_embd=64),
                                TrainConfig(steps=1500, batch_size=32, lr=1e-2, eval_every=1500), verbose=False)
_, val_tokens = split_tokens(torch.tensor(CharTokenizer(text).encode(text)))
items = pocket.agreement_items(200, with_attractor=True)


def evaluate(m):
    return estimate_loss(m, val_tokens, 32, 20), agreement_accuracy(m, tok, items)


base_loss, base_acc = evaluate(model)
n = sum(p.numel() for p in model.parameters() if p.ndim == 2)
print(f"{'weights':>22} {'bytes':>10} {'rel. weight error':>18} {'val loss':>9} {'delta':>8} {'agreement':>10}")
print(f"{'fp32 (trained)':>22} {n * 4:>10,} {0.0:>18.4f} {base_loss:>9.4f} {0.0:>8.4f} {base_acc:>10.2f}")
for bits in (8, 6, 4, 3, 2):
    for per_channel in (False, True):
        m = copy.deepcopy(model)
        stats = quantize_model_(m, bits=bits, per_channel=per_channel)
        loss, acc = evaluate(m)
        name = f"int{bits} {'per-channel' if per_channel else 'per-tensor'}"
        print(f"{name:>22} {int(stats['bytes_quantized']):>10,} {stats['relative_error']:>18.4f} {loss:>9.4f} {loss - base_loss:>+8.4f} {acc:>10.2f}")
