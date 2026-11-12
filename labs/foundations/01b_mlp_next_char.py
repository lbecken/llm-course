"""
Lab 01b — The same MLP, now predicting text.   Tier A (about 20 s, CPU)

    python labs/foundations/01b_mlp_next_char.py
    python labs/foundations/01b_mlp_next_char.py --context 6

The digit classifier maps 35 pixels to 10 classes. Here the SAME class (ManualMLP, hand-
written backprop) maps the previous `--context` characters, one-hot encoded and
concatenated, to V classes: the next character. Nothing about the network or the
training loop changes. Only the meaning of the input and of the classes does.
Success criterion: the loss falls from ln(V) to roughly the cross-entropy of a count-based
model with the same context length, and stays above the entropy of the source.
"""

import argparse
import math

import torch

from mlle.data import pocket
from mlle.foundations.mlp import ManualMLP
from mlle.foundations.ngram import CharTokenizer, NGramLM

ap = argparse.ArgumentParser()
ap.add_argument("--context", type=int, default=3)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.5)
args = ap.parse_args()

text = pocket.generate_corpus(8000, seed=0)
tok = CharTokenizer(text)
ids = torch.tensor(tok.encode(text))
V, K = tok.vocab_size, args.context

# Build (context, next char) pairs: x is K one-hot vectors side by side, y is a class index.
windows = ids.unfold(0, K + 1, 1)  # (N, K+1): every run of K+1 consecutive characters
x = torch.nn.functional.one_hot(windows[:, :K], V).float().view(-1, K * V)  # (N, K*V)
y = windows[:, K]  # (N,)
n_train = int(0.9 * len(x))
x_train, y_train, x_val, y_val = x[:n_train], y[:n_train], x[n_train:], y[n_train:]
print(f"{len(x):,} examples | input = {K} characters x {V} one-hot = {K * V} numbers | classes = {V} possible next characters")

model = ManualMLP(K * V, args.hidden, V, seed=0)
g = torch.Generator().manual_seed(0)
for step in range(args.steps + 1):
    if step % (args.steps // 6) == 0:
        print(f"step {step:5d} | train loss {model.loss(x_train[:5000], y_train[:5000]).item():.4f} | val loss {model.loss(x_val[:5000], y_val[:5000]).item():.4f}")
    idx = torch.randint(0, n_train, (128,), generator=g)
    model.forward(x_train[idx])
    model.backward(y_train[idx])
    model.sgd_step(args.lr)

count_model = NGramLM(K, V).fit(ids[:n_train].tolist())
print(f"\nuniform guess ln({V}) = {math.log(V):.4f}")
print(f"count-based model with the same {K}-character context: {count_model.cross_entropy(ids[n_train:n_train + 5000].tolist()):.4f}")
print(f"entropy of the source (unbeatable): {pocket.entropy_rate()['nats_per_char']:.4f}")

# Generate: classification, sample, slide the window, repeat.
context = tok.encode("\nthe ")[-K:] if K <= 5 else tok.encode("\nthe cat sees ")[-K:]
out = list(context)
for _ in range(80):
    xin = torch.nn.functional.one_hot(torch.tensor(out[-K:]), V).float().view(1, -1)
    probs = torch.softmax(model.forward(xin)[0], dim=-1)
    out.append(torch.multinomial(probs, 1, generator=g).item())
print("\nsample:", tok.decode(out).replace("\n", " / "))
