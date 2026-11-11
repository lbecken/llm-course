"""
Lab 01 — Rebuild the 1990s digit classifier, by hand.   Tier A (seconds, CPU)

    python labs/foundations/01_digits_mlp.py
    python labs/foundations/01_digits_mlp.py --hidden 4       # too few hidden units
    python labs/foundations/01_digits_mlp.py --flip-prob 0.3  # much noisier digits

What it does: shows a few noisy 5x7 digits, checks the hand-written backward pass against
finite differences and autograd, trains with plain SGD, then prints the confusion summary.
Success criterion: gradient check error < 1e-6 and test accuracy > 0.95 with the defaults.
"""

import argparse

import torch
import torch.nn.functional as F

from mlle.data.digits import NUM_CLASSES, NUM_PIXELS, make_digit_dataset, render
from mlle.foundations.mlp import ManualMLP, finite_difference_grad, train_digits

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.5)
ap.add_argument("--flip-prob", type=float, default=0.08)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# %% 1. Look at the data
x, y = make_digit_dataset(3, flip_prob=args.flip_prob, seed=args.seed)
for i in range(3):
    print(f"label {y[i].item()}:\n{render(x[i])}\n")

# %% 2. Gradient check: analytic backward pass vs. numerical derivative vs. autograd
xb, yb = make_digit_dataset(16, flip_prob=args.flip_prob, seed=1)
model = ManualMLP(NUM_PIXELS, 8, NUM_CLASSES, seed=0)
for name, p in model.parameters().items():
    setattr(model, name, p.double())
xb = xb.double()
model.forward(xb)
model.backward(yb)
print("gradient check (analytic vs finite differences):")
for name, index in [("W1", (3, 2)), ("b1", (5,)), ("W2", (1, 7)), ("b2", (0,))]:
    numeric = finite_difference_grad(model, xb, yb, name, index, eps=1e-5)
    analytic = model.grads[name][index].item()
    print(f"  d loss / d {name}{list(index)}: analytic {analytic:+.8f}  numeric {numeric:+.8f}  diff {abs(analytic - numeric):.1e}")
params = {k: v.clone().requires_grad_(True) for k, v in model.parameters().items()}
loss = F.cross_entropy(torch.tanh(xb @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"], yb)
loss.backward()
worst = max((model.grads[k] - params[k].grad).abs().max().item() for k in params)
print(f"  largest difference from PyTorch autograd over ALL parameters: {worst:.1e}\n")

# %% 3. Train with SGD
model, history = train_digits(steps=args.steps, d_hidden=args.hidden, lr=args.lr, seed=args.seed, flip_prob=args.flip_prob)

# %% 4. Where does it still fail?
x_test, y_test = make_digit_dataset(2000, flip_prob=args.flip_prob, seed=99)
pred = model.forward(x_test).argmax(-1)
print(f"\nfinal test accuracy on 2000 fresh digits: {(pred == y_test).float().mean().item():.4f}")
errors = [(int(t), int(p)) for t, p in zip(y_test, pred) if t != p]
if errors:
    from collections import Counter

    print("most common confusions (true -> predicted):", Counter(errors).most_common(5))
print(f"loss of a uniform guess = ln(10) = {torch.log(torch.tensor(10.0)).item():.4f}; initial loss was {history[0]['train_loss']:.4f}")
