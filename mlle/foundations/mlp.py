"""
A two-layer perceptron with hand-written backpropagation (Module 01).

    x (B, D_in) --W1,b1--> h_pre (B, H) --tanh--> h (B, H) --W2,b2--> logits (B, C)
    loss = mean over the batch of  -log softmax(logits)[target]

Nothing here uses autograd. `backward` computes every gradient by the chain rule, and
tests/test_mlp.py checks the result against PyTorch autograd and against finite
differences. The same three ideas (linear map, non-linearity, cross-entropy) are all that
a Transformer's output layer needs; only what happens *before* the last linear map changes.
"""

from __future__ import annotations

import math

import torch


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax, written out. Subtracting the max does not change the result
    (softmax is invariant to adding a constant) but prevents exp() from overflowing."""
    z = logits - logits.max(dim=-1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the target class. logits (B, C), targets (B,)."""
    probs = softmax(logits)
    picked = probs[torch.arange(len(targets)), targets]
    return -torch.log(picked).mean()


class ManualMLP:
    """Parameters are plain tensors; gradients are stored in `self.grads` by `backward`."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # Scale initial weights by 1/sqrt(fan_in) so that pre-activations start with
        # roughly unit variance regardless of layer width (Module 07 returns to this).
        self.W1 = torch.randn(d_in, d_hidden, generator=g) / math.sqrt(d_in)
        self.b1 = torch.zeros(d_hidden)
        self.W2 = torch.randn(d_hidden, d_out, generator=g) / math.sqrt(d_hidden)
        self.b2 = torch.zeros(d_out)
        self.grads: dict[str, torch.Tensor] = {}
        self._cache: dict[str, torch.Tensor] = {}

    def parameters(self) -> dict[str, torch.Tensor]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_pre = x @ self.W1 + self.b1  # (B, H)
        h = torch.tanh(h_pre)  # (B, H)
        logits = h @ self.W2 + self.b2  # (B, C)
        self._cache = {"x": x, "h": h, "logits": logits}
        return logits

    def loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return cross_entropy(self.forward(x), targets)

    def backward(self, targets: torch.Tensor) -> None:
        """Backpropagate d(loss)/d(everything) for the batch last given to `forward`."""
        x, h, logits = self._cache["x"], self._cache["h"], self._cache["logits"]
        B = x.shape[0]

        # 1) loss -> logits.  For softmax + cross-entropy the gradient is simply
        #    (predicted probabilities - one_hot(target)) / B.  Derived in the lesson.
        dlogits = softmax(logits)  # (B, C)
        dlogits[torch.arange(B), targets] -= 1.0
        dlogits /= B

        # 2) logits = h @ W2 + b2
        dW2 = h.T @ dlogits  # (H, C)   each weight sees (its input) x (its output's error)
        db2 = dlogits.sum(dim=0)  # (C,)
        dh = dlogits @ self.W2.T  # (B, H)   error flows back through the same weights

        # 3) h = tanh(h_pre),  d tanh(u)/du = 1 - tanh(u)^2
        dh_pre = dh * (1.0 - h**2)  # (B, H)

        # 4) h_pre = x @ W1 + b1
        dW1 = x.T @ dh_pre  # (D_in, H)
        db1 = dh_pre.sum(dim=0)  # (H,)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def sgd_step(self, lr: float) -> None:
        for name, p in self.parameters().items():
            p -= lr * self.grads[name]

    def accuracy(self, x: torch.Tensor, targets: torch.Tensor) -> float:
        return (self.forward(x).argmax(dim=-1) == targets).float().mean().item()


def finite_difference_grad(model: ManualMLP, x, targets, name: str, index: tuple, eps: float = 1e-3) -> float:
    """Numerical derivative of the loss with respect to one scalar parameter:
    (L(p + eps) - L(p - eps)) / (2 eps).  Slow but assumption-free: the standard way to
    check a hand-written backward pass.  Run it on float64 inputs/parameters for precision."""
    p = model.parameters()[name]
    original = p[index].item()
    p[index] = original + eps
    loss_plus = model.loss(x, targets).item()
    p[index] = original - eps
    loss_minus = model.loss(x, targets).item()
    p[index] = original
    return (loss_plus - loss_minus) / (2 * eps)


def train_digits(
    steps: int = 300,
    d_hidden: int = 32,
    lr: float = 0.5,
    batch_size: int = 64,
    seed: int = 0,
    flip_prob: float = 0.08,
    log_every: int = 50,
    verbose: bool = True,
):
    """Train the manual MLP on the bitmap-digit task. Returns (model, history)."""
    from mlle.data.digits import NUM_CLASSES, NUM_PIXELS, make_digit_dataset

    x_train, y_train = make_digit_dataset(4000, flip_prob=flip_prob, seed=seed)
    x_test, y_test = make_digit_dataset(1000, flip_prob=flip_prob, seed=seed + 1)
    model = ManualMLP(NUM_PIXELS, d_hidden, NUM_CLASSES, seed=seed)
    g = torch.Generator().manual_seed(seed)
    history = []
    for step in range(steps + 1):
        if step % log_every == 0:
            record = {
                "step": step,
                "train_loss": model.loss(x_train, y_train).item(),
                "test_loss": model.loss(x_test, y_test).item(),
                "test_acc": model.accuracy(x_test, y_test),
            }
            history.append(record)
            if verbose:
                print(f"step {step:4d} | train loss {record['train_loss']:.4f} | test loss {record['test_loss']:.4f} | test acc {record['test_acc']:.3f}")
        idx = torch.randint(0, len(x_train), (batch_size,), generator=g)
        model.forward(x_train[idx])
        model.backward(y_train[idx])
        model.sgd_step(lr)
    return model, history
