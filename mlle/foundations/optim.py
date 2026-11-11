"""
Optimizers written out by hand (Module 07).

    sgd_step          the 1990s baseline: p <- p - lr * grad
    AdamW             per-parameter adaptive steps + decoupled weight decay
    newton_schulz     approximate orthogonalization of a matrix, the heart of Muon
    Muon              momentum, then replace the update matrix by a (semi-)orthogonal one
    lr_multiplier     nanochat's schedule: linear warmup, constant, linear warmdown

These are teaching implementations: correct, tested (tests/test_optim.py), and slow.
nanochat's production versions live in nanochat/optim.py, where AdamW and Muon are single
fused, compiled kernels and Muon uses the "Polar Express" iteration plus several
refinements described in the lesson.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sgd_step(params, lr: float) -> None:
    for p in params:
        if p.grad is not None:
            p -= lr * p.grad


class AdamW:
    """AdamW (Loshchilov & Hutter). For each parameter tensor p with gradient g:

        p <- p * (1 - lr * weight_decay)          decoupled weight decay: shrink towards 0
        m <- beta1 * m + (1 - beta1) * g          running mean of gradients   (direction)
        v <- beta2 * v + (1 - beta2) * g^2        running mean of squares     (scale)
        m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)     undo the zero initialisation
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Dividing by sqrt(v_hat) makes the step size roughly `lr` for every coordinate,
    whatever the raw gradient scale: early on, each coordinate moves by about +-lr."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = [p for p in params]
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.weight_decay)
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately replace G = U S V^T by U V^T (same singular vectors, all singular values ~1).

    Uses only matrix multiplications, so it runs efficiently on a GPU and in low precision.
    Each step applies the odd polynomial  p(X) = aX + b(XX^T)X + c(XX^T)^2 X, which acts on
    every singular value s as  s -> a s + b s^3 + c s^5  while leaving U and V untouched.
    The coefficients (Keller Jordan's Muon) push singular values in (0, 1] towards 1 quickly;
    they land in roughly [0.7, 1.2] rather than exactly at 1, which is good enough for an
    optimizer."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)  # Frobenius norm >= largest singular value, so all s are now <= 1
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:  # work with the wide orientation so that X X^T is the small Gram matrix
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon:
    """Muon = MomentUm Orthogonalized by Newton-schulz, for 2-D weight matrices only.

        buf    <- momentum * buf + (1 - momentum) * grad
        update <- (1 - momentum) * grad + momentum * buf          (Nesterov look-ahead)
        update <- newton_schulz(update)                           all singular values ~1
        W      <- W - lr * sqrt(max(1, rows / cols)) * update

    AdamW equalises the step size per COORDINATE; Muon equalises it per singular
    DIRECTION of the whole matrix, so rarely-used directions of a layer learn as fast as
    dominant ones. Embeddings, the output head and scalars are not "matrices that
    transform vectors" in this sense and stay with AdamW, exactly as in nanochat."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5):
        self.params = [p for p in params]
        assert all(p.ndim == 2 for p in self.params), "Muon is for 2-D matrices"
        self.lr, self.momentum, self.ns_steps = lr, momentum, ns_steps
        self.buf = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, buf in zip(self.params, self.buf):
            if p.grad is None:
                continue
            g = p.grad
            buf.lerp_(g, 1 - self.momentum)
            update = g.lerp(buf, self.momentum)
            update = newton_schulz(update, self.ns_steps)
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            p.add_(update, alpha=-self.lr * scale)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def lr_multiplier(step: int, num_steps: int, warmup_steps: int = 40, warmdown_ratio: float = 0.65,
                  final_lr_frac: float = 0.05) -> float:
    """nanochat's learning-rate schedule (scripts/base_train.py, get_lr_multiplier), same defaults:
    ramp up linearly for `warmup_steps`, hold at 1, then ramp down linearly to `final_lr_frac`
    over the last `warmdown_ratio` of training."""
    warmdown_steps = round(warmdown_ratio * num_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step <= num_steps - warmdown_steps:
        return 1.0
    progress = (num_steps - step) / warmdown_steps
    return progress * 1.0 + (1 - progress) * final_lr_frac
