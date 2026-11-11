"""
A complete, small training loop for MiniGPT (Module 07 and every later experiment).

The loop is the same five lines as for the 1990s digit classifier:

    logits = model(x)        forward
    loss   = CE(logits, y)   how wrong?
    loss.backward()          gradients of the loss w.r.t. every parameter
    optimizer.step()         move parameters a little, downhill
    optimizer.zero_grad()

Everything else here is bookkeeping that makes an experiment trustworthy: a held-out
split, fixed seeds, periodic evaluation, throughput measurement and a structured record
of the configuration and results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch

from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.optim import lr_multiplier
from mlle.utils import seed_everything


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 20
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    grad_clip: float | None = 1.0  # nanochat no longer clips; small models with AdamW benefit
    eval_every: int = 200
    eval_batches: int = 20
    seed: int = 0
    device: str = "cpu"
    optimizer: str = "adamw"  # adamw | muon (Muon for 2-D block matrices, AdamW for the rest)
    muon_lr: float = 0.02


def split_tokens(tokens: torch.Tensor, val_fraction: float = 0.1):
    """Hold out the LAST part of the token stream for validation (never shuffled in)."""
    n_val = int(len(tokens) * val_fraction)
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator, device="cpu"):
    """Random windows from a 1-D token stream.

    x = tokens[i : i + T],  y = tokens[i + 1 : i + T + 1]: the target at every position is
    simply the NEXT token, so one window of T tokens yields T training examples."""
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: MiniGPT, tokens: torch.Tensor, batch_size: int, num_batches: int, seed: int = 1234) -> float:
    """Mean loss over a FIXED set of random windows (same seed every call), so successive
    evaluations are comparable and the curve is not blurred by sampling noise."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(tokens, batch_size, model.config.block_size, g, device)
        losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def build_optimizers(model: MiniGPT, cfg: TrainConfig) -> list:
    """AdamW for everything, or nanochat's split: Muon for the 2-D matrices inside the
    blocks, AdamW for embeddings, the output head, norms and biases."""
    if cfg.optimizer == "adamw":
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return [torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))]
    from mlle.foundations.optim import Muon

    matrix_ids = {id(p) for block in model.h for p in block.parameters() if p.ndim == 2}
    matrices = [p for p in model.parameters() if id(p) in matrix_ids]
    others = [p for p in model.parameters() if id(p) not in matrix_ids]
    adam = torch.optim.AdamW(others, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    muon = Muon(matrices, lr=cfg.muon_lr)
    return [adam, muon]


@dataclass
class TrainResult:
    history: list = field(default_factory=list)  # [{'step', 'train_loss', 'val_loss'}]
    final_val_loss: float = float("nan")
    tokens_per_second: float = 0.0
    seconds: float = 0.0
    num_params: int = 0
    config: dict = field(default_factory=dict)


def train(model: MiniGPT, train_tokens: torch.Tensor, val_tokens: torch.Tensor, cfg: TrainConfig, verbose: bool = True) -> TrainResult:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device).train()
    optimizers = build_optimizers(model, cfg)
    base_lrs = [[getattr(o, "lr", None)] if not hasattr(o, "param_groups") else [g["lr"] for g in o.param_groups] for o in optimizers]
    g = torch.Generator().manual_seed(cfg.seed)
    result = TrainResult(num_params=model.num_params(), config={"model": asdict(model.config), "train": asdict(cfg)})
    block = model.config.block_size
    t_train = 0.0

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            record = {
                "step": step,
                "train_loss": estimate_loss(model, train_tokens, cfg.batch_size, cfg.eval_batches),
                "val_loss": estimate_loss(model, val_tokens, cfg.batch_size, cfg.eval_batches),
            }
            result.history.append(record)
            if verbose:
                print(f"step {step:5d} | train {record['train_loss']:.4f} | val {record['val_loss']:.4f}")
        if step == cfg.steps:
            break

        t0 = time.perf_counter()
        lrm = lr_multiplier(step, cfg.steps, cfg.warmup_steps, cfg.warmdown_ratio, cfg.final_lr_frac)
        for opt, lrs in zip(optimizers, base_lrs):
            if hasattr(opt, "param_groups"):
                for group, lr in zip(opt.param_groups, lrs):
                    group["lr"] = lr * lrm
            else:
                opt.lr = lrs[0] * lrm
        x, y = get_batch(train_tokens, cfg.batch_size, block, g, device)
        loss = model(x, y)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad()
        t_train += time.perf_counter() - t0

    result.final_val_loss = result.history[-1]["val_loss"]
    result.seconds = t_train
    result.tokens_per_second = cfg.steps * cfg.batch_size * block / max(t_train, 1e-9)
    return result


def train_on_text(text: str, model_config_kwargs: dict | None = None, train_config: TrainConfig | None = None,
                  style: str = "nanochat", verbose: bool = True):
    """Convenience wrapper: character-tokenize `text`, build a MiniGPT, train it.
    Returns (model, tokenizer, result)."""
    from mlle.foundations.ngram import CharTokenizer

    tok = CharTokenizer(text)
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    train_tokens, val_tokens = split_tokens(tokens)
    kwargs = dict(block_size=64, n_layer=2, n_head=2, n_embd=64)
    kwargs.update(model_config_kwargs or {})
    make = MiniGPTConfig.nanochat_style if style == "nanochat" else MiniGPTConfig.gpt2_style
    seed_everything((train_config or TrainConfig()).seed)
    model = MiniGPT(make(tok.vocab_size, **kwargs))
    result = train(model, train_tokens, val_tokens, train_config or TrainConfig(), verbose=verbose)
    return model, tok, result
