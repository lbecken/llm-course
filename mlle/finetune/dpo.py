"""
Direct Preference Optimization (Module 15). A course extension; not in nanochat.

Data: triples (prompt x, preferred answer y_w, rejected answer y_l). No reward model and
no sampling during training. With a frozen copy of the starting model as reference,

    margin = [log pi(y_w | x) - log pi_ref(y_w | x)]  -  [log pi(y_l | x) - log pi_ref(y_l | x)]
    loss   = - log sigmoid(beta * margin)

The policy is pushed to raise the preferred answer's probability RELATIVE to where it
started and lower the rejected one's; the reference terms keep it from drifting
arbitrarily far, with beta controlling how strongly (Rafailov et al., 2023).
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from mlle.finetune.chat import ChatTokenizer


def sequence_logprob(model, prompt: list[int], answer: list[int]) -> torch.Tensor:
    """log pi(answer | prompt): sum of the log-probabilities of the answer's tokens."""
    ids = torch.tensor([prompt + answer], dtype=torch.long)
    targets = ids[:, 1:].clone()
    targets[:, : len(prompt) - 1] = -1  # only the answer tokens count
    return -model(ids[:, :-1], targets, loss_reduction="sum")


def dpo_loss(model, reference, prompt, preferred, rejected, beta: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        ref_w, ref_l = sequence_logprob(reference, prompt, preferred), sequence_logprob(reference, prompt, rejected)
    pol_w, pol_l = sequence_logprob(model, prompt, preferred), sequence_logprob(model, prompt, rejected)
    return -F.logsigmoid(beta * ((pol_w - ref_w) - (pol_l - ref_l)))


def make_preference_pairs(tok: ChatTokenizer, convs: list[dict], seed: int = 0) -> list[tuple]:
    """(prompt, preferred, rejected): the correct answer versus a plausible wrong one (off by 1, 2 or 10)."""
    rng = random.Random(seed)
    end = tok.encode_special("<|assistant_end|>")
    pairs = []
    for conv in convs:
        truth = int(conv["problem"].answer)
        wrong = max(0, truth + rng.choice([-10, -2, -1, 1, 2, 10]))
        if wrong == truth:
            wrong = truth + 1
        pairs.append((tok.render_for_completion(conv), tok.encode(str(truth)) + [end], tok.encode(str(wrong)) + [end]))
    return pairs


def dpo_train(model, pairs: list[tuple], steps: int = 200, batch_size: int = 16, lr: float = 1e-3, beta: float = 1.0, seed: int = 0):
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(seed)
    losses = []
    model.train()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * (1.0 - step / steps)
        loss = torch.stack([dpo_loss(model, reference, *rng.choice(pairs), beta=beta) for _ in range(batch_size)]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses
