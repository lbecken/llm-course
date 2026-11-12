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


def dpo_loss(model, reference, prompt, preferred, rejected, beta: float = 1.0, sft_weight: float = 0.0) -> torch.Tensor:
    """The DPO loss for one triple. `sft_weight` > 0 adds an anchor, -sft_weight * log pi(preferred):
    the plain DPO loss constrains only the DIFFERENCE between the two answers, so both
    log-probabilities are free to fall together ("likelihood displacement"); the anchor
    forbids that for the preferred one."""
    with torch.no_grad():
        ref_w, ref_l = sequence_logprob(reference, prompt, preferred), sequence_logprob(reference, prompt, rejected)
    pol_w, pol_l = sequence_logprob(model, prompt, preferred), sequence_logprob(model, prompt, rejected)
    return -F.logsigmoid(beta * ((pol_w - ref_w) - (pol_l - ref_l))) - sft_weight * pol_w


@torch.no_grad()
def mean_logprobs(model, pairs: list[tuple], n: int = 100) -> tuple[float, float]:
    """(mean log pi(preferred), mean log pi(rejected)) over the first n triples."""
    lw = sum(sequence_logprob(model, p, yw).item() for p, yw, _ in pairs[:n]) / min(n, len(pairs))
    ll = sum(sequence_logprob(model, p, yl).item() for p, _, yl in pairs[:n]) / min(n, len(pairs))
    return lw, ll


def make_style_pairs(tok: ChatTokenizer, convs: list[dict], prefer: str = "long") -> list[tuple]:
    """(prompt, preferred, rejected) expressing a STYLE preference: the same correct answer
    in the long form ("it is 85") versus the short form ("85")."""
    from mlle.finetune.data import LONG_PREFIX

    end = tok.encode_special("<|assistant_end|>")
    pairs = []
    for conv in convs:
        short = tok.encode(conv["problem"].answer) + [end]
        long = tok.encode(LONG_PREFIX + conv["problem"].answer) + [end]
        pairs.append((tok.render_for_completion(conv),) + ((long, short) if prefer == "long" else (short, long)))
    return pairs


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


def dpo_train(model, pairs: list[tuple], steps: int = 200, batch_size: int = 16, lr: float = 1e-3, beta: float = 1.0,
              sft_weight: float = 0.0, seed: int = 0):
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
        loss = torch.stack([dpo_loss(model, reference, *rng.choice(pairs), beta=beta, sft_weight=sft_weight) for _ in range(batch_size)]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses
