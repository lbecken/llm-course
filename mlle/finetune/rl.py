"""
Policy-gradient fine-tuning with a verifiable reward (Module 15).

This mirrors nanochat's scripts/chat_rl.py, which the source itself describes as "a lot
simpler and more similar to just REINFORCE" than GRPO:

    for each prompt:
        sample K completions from the CURRENT model               (rollouts, no gradient)
        reward r_k = 1 if completion k is correct else 0          (a program checks it)
        advantage a_k = r_k - mean(r)                             (baseline: the group's mean)
    loss = - sum_k a_k * sum_t log pi(token_t of completion k)  / (number of sampled tokens)

Raising the log-probability of better-than-average samples and lowering that of
worse-than-average ones is all there is. No reference answers enter the loss: the only
supervision is the number r_k.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from mlle.finetune.chat import ChatTokenizer
from mlle.finetune.sft import complete


def reward_exact(problem, text: str) -> float:
    """The true objective: 1 if the answer is right."""
    return float(text == problem.answer)


def reward_last_digit(problem, text: str) -> float:
    """A FLAWED proxy: a lazy checker that only looks at the last digit (and at the length).
    Used in Module 15 to demonstrate reward hacking."""
    return float(len(text) == len(problem.answer) and text[-1:] == problem.answer[-1:])


def rollout_batch(model, tok: ChatTokenizer, conv: dict, num_samples: int, reward_fn: Callable, generator, max_new_tokens: int = 8):
    """Sample completions for one prompt. Returns (inputs, targets, rewards):
    inputs/targets are (K, T) with targets = -1 everywhere except the SAMPLED tokens."""
    end = tok.encode_special("<|assistant_end|>")
    prompt = tok.render_for_completion(conv)
    samples = complete(model, prompt, {end, tok.get_bos_token_id()}, max_new_tokens, temperature=1.0,
                       num_samples=num_samples, generator=generator)
    rewards = torch.tensor([reward_fn(conv["problem"], tok.decode(s)) for s in samples])
    seqs = [prompt + s + [end] for s in samples]  # the end token is part of the action sequence
    T = max(len(s) for s in seqs)
    ids = torch.full((num_samples, T), end, dtype=torch.long)
    mask = torch.zeros((num_samples, T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, len(prompt) : len(s)] = 1  # only the tokens the policy chose
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    targets[mask[:, 1:] == 0] = -1
    return inputs, targets, rewards


def reinforce_step(model, optimizer, batches) -> dict:
    """One policy-gradient update from a list of (inputs, targets, rewards) rollout batches."""
    model.train()
    optimizer.zero_grad()
    total_reward, n, informative = 0.0, 0, 0
    for inputs, targets, rewards in batches:
        informative += int(rewards.max() != rewards.min())  # all-equal rewards => zero advantage => no gradient
        advantages = rewards - rewards.mean()  # baseline = mean reward of this prompt's samples
        logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)  # (K, T); 0 where target == -1
        objective = (logp * advantages[:, None]).sum() / (targets >= 0).sum().clamp(min=1)
        (-objective / len(batches)).backward()
        total_reward += rewards.sum().item()
        n += len(rewards)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    return {"mean_reward": total_reward / n, "informative_prompts": informative / len(batches)}


def reinforce_train(model, tok: ChatTokenizer, convs: list[dict], reward_fn: Callable = reward_exact, steps: int = 100,
                    prompts_per_step: int = 8, num_samples: int = 8, lr: float = 1e-3, seed: int = 0,
                    eval_fn: Callable | None = None, eval_every: int = 25, verbose: bool = True) -> list[dict]:
    """Train with REINFORCE. Returns a history of {'step', 'mean_reward', ...eval_fn(model)}."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = []
    for step in range(steps + 1):
        if eval_fn is not None and step % eval_every == 0:
            entry = {"step": step, **eval_fn(model)}
            history.append(entry)
            if verbose:
                print("  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()))
        if step == steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr * (1.0 - step / steps)  # linear decay to zero, as in chat_rl.py
        batches = [rollout_batch(model, tok, rng.choice(convs), num_samples, reward_fn, g) for _ in range(prompts_per_step)]
        stats = reinforce_step(model, opt, batches)
        if history:
            history[-1].setdefault("train_reward", stats["mean_reward"])
            history[-1].setdefault("informative_prompts", stats["informative_prompts"])
    return history
