"""Reusable experiment helpers for Part I labs (kept out of the model files for clarity)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mlle.data.tasks import LookupSpec, lookup_batch
from mlle.foundations.attention import attention


# ----------------------------------------------------------------------------
# Agreement test on the Pocket language (Modules 02, 05, 06, 07)


@torch.no_grad()
def continuation_logprob(model, tok, context: str, continuation: str) -> float:
    """log P(continuation | context) under a character-level MiniGPT: the sum of the
    log-probabilities of each character of the continuation, each given everything before it."""
    ids = torch.tensor([tok.encode(context + continuation)])
    logp = torch.log_softmax(model(ids[:, :-1]), dim=-1)  # predictions for positions 1..n
    n = len(continuation)
    targets = ids[0, -n:]
    return logp[0, -n:].gather(1, targets[:, None]).sum().item()


def agreement_accuracy(model, tok, items) -> float:
    """Fraction of minimal pairs where the model prefers the verb with the correct number.
    Each item is (prefix, correct_verb, wrong_verb); see mlle.data.pocket.agreement_items."""
    model.eval()
    correct = 0
    for prefix, good, bad in items:
        context = "\n" + prefix  # sentences in the corpus start after a newline
        lp_good = continuation_logprob(model, tok, context, good + " ")
        lp_bad = continuation_logprob(model, tok, context, bad + " ")
        correct += 1.0 if lp_good > lp_bad else (0.5 if lp_good == lp_bad else 0.0)
    return correct / len(items)


def ngram_agreement_accuracy(lm, tok, items) -> float:
    """Same test for a count-based n-gram model."""
    correct = 0
    for prefix, good, bad in items:
        scores = []
        for verb in (good, bad):
            ids = tok.encode("\n" + prefix + verb + " ")
            n = len(verb) + 1
            total = 0.0
            for t in range(len(ids) - n, len(ids)):
                total += math.log(lm.probs(ids[:t])[ids[t]].item())
            scores.append(total)
        correct += 1.0 if scores[0] > scores[1] else (0.5 if scores[0] == scores[1] else 0.0)  # ties count half
    return correct / len(items)


# ----------------------------------------------------------------------------
# One attention head as a soft dictionary (Module 05)


class LookupAttention(nn.Module):
    """The smallest model in which attention does all the work.

        tokens --embed--> x (B, T, C)
        q = W_q x[query position]      k = W_k x[pair positions]      v = W_v x[pair positions]
        weights = softmax(q . k / sqrt(C))        over the pairs
        logits  = W_out (weights @ v)             over the possible values

    No MLP, no residual connection, no positions, a single head and a single layer."""

    def __init__(self, spec: LookupSpec, n_embd: int = 32):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, n_embd)
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_out = nn.Linear(n_embd, spec.num_values, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)  # (B, P+1, C)
        pairs, query = x[:, :-1], x[:, -1:]  # (B, P, C), (B, 1, C)
        out, weights = attention(self.W_q(query), self.W_k(pairs), self.W_v(pairs))  # (B, 1, C), (B, 1, P)
        return self.W_out(out[:, 0]), weights[:, 0]  # logits (B, num_values), weights (B, P)


def train_lookup(spec: LookupSpec | None = None, steps: int = 1500, batch_size: int = 128, lr: float = 3e-3,
                 n_embd: int = 32, seed: int = 0, log_every: int = 250, verbose: bool = True):
    """Train LookupAttention. Returns (model, history) with accuracy and the average attention
    weight placed on the pair that holds the answer."""
    spec = spec or LookupSpec()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = LookupAttention(spec, n_embd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps + 1):
        tokens, labels, target_pos = lookup_batch(spec, batch_size, g)
        logits, weights = model(tokens)
        loss = nn.functional.cross_entropy(logits, labels)
        if step % log_every == 0:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            on_target = weights[torch.arange(batch_size), target_pos].mean().item()
            history.append({"step": step, "loss": loss.item(), "accuracy": acc, "attention_on_answer": on_target})
            if verbose:
                print(f"step {step:5d} | loss {loss.item():.4f} | accuracy {acc:.3f} | attention on the answer pair {on_target:.3f}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, history
