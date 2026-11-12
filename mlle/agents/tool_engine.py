"""
Tool use inside the generation loop of a tiny trained model (Module 16).

This is nanochat's mechanism (nanochat/engine.py, Engine.generate) rebuilt for MiniGPT:

    while generating, watch the sampled tokens:
      <|python_start|>   start collecting an expression
      <|python_end|>     stop collecting; EVALUATE the expression with the calculator;
                         FORCE the tokens  <|output_start|> result <|output_end|>
                         into the sequence, as if the model had produced them
      <|assistant_end|>  stop

The model's learned contribution is to emit the request and to use the result. The
arithmetic is done by ordinary code. `mask` records which tokens were sampled (1) and
which were forced (0), exactly like nanochat's token masks.
"""

from __future__ import annotations

from collections import deque

import torch

from mlle.agents.tools import safe_calculate
from mlle.finetune.chat import ChatTokenizer
from mlle.inference.kvcache import KVCache


@torch.no_grad()
def generate_with_tools(model, tok: ChatTokenizer, prompt_ids: list[int], max_new_tokens: int = 40, temperature: float = 0.0,
                        tools_enabled: bool = True, generator=None):
    """Returns (tokens, mask): the generated tokens after the prompt and, for each, 1 if the
    model sampled it or 0 if the engine forced it (tool output)."""
    sp = tok.special
    device = next(model.parameters()).device
    cache = KVCache.for_model(model, 1, max_len=min(model.config.block_size, len(prompt_ids) + max_new_tokens))
    logits = model(torch.tensor([prompt_ids], device=device), cache=cache)[:, -1, :]
    forced: deque[int] = deque()
    out, mask, expr = [], [], None
    for _ in range(max_new_tokens):
        if forced:
            token, sampled = forced.popleft(), 0
        elif temperature == 0.0:
            token, sampled = logits.argmax(dim=-1).item(), 1
        else:
            token, sampled = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator).item(), 1
        if token in (sp["<|assistant_end|>"], sp["<|bos|>"]):
            break
        out.append(token)
        mask.append(sampled)
        if token == sp["<|python_start|>"]:
            expr = []
        elif token == sp["<|python_end|>"] and expr is not None:
            if tools_enabled:
                try:
                    value = safe_calculate(tok.decode(expr))
                    result = str(int(value)) if float(value).is_integer() else str(value)
                    forced.extend([sp["<|output_start|>"]] + tok.encode(result) + [sp["<|output_end|>"]])
                except (ValueError, KeyError):
                    pass  # an invalid request gets no output; the model continues on its own
            expr = None
        elif expr is not None:
            expr.append(token)
        if cache.pos >= cache.max_len:
            break
        logits = model(torch.tensor([[token]], device=device), cache=cache)[:, -1, :]
    return out, mask


def final_answer(tok: ChatTokenizer, tokens: list[int]) -> str:
    """The assistant's visible answer: the text after the last tool output (or all of it)."""
    end = tok.special["<|output_end|>"]
    if end in tokens:
        tokens = tokens[len(tokens) - tokens[::-1].index(end) :]
    return "".join(tok.itos[t] for t in tokens if t < len(tok.chars))


def tool_chat_accuracy(model, tok: ChatTokenizer, convs: list[dict], tools_enabled: bool = True) -> dict:
    """Exact-match accuracy, and how often the model requested the tool."""
    correct = called = 0
    for conv in convs:
        tokens, _ = generate_with_tools(model, tok, tok.render_for_completion(conv), tools_enabled=tools_enabled)
        called += tok.special["<|python_start|>"] in tokens
        correct += final_answer(tok, tokens) == conv["problem"].answer
    return {"accuracy": correct / len(convs), "tool_call_rate": called / len(convs)}
