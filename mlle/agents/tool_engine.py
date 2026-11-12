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
from typing import Callable

import torch

from mlle.agents.tools import safe_calculate
from mlle.finetune.chat import ChatTokenizer
from mlle.inference.kvcache import KVCache


def calculator(expression: str) -> str:
    """The default tool: the restricted calculator, with integers printed without a decimal point."""
    value = safe_calculate(expression)
    return str(int(value)) if float(value).is_integer() else str(value)


@torch.no_grad()
def generate_with_tools(model, tok: ChatTokenizer, prompt_ids: list[int], max_new_tokens: int = 40, temperature: float = 0.0,
                        tools_enabled: bool = True, generator=None, tool_fn: Callable[[str], str] = calculator):
    """Returns (tokens, mask): the generated tokens after the prompt and, for each, 1 if the
    model sampled it or 0 if the engine forced it (tool output). `tool_fn` maps the
    requested expression to the result text; it may raise ValueError to refuse."""
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
                    result = tool_fn(tok.decode(expr))
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


def tool_chat_accuracy(model, tok: ChatTokenizer, convs: list[dict], tools_enabled: bool = True,
                       tool_fn: Callable[[str], str] = calculator) -> dict:
    """Exact-match accuracy; how often the model requested the tool; how often a tool OUTPUT
    block appears that the engine did not write (the model fabricated a tool result); and how
    often the final answer equals what the tool said (right or wrong)."""
    sp = tok.special
    correct = called = fabricated = follows = 0
    for conv in convs:
        tokens, mask = generate_with_tools(model, tok, tok.render_for_completion(conv), tools_enabled=tools_enabled, tool_fn=tool_fn)
        answer = final_answer(tok, tokens)
        called += sp["<|python_start|>"] in tokens
        fabricated += any(t == sp["<|output_start|>"] and m == 1 for t, m in zip(tokens, mask))
        correct += answer == conv["problem"].answer
        try:
            follows += tools_enabled and answer == tool_fn(conv["problem"].question[:-1])
        except (ValueError, KeyError):
            pass
    n = len(convs)
    return {"accuracy": correct / n, "tool_call_rate": called / n, "fabricated_output_rate": fabricated / n, "follows_tool_rate": follows / n}
