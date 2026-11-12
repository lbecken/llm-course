"""
A two-skill toy world for Parts IV-V.

Pretraining text ("documents", one per line) mixes two sources:
    * Pocket sentences          (language)
    * arithmetic facts          "47+38=85"   (knowledge, stated as plain text)

Fine-tuning then teaches a BEHAVIOUR on top: answer a user's question in the chat format,

    user: 47+38        assistant: 85

optionally through a calculator tool,

    assistant: <|python_start|>47+38<|python_end|><|output_start|>85<|output_end|>85

Because every answer can be checked exactly, fine-tuning, reinforcement learning, tool
use and reasoning strategies can all be evaluated without a human in the loop.
"""

from __future__ import annotations

import random

import torch

from mlle.data import pocket
from mlle.data.tasks import AdditionProblem, AdditionTask
from mlle.finetune.chat import ChatTokenizer

ALPHABET = "abcdefghijklmnopqrstuvwxyz .\n0123456789+-*=()'"


def make_tokenizer() -> ChatTokenizer:
    return ChatTokenizer(ALPHABET)


def base_documents(n_sentences: int, n_facts: int, task: AdditionTask, seed: int = 0) -> list[str]:
    """Pretraining documents: Pocket sentences and addition facts from the TRAIN split, shuffled."""
    rng = random.Random(seed)
    docs = [pocket.sample_sentence(rng).text.strip() for _ in range(n_sentences)]
    docs += [p.question + p.answer for p in task.problems(n_facts, "train", seed)]
    rng.shuffle(docs)
    return docs


def documents_to_stream(tok: ChatTokenizer, docs: list[str]) -> torch.Tensor:
    """<|bos|> doc <|bos|> doc ...  as one token stream (the 'stream' packing of Module 08)."""
    bos = tok.get_bos_token_id()
    ids: list[int] = []
    for d in docs:
        ids.append(bos)
        ids.extend(tok.encode(d))
    return torch.tensor(ids, dtype=torch.long)


LONG_PREFIX = "it is "  # the "long" answer style: "it is 85" instead of "85"


def addition_conversation(p: AdditionProblem, use_tool: bool = False, style: str = "short") -> dict:
    question = p.question[:-1]  # "47+38" without the "="
    final = p.answer if style == "short" else LONG_PREFIX + p.answer
    if not use_tool:
        answer = final
    else:
        answer = [
            {"type": "python", "text": question},
            {"type": "python_output", "text": p.answer},
            {"type": "text", "text": final},
        ]
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "problem": p}


def conversations(task: AdditionTask, n: int, split: str, seed: int = 0, use_tool: bool = False, long_fraction: float = 0.0,
                  ops: tuple[str, ...] = ("+",)) -> list[dict]:
    """n conversations from the given split. `long_fraction` of them use the long answer
    style; with several `ops` the operations alternate."""
    rng = random.Random(f"style-{split}-{seed}")
    return [addition_conversation(p, use_tool, "long" if rng.random() < long_fraction else "short")
            for p in task.problems(n, split, seed, ops)]


def extract_answer(text: str) -> str:
    """The answer proper, whatever the style: the last space-separated piece of the reply."""
    return text.split(" ")[-1]
