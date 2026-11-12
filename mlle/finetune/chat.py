"""
Conversations as token sequences (Module 14), mirroring nanochat/tokenizer.py.

`ChatTokenizer` is a character-level tokenizer plus nanochat's nine special tokens.
`render_conversation` turns a conversation into (ids, mask) exactly as nanochat does:

    <|bos|> <|user_start|> ...user text... <|user_end|>
            <|assistant_start|> ...assistant text... <|assistant_end|>

mask[i] = 1 for the tokens the assistant must learn to PRODUCE (its text, its tool
requests, and <|assistant_end|>), 0 for everything it only READS (user text, structure
tokens that the application inserts, tool outputs).

A conversation is a dict {"messages": [{"role": "user"|"assistant", "content": ...}]}.
Assistant content is a string, or a list of parts {"type": "text"|"python"|"python_output", "text": ...}.
"""

from __future__ import annotations

import copy

from mlle.foundations.bpe import SPECIAL_TOKENS


class ChatTokenizer:
    def __init__(self, alphabet: str):
        self.chars = sorted(set(alphabet))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.special = {name: len(self.chars) + i for i, name in enumerate(SPECIAL_TOKENS)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.itos.update({i: name for name, i in self.special.items()})

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.special)

    def encode(self, text: str) -> list[int]:
        """Ordinary text only: special-token strings are not interpreted (see Module 03)."""
        return [self.stoi[ch] for ch in text]

    def encode_special(self, name: str) -> int:
        return self.special[name]

    def get_bos_token_id(self) -> int:
        return self.special["<|bos|>"]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ------------------------------------------------------------------ chat rendering
    def render_conversation(self, conversation: dict, max_tokens: int = 2048, supervise_tool_output: bool = False) -> tuple[list[int], list[int]]:
        """`supervise_tool_output=True` is a deliberate MISTAKE used in Module 16 to show what
        happens when tool results are treated as something the model should produce."""
        out_mask = 1 if supervise_tool_output else 0
        ids: list[int] = []
        mask: list[int] = []

        def add(tokens, m: int):
            tokens = [tokens] if isinstance(tokens, int) else tokens
            ids.extend(tokens)
            mask.extend([m] * len(tokens))

        sp = self.special
        add(sp["<|bos|>"], 0)
        for i, message in enumerate(conversation["messages"]):
            expected = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected, f"message {i} must be from the {expected}"
            content = message["content"]
            if message["role"] == "user":
                add(sp["<|user_start|>"], 0)
                add(self.encode(content), 0)
                add(sp["<|user_end|>"], 0)
            else:
                add(sp["<|assistant_start|>"], 0)
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                for part in parts:
                    tokens = self.encode(part["text"])
                    if part["type"] == "text":
                        add(tokens, 1)
                    elif part["type"] == "python":  # the assistant REQUESTS a tool call: supervised
                        add(sp["<|python_start|>"], 1)
                        add(tokens, 1)
                        add(sp["<|python_end|>"], 1)
                    elif part["type"] == "python_output":  # produced by the tool at run time: not supervised
                        add(sp["<|output_start|>"], out_mask)
                        add(tokens, out_mask)
                        add(sp["<|output_end|>"], out_mask)
                    else:
                        raise ValueError(f"unknown part type {part['type']}")
                add(sp["<|assistant_end|>"], 1)
        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation: dict) -> list[int]:
        """Drop the final assistant message and end with <|assistant_start|>: the prompt
        from which the model must produce its answer (nanochat: render_for_completion)."""
        conversation = copy.deepcopy(conversation)
        assert conversation["messages"][-1]["role"] == "assistant"
        conversation["messages"].pop()
        ids, _ = self.render_conversation(conversation)
        return ids + [self.special["<|assistant_start|>"]]

    def visualize(self, ids, mask) -> str:
        """One token per line with its mask: the quickest way to debug a chat format."""
        return "\n".join(f"{self.itos[int(i)]!r:24s} mask={m}" for i, m in zip(ids, mask))
