"""
A minimal byte-level BPE tokenizer (Module 03).

Algorithm (Sennrich et al. 2016, as used by GPT-2/GPT-4 and nanochat):

1. Start from raw UTF-8 bytes: 256 base tokens, so ANY string can be encoded.
2. Split the text into chunks with a regular expression (words, numbers, punctuation,
   whitespace). Merges never cross chunk boundaries.
3. Repeat until the vocabulary is full: count all adjacent token pairs, merge the most
   frequent pair into a new token id.
4. To encode new text: split into chunks, then within each chunk repeatedly apply the
   learned merge with the lowest rank (the earliest-learned merge that is present).

Special tokens such as <|bos|> are NOT learned. They are appended to the vocabulary and
can only be produced deliberately by code (`encode_special`), never by encoding user
text. That separation is a safety property of chat formats (Module 14, Module 16).

nanochat does the same thing with two fast libraries: `rustbpe` for training and
`tiktoken` for encoding (nanochat/tokenizer.py). The split pattern below is a
simplified, standard-library-`re` version of nanochat's SPLIT_PATTERN.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# Same special tokens, in the same order, as nanochat/tokenizer.py at the pinned revision.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# Chunks: contractions | optional leading space + letters | 1-2 digits | punctuation runs | whitespace.
# nanochat's pattern uses \p{L}/\p{N} (the third-party `regex` module); [^\W\d_] is the
# closest standard-library spelling of "a Unicode letter".
SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+|\d{1,2}| ?[^\s\w]+|\s+(?!\S)|\s+"


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern: str = SPLIT_PATTERN):
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = {}  # (id, id) -> new id, in learned order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, special_tokens: list[str] | None = None, verbose: bool = False):
        special_tokens = SPECIAL_TOKENS if special_tokens is None else special_tokens
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab_size must be at least 256 + number of special tokens"

        # Work on unique chunks with their frequencies: far faster than scanning the raw text.
        chunk_counts = Counter(self._regex.findall(text))
        chunks = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts}

        for i in range(num_merges):
            pair_counts: Counter = Counter()
            for chunk, ids in chunks.items():
                weight = chunk_counts[chunk]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += weight
            if not pair_counts:
                break  # nothing left to merge: every chunk is a single token
            # most frequent pair; ties broken by the smaller pair for determinism
            pair = min(pair_counts, key=lambda p: (-pair_counts[p], p))
            new_id = 256 + i
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            for chunk in chunks:
                chunks[chunk] = _merge(chunks[chunk], pair, new_id)
            if verbose:
                print(f"merge {i + 1:4d}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} -> {new_id} ({pair_counts[pair]} occurrences)")

        offset = 256 + len(self.merges)
        self.special_tokens = {name: offset + j for j, name in enumerate(special_tokens)}
        return self

    # ------------------------------------------------------------------ encoding
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # among the pairs present, apply the one that was learned earliest
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, prepend: str | None = None, append: str | None = None) -> list[int]:
        """Encode ordinary text. Special-token strings inside `text` are NOT recognised:
        "<|bos|>" typed by a user becomes ordinary characters, exactly like
        tiktoken's `encode_ordinary`, which nanochat uses."""
        ids: list[int] = []
        if prepend is not None:
            ids.append(self.encode_special(prepend))
        for chunk in self._regex.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if append is not None:
            ids.append(self.encode_special(append))
        return ids

    def encode_special(self, name: str) -> int:
        return self.special_tokens[name]

    def get_bos_token_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    def token_bytes(self, token_id: int) -> bytes:
        for name, sid in self.special_tokens.items():
            if sid == token_id:
                return name.encode("utf-8")
        return self.vocab[token_id]

    def decode(self, ids) -> str:
        """Concatenate token bytes, then decode UTF-8. A single token can be a fragment of
        a multi-byte character, so invalid sequences are replaced rather than raising."""
        return b"".join(self.token_bytes(int(i)) for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        data = {
            "pattern": self.pattern,
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["pattern"])
        for a, b, new in data["merges"]:
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = data["special_tokens"]
        return tok


def compression_ratio(tokenizer: BPETokenizer, text: str) -> float:
    """UTF-8 bytes per token: higher means each token carries more text."""
    return len(text.encode("utf-8")) / max(1, len(tokenizer.encode(text)))
