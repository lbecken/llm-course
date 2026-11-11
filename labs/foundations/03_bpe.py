"""
Lab 03 — Train and inspect a byte-level BPE tokenizer.   Tier A (seconds, CPU)

    python labs/foundations/03_bpe.py
    python labs/foundations/03_bpe.py --text-file my_notes.txt --vocab-size 600

Shows the first merges, how text is split into tokens, what happens to unseen words and
non-English text, the trade-off between vocabulary size and sequence length, and why a
user cannot type a special token.
Success criterion: every round trip decode(encode(text)) == text, and bytes/token grows
with vocabulary size on in-domain text.
"""

import argparse
from pathlib import Path

from mlle.data import pocket
from mlle.foundations.bpe import SPECIAL_TOKENS, BPETokenizer, compression_ratio

ap = argparse.ArgumentParser()
ap.add_argument("--text-file", type=str, default=None)
ap.add_argument("--vocab-size", type=int, default=256 + 60 + len(SPECIAL_TOKENS))
args = ap.parse_args()

text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else pocket.generate_corpus(3000, seed=0)


def show(tok: BPETokenizer, s: str) -> str:
    return " | ".join(repr(tok.token_bytes(i).decode("utf-8", errors="replace"))[1:-1] for i in tok.encode(s))


# %% 1. Train and watch the first merges
tok = BPETokenizer().train(text, args.vocab_size)
print(f"trained on {len(text):,} characters; vocabulary = 256 bytes + {len(tok.merges)} merges + {len(tok.special_tokens)} special = {tok.vocab_size}")
print("first 12 merges (each joins the most frequent adjacent pair):")
for (a, b), new in list(tok.merges.items())[:12]:
    print(f"  {new}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[new]!r}")

# %% 2. Token boundaries
for s in ["the small cats chase the dog .", "the unicorns chase the dog .", "The Cats", "naïve café 你好 🌍", "12345 + 678"]:
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    print(f"\n{s!r}\n  {len(s.encode('utf-8'))} bytes -> {len(ids)} tokens: {show(tok, s)}")

# %% 3. Vocabulary size versus sequence length
sample = text[:20000]
print("\nvocab size -> bytes per token on in-domain text")
for extra in (0, 10, 30, 60, 120):
    t = BPETokenizer().train(text, 256 + extra + len(SPECIAL_TOKENS))
    print(f"  {t.vocab_size:5d} ({len(t.merges):3d} merges) -> {compression_ratio(t, sample):.2f}")

# %% 4. Special tokens cannot be typed
bos = tok.get_bos_token_id()
typed = tok.encode("<|bos|>")
print(f"\n<|bos|> as a control token: id {bos}")
print(f"the STRING '<|bos|>' typed by a user: {len(typed)} ordinary tokens {typed} -> {show(tok, '<|bos|>')}")
assert bos not in typed
