"""
Lab 03b — nanochat's tokenizer on your own text.   Tier A (seconds, CPU). Needs `make env-cpu`.

    python labs/foundations/03_nanochat_tokenizer.py

Trains nanochat's RustBPETokenizer (rustbpe for training, tiktoken for encoding) directly
from an in-memory iterator, compares it with the course's pure-Python BPE trained on the
same text, and renders a chat conversation into tokens with its loss mask.
Success criterion: both tokenizers round-trip; the rendered conversation shows mask = 1
only on assistant-generated tokens.
"""

from mlle.data import pocket
from mlle.foundations.bpe import SPECIAL_TOKENS, BPETokenizer
from mlle.nanochat_bridge import require_nanochat

require_nanochat()
from nanochat.tokenizer import SPLIT_PATTERN, RustBPETokenizer  # noqa: E402

docs = [pocket.generate_corpus(20, seed=i) for i in range(400)]
vocab_size = 256 + 64 + len(SPECIAL_TOKENS)

# %% 1. Train both tokenizers on identical text
nc_tok = RustBPETokenizer.train_from_iterator(iter(docs), vocab_size)
my_tok = BPETokenizer().train("".join(docs), vocab_size)
print("nanochat split pattern:", SPLIT_PATTERN)
print(f"nanochat vocab size: {nc_tok.get_vocab_size()}   course BPE vocab size: {my_tok.vocab_size}")

sample = "the quick foxes that chase the dog sleep ."
a, b = nc_tok.encode(sample), my_tok.encode(sample)
print(f"\n{sample!r}")
print("  nanochat :", [nc_tok.decode([i]) for i in a])
print("  course   :", [my_tok.decode([i]) for i in b])
assert nc_tok.decode(a) == sample and my_tok.decode(b) == sample

# %% 2. Special tokens sit at the end of the vocabulary and are added only by code
for name in SPECIAL_TOKENS:
    print(f"  {name:22s} nanochat id {nc_tok.encode_special(name)}   course id {my_tok.encode_special(name)}")
print("encode('<|bos|>') as ordinary text ->", [nc_tok.decode([i]) for i in nc_tok.encode("<|bos|>")])

# %% 3. A conversation rendered to tokens, with the training mask (Module 14 builds on this)
conversation = {"messages": [
    {"role": "user", "content": "the cat sleeps ."},
    {"role": "assistant", "content": [
        {"type": "text", "text": "the dog "},
        {"type": "python", "text": "1+1"},
        {"type": "python_output", "text": "2"},
        {"type": "text", "text": " runs ."},
    ]},
]}
ids, mask = nc_tok.render_conversation(conversation)
print("\nrendered conversation (token, mask):")
for i, m in zip(ids, mask):
    print(f"  {nc_tok.decode([i])!r:26s} mask={m}")
