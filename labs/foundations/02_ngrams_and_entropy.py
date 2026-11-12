"""
Lab 02 — Counting language models and the entropy floor.   Tier A (about 30 s, CPU)

    python labs/foundations/02_ngrams_and_entropy.py
    python labs/foundations/02_ngrams_and_entropy.py --sentences 500     # data sparsity

Trains unigram ... 6-gram character models on the synthetic Pocket language, compares
their cross-entropy with the exactly known entropy of the source, samples from them, and
runs the subject-verb agreement test that later separates n-grams from Transformers.
Success criterion: cross-entropy decreases with context length but stays above the
entropy floor; agreement WITH an attractor stays near chance (0.5) for short contexts.
"""

import argparse
import math

from mlle.data import pocket
from mlle.foundations.experiments import ngram_agreement_accuracy
from mlle.foundations.ngram import CharTokenizer, NGramLM, perplexity

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", type=int, default=20000, help="size of the training corpus")
ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6], help="context lengths to try")
args = ap.parse_args()

# %% 1. Data and the theoretical optimum
text = pocket.generate_corpus(args.sentences, seed=0)
tok = CharTokenizer(text)
ids = tok.encode(text)
split = int(0.9 * len(ids))
train = ids[:split]
val = tok.encode(pocket.generate_corpus(600, seed=999))  # fresh sentences, same for every corpus size
H = pocket.entropy_rate()
print(f"corpus: {len(text):,} characters, vocabulary {tok.vocab_size} symbols")
print(f"first sentences:\n{text[:160]}")
print(f"source entropy: {H['nats_per_char']:.4f} nats/char = {H['bits_per_char']:.4f} bits/char "
      f"(perplexity {perplexity(H['nats_per_char']):.3f});  uniform guess: ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}\n")

# %% 2. More context -> lower cross-entropy, until the data runs out
items_plain = pocket.agreement_items(200, with_attractor=False)
items_attr = pocket.agreement_items(200, with_attractor=True)
print(f"{'context':>8} {'val nats/char':>14} {'perplexity':>11} {'agreement':>10} {'agreement+attractor':>20} {'contexts seen':>14}")
for order in args.orders:
    lm = NGramLM(order, tok.vocab_size).fit(train)
    ce = lm.cross_entropy(val)
    a1 = ngram_agreement_accuracy(lm, tok, items_plain)
    a2 = ngram_agreement_accuracy(lm, tok, items_attr)
    print(f"{order:>8} {ce:>14.4f} {perplexity(ce):>11.3f} {a1:>10.2f} {a2:>20.2f} {lm.num_contexts():>14,}")
print(f"{'floor':>8} {H['nats_per_char']:>14.4f} {perplexity(H['nats_per_char']):>11.3f}")

# %% 3. Samples
for order in [o for o in (0, 1, 3, 6) if o in args.orders]:
    lm = NGramLM(order, tok.vocab_size).fit(train)
    sample = tok.decode(lm.generate(tok.encode("\nthe "), 70, seed=0))
    print(f"\n--- {order} characters of context ---{sample}")
