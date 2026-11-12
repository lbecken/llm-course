"""
Lab 08b — Duplicates, contamination, and a loss that is too good to be true.
Tier A (about 20 s, CPU)

    python labs/pretraining/08_data_quality.py

1. Find exact and near duplicates in a small document set.
2. Contaminate a validation set with training documents and watch a memorising model
   report a loss BELOW the entropy of the language on the leaked part, which is
   impossible for honest data, while the half-contaminated average still looks plausible.
3. Detect the contamination with n-gram overlap.
Success criterion: you can state the rule "a validation loss below the plausible floor
means leakage until proven otherwise", and the detector flags exactly the leaked items.
"""

import random

from mlle.data import pocket
from mlle.data import quality as dq
from mlle.foundations.ngram import CharTokenizer, NGramLM

# %% 1. Duplicates
docs = [
    "The cat sat on the mat and looked at the dog in the garden for a while.",
    "the cat sat on  the mat and looked at the dog in the garden for a while.",       # case/space variant
    "The cat sat on the mat and looked at the dog in the garden for a long while.",   # one word added
    "Transformers process all positions of a sequence in parallel using attention.",
]
print("1. exact duplicates after normalisation:", dq.exact_duplicates(docs))
print("   near duplicates (Jaccard of 5-word shingles >= 0.6):",
      [(i, j, round(s, 2)) for i, j, s in dq.near_duplicates(docs, threshold=0.6)])

# %% 2. A contaminated validation set
rng = random.Random(0)
make_doc = lambda: "".join(pocket.sample_sentence(rng).text for _ in range(8))
train_docs = [make_doc() for _ in range(400)]
clean_val = [make_doc() for _ in range(100)]
leaked_val = clean_val[:50] + rng.sample(train_docs, 50)  # half of it was seen in training

tok = CharTokenizer("".join(train_docs + clean_val))
train_ids = tok.encode("".join(train_docs))
H = pocket.entropy_rate()["nats_per_char"]
sets = {"clean val": clean_val, "contaminated val": leaked_val, "leaked half only": leaked_val[50:]}
print(f"\n2. validation loss in nats/char (entropy floor of the language: {H:.4f})")
print(f"{'model':>28} " + " ".join(f"{name:>17}" for name in sets))
for order, name in ((4, "4-char model (generalises)"), (12, "12-char model"), (40, "40-char model (memorises)")):
    lm = NGramLM(order, tok.vocab_size, alpha=1e-3).fit(train_ids)
    losses = [lm.cross_entropy(tok.encode("".join(d))) for d in sets.values()]
    flag = "   <-- below the floor" if min(losses) < H else ""
    print(f"{name:>28} " + " ".join(f"{loss:>17.4f}" for loss in losses) + flag)

# %% 3. Detect it
report = dq.contamination_rate(train_docs, leaked_val, n=8, threshold=0.5)
print(f"\n3. contamination check (8-word shingles): {report['rate']:.0%} of validation documents flagged")
print(f"   flagged indices are exactly the leaked half: {report['flagged'] == list(range(50, 100))}")
clean_report = dq.contamination_rate(train_docs, clean_val, n=8, threshold=0.5)
print(f"   same check on the clean validation set: {clean_report['rate']:.0%} flagged "
      f"(mean overlap {sum(clean_report['overlaps']) / len(clean_report['overlaps']):.2f}: short common phrases are shared legitimately)")
