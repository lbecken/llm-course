"""
Data hygiene tools (Module 08): duplicates, near-duplicates, train/eval contamination.

All three reduce to comparing sets of n-gram "shingles":

    exact duplicates      identical normalised text         -> hash equality
    near duplicates       high Jaccard overlap of shingles  -> |A & B| / |A | B|
    contamination         an evaluation item whose shingles mostly occur in training data

Production pipelines use MinHash/LSH to do this over billions of documents; the logic is
the same as in these few lines.
"""

from __future__ import annotations

import hashlib
import re


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace, so trivial formatting differences do not hide duplicates."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def exact_duplicates(docs: list[str]) -> list[tuple[int, int]]:
    """Pairs (first_index, duplicate_index) of documents identical after normalisation."""
    seen: dict[str, int] = {}
    pairs = []
    for i, doc in enumerate(docs):
        h = content_hash(doc)
        if h in seen:
            pairs.append((seen[h], i))
        else:
            seen[h] = i
    return pairs


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Set of word n-grams of the normalised text."""
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def near_duplicates(docs: list[str], threshold: float = 0.7, n: int = 5) -> list[tuple[int, int, float]]:
    """All pairs with shingle Jaccard similarity >= threshold. O(N^2): for small corpora and teaching."""
    sets = [shingles(d, n) for d in docs]
    out = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            s = jaccard(sets[i], sets[j])
            if s >= threshold:
                out.append((i, j, s))
    return out


def contamination_rate(train_docs: list[str], eval_docs: list[str], n: int = 8, threshold: float = 0.5) -> dict:
    """Fraction of evaluation documents whose n-gram shingles mostly appear in the training data."""
    train_shingles: set = set()
    for d in train_docs:
        train_shingles |= shingles(d, n)
    overlaps = []
    for d in eval_docs:
        s = shingles(d, n)
        overlaps.append(len(s & train_shingles) / max(1, len(s)))
    flagged = [i for i, o in enumerate(overlaps) if o >= threshold]
    return {"rate": len(flagged) / max(1, len(eval_docs)), "flagged": flagged, "overlaps": overlaps}
