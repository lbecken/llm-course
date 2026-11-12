"""
Packing documents into fixed-length training rows (Module 08).

A training batch is a (B, T) grid of tokens. Documents have every possible length.
How they are laid into rows decides (a) how many tokens are thrown away, (b) how many
positions are wasted on padding, and (c) whether a position can see the start of its
own document. Three strategies, all operating on a list of document lengths:

    stream      concatenate everything and cut into rows of T+1 (the classic approach,
                nanochat's original loader): no waste, but most rows start mid-document.
    bestfit_crop  every row starts with a document start (BOS); fill the row with the
                largest documents that still fit; when none fits, CROP one to fill the
                row exactly.  nanochat's pretraining loader (nanochat/dataloader.py).
    bestfit_pad   same, but when nothing fits the rest of the row is PADDING with the
                loss masked.  nanochat's fine-tuning loader (scripts/chat_sft.py).
                (A document longer than a whole row is truncated to the row.)

`simulate_packing` replays the algorithm on lengths only (no tokens needed) and reports
the accounting; `pack_rows` does the real thing for token lists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackingStats:
    strategy: str
    rows: int
    row_capacity: int
    tokens_in: int  # tokens in the documents that were consumed
    tokens_used: int  # tokens that ended up in rows
    tokens_cropped: int  # tokens discarded by cropping
    pad_positions: int  # positions filled with padding
    rows_starting_with_document: int

    @property
    def utilization(self) -> float:
        """Fraction of row positions that hold real tokens."""
        return self.tokens_used / (self.rows * self.row_capacity)

    @property
    def cropped_fraction(self) -> float:
        """Fraction of the consumed tokens that were thrown away."""
        return self.tokens_cropped / max(1, self.tokens_in)


def _best_fit(buffer: list[int], remaining: int) -> int:
    """Index of the LARGEST buffered document that fits in `remaining`, or -1."""
    best, best_len = -1, 0
    for i, n in enumerate(buffer):
        if best_len < n <= remaining:
            best, best_len = i, n
    return best


def simulate_packing(doc_lengths: list[int], row_capacity: int, strategy: str, num_rows: int, buffer_size: int = 1000) -> PackingStats:
    """Replay a packing strategy on document lengths. Documents are consumed in order,
    through a look-ahead buffer of `buffer_size` documents (as in nanochat)."""
    assert strategy in ("stream", "bestfit_crop", "bestfit_pad")
    it = iter(doc_lengths)
    stats = PackingStats(strategy, num_rows, row_capacity, 0, 0, 0, 0, 0)

    if strategy == "stream":
        carry = 0  # tokens of the current document still to be placed
        for _ in range(num_rows):
            pos = 0
            if carry == 0:
                stats.rows_starting_with_document += 1
            while pos < row_capacity:
                if carry == 0:
                    carry = next(it)
                    stats.tokens_in += carry
                take = min(carry, row_capacity - pos)
                pos += take
                carry -= take
                stats.tokens_used += take
        stats.tokens_in -= carry  # the unfinished tail has not been consumed yet
        return stats

    buffer: list[int] = []
    for _ in range(num_rows):
        pos = 0
        stats.rows_starting_with_document += 1
        while pos < row_capacity:
            while len(buffer) < buffer_size:
                buffer.append(next(it))
            remaining = row_capacity - pos
            i = _best_fit(buffer, remaining)
            if i >= 0:
                n = buffer.pop(i)
                stats.tokens_in += n
                stats.tokens_used += n
                pos += n
            elif strategy == "bestfit_crop":
                # nothing fits: crop the SHORTEST buffered document to fill the row exactly
                j = min(range(len(buffer)), key=buffer.__getitem__)
                n = buffer.pop(j)
                stats.tokens_in += n
                stats.tokens_used += remaining
                stats.tokens_cropped += n - remaining
                pos += remaining
            elif pos == 0:
                # bestfit_pad, empty row, and even the shortest document is longer than a row:
                # truncate it to the row (nanochat's render_conversation does this with max_tokens)
                j = min(range(len(buffer)), key=buffer.__getitem__)
                n = buffer.pop(j)
                stats.tokens_in += n
                stats.tokens_used += remaining
                stats.tokens_cropped += n - remaining
                pos += remaining
            else:  # bestfit_pad: nothing fits in what is left of this row -> pad it
                stats.pad_positions += remaining
                pos += remaining
    return stats


def pack_rows(docs: list[list[int]], row_capacity: int, pad_id: int, strategy: str = "bestfit_pad"):
    """Pack token lists into rows. Returns (rows, loss_mask) as lists of lists; mask is 0 on padding.
    Small-scale reference implementation (no look-ahead limit) used in tests and Module 14."""
    assert strategy in ("bestfit_crop", "bestfit_pad")
    buffer = [list(d) for d in docs]
    rows, masks = [], []
    while buffer:
        row: list[int] = []
        while len(row) < row_capacity and buffer:
            remaining = row_capacity - len(row)
            i = _best_fit([len(d) for d in buffer], remaining)
            if i >= 0:
                row.extend(buffer.pop(i))
            elif strategy == "bestfit_crop":
                j = min(range(len(buffer)), key=lambda k: len(buffer[k]))
                row.extend(buffer.pop(j)[:remaining])
            else:
                break
        mask = [1] * len(row) + [0] * (row_capacity - len(row))
        row = row + [pad_id] * (row_capacity - len(row))
        rows.append(row)
        masks.append(mask)
    return rows, masks
