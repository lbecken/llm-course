"""
Build a tiny pretraining dataset in nanochat's on-disk format, from local text.

nanochat's dataloader (nanochat/dataloader.py + nanochat/dataset.py) reads every
*.parquet file in  $NANOCHAT_BASE_DIR/base_data_climbmix/ , expects a string column
named "text" (one document per row), uses all files but the last for training and the
LAST file as the validation split. That is the whole contract, so we can substitute our
own miniature dataset and run the unmodified pipeline in minutes on a CPU.

    # synthetic "Pocket" language (default), ~2 MB
    python scripts/experiments/make_local_dataset.py

    # your own text: documents are separated by blank lines
    python scripts/experiments/make_local_dataset.py --text-file my_corpus.txt

Always set NANOCHAT_BASE_DIR to a scratch directory for these runs, so that a miniature
tokenizer/checkpoint can never be confused with a real one in ~/.cache/nanochat.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mlle.data import pocket


def pocket_documents(num_docs: int, seed: int) -> list[str]:
    """Each document is a short "story": 3 to 12 Pocket sentences."""
    rng = random.Random(seed)
    docs = []
    for _ in range(num_docs):
        n = rng.randint(3, 12)
        docs.append("".join(pocket.sample_sentence(rng).text for _ in range(n)).strip())
    return docs


def file_documents(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-file", type=str, default=None, help="plain-text file; blank lines separate documents")
    ap.add_argument("--num-docs", type=int, default=12000, help="number of synthetic documents (ignored with --text-file)")
    ap.add_argument("--num-shards", type=int, default=4, help="total parquet files; the last one becomes the validation split")
    ap.add_argument("--row-group-size", type=int, default=256, help="documents per parquet row group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        raise SystemExit("Set NANOCHAT_BASE_DIR to a scratch directory first, e.g. export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local")
    out_dir = Path(base) / "base_data_climbmix"  # the directory name nanochat/dataset.py looks for
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = file_documents(args.text_file) if args.text_file else pocket_documents(args.num_docs, args.seed)
    random.Random(args.seed).shuffle(docs)
    assert len(docs) >= args.num_shards >= 2, "need at least 2 shards (train + val) and one document per shard"
    per_shard = len(docs) // args.num_shards
    total_chars = 0
    for i in range(args.num_shards):
        chunk = docs[i * per_shard : (i + 1) * per_shard]
        total_chars += sum(len(d) for d in chunk)
        table = pa.table({"text": chunk})
        pq.write_table(table, out_dir / f"shard_{i:05d}.parquet", row_group_size=args.row_group_size)
    print(f"wrote {args.num_shards} shards ({per_shard} docs each, {total_chars:,} chars total) to {out_dir}")
    print(f"train: shards 0..{args.num_shards - 2}   val: shard {args.num_shards - 1}")


if __name__ == "__main__":
    main()
