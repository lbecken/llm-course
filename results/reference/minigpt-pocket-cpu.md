# Reference result: MiniGPT on Pocket (CPU)

| | |
|---|---|
| Date | 2026-11-11 |
| Hardware | 4-vCPU x86-64 sandbox, 15 GiB RAM, no GPU |
| Software | Python 3.13, torch 2.15.0+cpu |
| Status | **Measured** by the course author, one run per configuration, seed 0 |

## Command

```bash
python labs/foundations/07_train_minigpt.py --style nanochat --steps 1500
python labs/foundations/07_train_minigpt.py --style gpt2 --steps 1500
```

Configuration: 20 000 Pocket sentences (678 404 characters), character tokenizer
(25 symbols), `block_size=64, n_layer=2, n_head=2, n_embd=64`, AdamW lr 3e-3, batch 32.

## Results

| style | parameters | val loss @0 | @500 | @1000 | @1500 | train time | tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| nanochat | 101,504 | 3.2203 | 0.3640 | 0.3523 | 0.3481 | 35 s | ~93 k |
| gpt2 | 105,792 | 3.2468 | 0.4019 | 0.3581 | 0.3492 | 25 s | ~126 k |

- ln(25) = 3.219: both start at the loss of a uniform guess over the vocabulary.
- Source entropy: 0.3139 nats/char. Both models end about 0.035 nats above it.
  Part of the gap is structural: training windows start in the middle of sentences, where
  the first characters cannot be predicted well (Module 08 returns to this).
- Subject-verb agreement, 200 minimal pairs each: 100 % without and 100 % with an
  attractor noun of the opposite number, for both styles.
- Cached and uncached generation produced identical tokens (Module 11).

Timings on a shared sandbox vary by tens of percent between runs; the loss values are
reproducible on the same PyTorch version.
