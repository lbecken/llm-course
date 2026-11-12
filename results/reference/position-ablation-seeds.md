# Reference result: position information, three seeds (CPU)

| | |
|---|---|
| Date | 2026-11-11 |
| Hardware | 4-vCPU x86-64 sandbox, no GPU |
| Software | Python 3.13, torch 2.15.0+cpu |
| Status | **Measured** by the course author. 24 runs: 4 variants × 3 seeds × 2 training lengths. |

Setup: MiniGPT, nanochat style, `block_size=64, n_layer=2, n_head=2, n_embd=64`, 10 000
Pocket sentences, AdamW lr 3e-3, batch 32. Only `pos` and the seed vary. "Agreement" is
accuracy on 200 minimal pairs with an attractor noun of the opposite number (chance = 0.5).
Entropy floor: 0.3139 nats/char. Reproduce one cell with
`python labs/foundations/04_positions.py --steps 800` (seed 0 only).

## 800 steps

| pos | val loss (seeds 0, 1, 2) | agreement with attractor (seeds 0, 1, 2) |
|---|---|---|
| none | 0.5305, 0.5326, 0.5213 | 0.365, 0.42, 0.43 |
| learned | 0.3596, 0.3582, 0.3546 | 0.035, 0.59, 0.92 |
| sinusoidal | 0.3541, 0.3540, 0.3540 | 0.22, 0.70, 1.00 |
| rope | 0.3513, 0.3534, 0.3502 | 0.985, 0.18, 1.00 |

## 2000 steps

| pos | val loss (seeds 0, 1, 2) | agreement with attractor (seeds 0, 1, 2) |
|---|---|---|
| none | 0.3948, 0.3857, 0.3860 | 0.32, 0.39, 0.35 |
| learned | 0.3475, 0.3475, 0.3474 | 1.00, 1.00, 1.00 |
| sinusoidal | 0.3467, 0.3471, 0.3474 | 1.00, 1.00, 1.00 |
| rope | 0.3469, 0.3475, 0.3474 | 1.00, 1.00, 1.00 |

## What the data support

- **Position information matters.** Without it the loss stays clearly higher at both
  training lengths, and agreement stays *below* chance: the model follows the nearest
  noun. This holds for all three seeds.
- **At 2000 steps the three ways of encoding position are indistinguishable** on this
  task, in loss and in agreement.
- **At 800 steps the agreement score of every variant swings between about 0 and 1 across
  seeds**, while the loss differs only in the third decimal. The first single-seed run of
  this experiment showed rope 0.98 vs. learned 0.01 and invited the conclusion "RoPE
  learns long-range structure faster". Three seeds show that conclusion is not supported:
  the models are caught mid-transition, and which side of the transition a run is on at
  step 800 depends on the seed.

The general lessons (Module 10): a behavioural metric can be far noisier than the loss; a
difference seen with one seed is a hypothesis, not a result.
