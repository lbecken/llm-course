# Hardware, experiment tiers and cost

How long things take, how to find out before you start, and when renting a bigger
machine is worth it.

## The four tiers

| Tier | Purpose | Target duration | Typical hardware | Examples |
|---|---|---|---|---|
| **A — Micro** | Numerical demonstrations, tiny models, unit tests | seconds to minutes | CPU is enough | every lab in Part I; MiniGPT on Pocket; nanochat on the local miniature dataset; all agent labs with scripted or tiny models |
| **B — Small** | Architecture changes, miniature training runs, benchmarks | minutes to about an hour | DGX Spark GPU (some run on CPU) | nanochat depth 4–8 probes and short runs; KV-cache and precision benchmarks; LoRA on a small model |
| **C — Extended** | Meaningful pretraining, SFT, selected optimization studies | hours to overnight | DGX Spark GPU | nanochat depth 8–12 pretraining at the planned token budget; SFT of your base model; RL on GSM8K |
| **D — Optional research** | Larger models and comparative studies | days, or rented multi-GPU | DGX Spark for days, or cloud | depth 16–20 on the Spark; miniseries across depths; `SSSL` + FP8 runs on Hopper-class hardware |

These are planning categories. They are not performance guarantees. Each lab states its
tier, and every Tier B–D lab starts with a measurement step.

## What was actually measured for this course

Only CPU measurements exist so far. They were taken on 2026-11-11 on a 4-vCPU x86-64
sandbox (15 GiB RAM, no GPU), and are recorded with commands in
[results/reference/](../results/reference/).

| Experiment | Tier | Measured |
|---|---|---|
| Course test-suite (`make test`) | A | 15 s |
| MiniGPT, 100 k parameters, 1 500 steps on Pocket | A | 25–35 s; ~90–125 k tokens/s; validation loss 0.348 nats/char vs. source entropy 0.314 |
| nanochat depth 4 (`n_embd` 64), 300 steps, local dataset, compiled | A | 80 s total wall time (about 60 s of it compilation and start-up); ~40 k tokens/s; validation 0.466 bits/byte vs. source entropy 0.453 |
| same with `TORCHDYNAMO_DISABLE=1` | A | 24 s total; ~32 k tokens/s |

Everything about GPUs below is either exact arithmetic about the model, or clearly
labelled extrapolation.

## How big is a nanochat run? (exact arithmetic)

nanochat sizes a run from a single dial, `--depth`
([`scripts/base_train.py` L129–L143, L263–L304, L340–L357](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L129-L143)):

1. Width: `n_embd = depth × 64`, rounded up to a multiple of `head_dim = 128`.
2. Token budget: `12 ×` the number of "scaling parameters" (block matrices + output head).
3. Batch size: 2¹⁹ tokens at depth 12, scaled as budget^0.383 and rounded to a power of two.
4. Steps: budget ÷ batch size.

[`scripts/experiments/estimate_run.py`](../scripts/experiments/estimate_run.py) reproduces
this arithmetic from the pinned code on the "meta" device (no memory needed). Its output
for the defaults (vocabulary 32 768, sequence length 2 048, `--window-pattern SSSL`):

| depth | n_embd | heads | parameters (total) | scaling parameters | FLOPs / token | batch (tokens) | steps | training tokens | total FLOPs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 2 | 36,700,242 | 11,534,384 | 8.02e+07 | 262,144 | 528 | 138,412,032 | 1.11e+16 |
| 6 | 384 | 3 | 73,531,538 | 23,199,852 | 1.68e+08 | 262,144 | 1,062 | 278,396,928 | 4.66e+16 |
| 8 | 512 | 4 | 125,829,354 | 41,943,232 | 2.96e+08 | 262,144 | 1,920 | 503,316,480 | 1.49e+17 |
| 12 | 768 | 6 | 286,261,730 | 110,100,912 | 7.60e+08 | 524,288 | 2,520 | 1,321,205,760 | 1.00e+18 |
| 16 | 1,024 | 8 | 536,871,738 | 234,881,792 | 1.59e+09 | 524,288 | 5,376 | 2,818,572,288 | 4.47e+18 |
| 20 | 1,280 | 10 | 896,533,746 | 435,160,240 | 2.89e+09 | 1,048,576 | 4,980 | 5,221,908,480 | 1.51e+19 |
| 24 | 1,536 | 12 | 1,384,122,122 | 729,810,624 | 4.78e+09 | 1,048,576 | 8,352 | 8,757,706,752 | 4.18e+19 |
| 26 | 1,664 | 13 | 1,681,786,938 | 918,423,532 | 5.99e+09 | 1,048,576 | 10,510 | 11,020,533,760 | 6.60e+19 |

With `--window-pattern L` (recommended without Flash Attention 3) the parameter counts and
token budgets are identical; FLOPs per token rise because every layer attends to the full
context: 1.96e+08 (d6), 8.87e+08 (d12), 3.24e+09 (d20).

Two things worth noticing now (Module 09 explains them):

- "Total parameters" is dominated by embedding tables: at depth 12, the token embedding,
  six value-embedding tables and the output head account for about 201 M of the 286 M
  parameters. Lookups cost no matmul FLOPs, which is why nanochat sizes the run by
  *scaling* parameters instead.
- The reference speedrun ([`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh)) trains depth 24 with
  `--target-param-data-ratio=8`, i.e. two thirds of the default budget, on eight H100s
  with FP8. None of those three conditions holds on a Spark.

## From tokens to hours (needs your measurement)

```text
hours = training tokens / (measured tokens per second) / 3600
```

Get the denominator from the probe ([dgx-spark-setup.md, Section 6](dgx-spark-setup.md#6-measure-first-the-probe)):

```bash
DEPTH=12 BATCH=32 scripts/experiments/spark_probe.sh
# or, for any existing log:
python scripts/experiments/parse_train_log.py my_run.log --depth 12 --window-pattern L
```

### A planning extrapolation, clearly labelled as such

The only public DGX Spark numbers for nanochat known to the author are from
October–November 2025, for the **2025 architecture** at depth 20 (561 M parameters,
3.49e9 FLOPs/token): about 39.5 s per 524 288-token step, i.e. roughly 13 k tokens/s,
and a complete 11.2 B-token pretraining in about 9.8 days
([nanochat Discussion #28](https://github.com/karpathy/nanochat/discussions/28)).
That corresponds to roughly 4.6e13 model FLOPs per second.

*If* a current-revision model reached the same FLOPs per second on your Spark (it may
not: the architecture, attention path, compiler and PyTorch version all differ), the
table above would translate to:

| depth (`--window-pattern L`) | training tokens | hypothetical tokens/s | hypothetical duration | tier |
|---:|---:|---:|---:|---|
| 6 | 0.28 B | ~235 k | ~20 min | B |
| 8 | 0.50 B | ~130 k | ~1 h | B–C |
| 12 | 1.32 B | ~52 k | ~7 h | C |
| 16 | 2.82 B | ~26 k | ~30 h | D |
| 20 | 5.22 B | ~14 k | ~100 h | D |

Small models rarely reach the FLOP rate of large ones (fixed overheads dominate), so the
upper rows are probably optimistic. **These are not predictions. Replace this table with
your probe results** and record them in `results/runs/`.

## Recommended ladder for the Spark

1. **Tier A, CPU or GPU:** `make nanochat-local-data nanochat-local-train nanochat-local-eval`.
   Confirms that the whole pipeline works before any download or long run.
2. **Tier B:** probe depth 6 and depth 12 (30 steps each). You now know your tokens/second and memory.
3. **Tier B–C:** train depth 6 or 8 to its planned budget. This is the Part II milestone model.
4. **Tier C:** train depth 12 overnight if the probe says it fits. Use it for Parts III–IV.
5. **Tier D (optional):** depth 16–20, or a miniseries, only with a specific question in mind.

Reduce cost without changing the question: lower `--target-param-data-ratio` (shorter
horizon, undertrained model), use `--num-iterations` for fixed-step comparisons, turn off
`--core-metric-every` and `--sample-every` during sweeps, and compare runs at equal
`total_training_flops` or equal tokens, never at equal wall-clock time across machines.

## When is renting a GPU worth it?

Rent when the *question* needs it, not to make progress bars move faster:

- **Reproducing upstream numbers** (CORE at depth 24–26, the leaderboard). Those assume
  Flash Attention 3 on Hopper, `SSSL` windows, FP8 and 8 ranks. A Spark cannot run that
  configuration; an 8×H100 node can, for roughly two hours at upstream's stated ~$24/hour.
- **Experiments about multi-GPU behaviour** (the distributed optimizer, DDP scaling).
- **Sweeps** where you need many depth-12-class runs in a day.

Do not rent for Parts I, V and VI, for the benchmarking methodology of Part III, or for
SFT/RL on a small base model: the Spark (and often a CPU) is sufficient, and a slower
machine you can inspect teaches more than a fast one you are paying for by the hour.

## Recording results

Every lab writes structured records (JSON lines) into `results/runs/`, which is
git-ignored. Templates for written observations are in
[results/templates/](../results/templates/). Reference measurements made by the course
author are in [results/reference/](../results/reference/), each with its hardware, date
and exact command. A number without those three things is an anecdote.
