# Reference result: course benchmark suite (CPU)

| | |
|---|---|
| Date | 2026-11-12 |
| Hardware | 4-vCPU x86-64 sandbox, 15 GiB RAM, no GPU |
| Software | Python 3.13, torch 2.15.0+cpu |
| Status | **Measured** by the course author: three consecutive runs, nothing else running |

## Command

```bash
make bench        # = python scripts/benchmarks/run_all.py     (45 s per run on this machine)
```

## Results

`ratio` = baseline ÷ optimised. Values are the median of the three runs; the range over
the runs is given where it exceeded 2 %.

| Comparison | Kind | Baseline | Optimised | Ratio | Correctness / quality |
|---|---|---:|---:|---:|---|
| Generate 256 tokens: full recomputation vs KV cache | exact | 1.09 s (1.07–1.13) | 0.43 s | 2.6× (2.5–2.6) | max logit diff 4.4e-06; greedy tokens identical |
| Causal attention, T = 2 048: materialised scores vs fused kernel (SDPA) | exact | 69 ms (68–81) | 8.4 ms (7.4–8.6) | 9.2× (8.3–9.4) | max output diff 4.2e-07 |
| Causal attention, T = 2 048: largest score intermediate, full matrix vs 128 × 128 tiles | exact | 64 MiB | 0.25 MiB | 256× | max output diff 4.8e-07 |
| Forward pass, 1 024 tokens: fp32 vs bf16 autocast | approximate (rounding) | 16.6 ms (16.5–21.2) | 14.2 ms (12.7–15.8) | 1.3× (1.2–1.3) | loss 6.16721 vs 6.16753 (diff 3.2e-04) |
| Training step, 32 × 256 tokens, 6.8 M parameters: one batch vs 4 accumulated micro-batches | exact | 2 088 MiB | 553 MiB (545–558) | 3.8× | max gradient diff 8.4e-09; step 2.49 s (2.15–2.88) vs 1.58 s (1.55–1.60) |
| Same step: plain vs activation checkpointing | exact | 2 088 MiB | 795 MiB (746–825) | 2.6× (2.5–2.8) | max gradient diff 0; step 2.49 s (2.15–2.88) vs 2.62 s (2.56–2.78) |
| Weight matrices of a trained MiniGPT: fp32 vs int8 per channel | approximate | 406 016 bytes | 101 504 bytes | 4× | validation loss 0.3501 → 0.3501 (change below 1e-4 nats/char) |

## Reading these numbers

- **Run-to-run variation is large for timings.** The plain training step took 2.15, 2.49
  and 2.88 s in three consecutive runs (±15 %); the attention baseline 68–81 ms. Sizes in
  bytes and the correctness columns did not vary. A timing difference below roughly 20 %
  on this machine is not evidence of anything without repetition.
- **Accumulation was faster than the single large batch here** (1.58 s versus about
  2.5 s), which the lesson's "no cost in arithmetic" does not predict. On this CPU the
  smaller micro-batches fit the processor caches better. It is a property of this
  machine and model, not a general law; on a GPU large batches are normally at least as
  efficient.
- **Checkpointing's time cost is inside the noise in this run** (2.62 s versus 2.49 s),
  whereas the dedicated lab measured +36 % (lesson D.4). Two measurements of the same
  quantity that disagree are the usual situation in benchmarking; the memory saving
  (2.6×) was stable.
- **bf16 on CPU** is a different code path from bf16 on a GPU's tensor cores. The 1.3×
  here says nothing about GPU speed-ups.
- The KV-cache ratio (2.6× at 256 tokens) lies between the lab's 1.9× at 128 and 6.8× at
  512 tokens: the ratio depends on the sequence length (Module 11).

GPU: *not measured by the author.* Run `python scripts/benchmarks/run_all.py --device cuda`;
the two memory rows report CPU process memory and should be replaced by
`torch.cuda.max_memory_allocated` on a GPU (the training-step workers run on CPU).
