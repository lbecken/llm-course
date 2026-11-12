# Solutions — Module 13

## Lab exercises

### Exercise 1: by hand

1. Eight significant bits: the spacing in $[2^e, 2^{e+1})$ is $2^{e-7}$. Near 1.0: $2^{-7} = 0.0078$. Near 0.01 (in $[2^{-7}, 2^{-6})$): $2^{-14} = 6.1\times10^{-5}$. A weight of 0.05 lies in $[2^{-5}, 2^{-4})$ with spacing $2^{-12} = 2.4\times10^{-4}$; an update of $10^{-5}$ is below half the spacing and is **lost**.
2. Tokens per pass per GPU: (a, b) $32\times2048 = 65\,536$; (c) $8\times2048 = 16\,384$. Accumulation steps: (a) $1\,048\,576 / (8\cdot65\,536) = $ **2**; (b) **16**; (c) **64**. None changes the training result (same total batch, same number of optimizer steps, same learning rates), up to rounding and the order in which data is consumed. Only wall-clock time and memory differ.
3. Weights $900\text{M}\times4 = 3.6$ GB; gradients 3.6 GB; state: $440\text{M}\times4 = 1.76$ GB (one momentum buffer) + $460\text{M}\times8 = 3.68$ GB (two AdamW moments). Total ≈ **12.6 GB**, before any activation. (nanochat is somewhat below this, since it stores its embedding tables in bf16.)
4. Step time $524\,288/20\,000 = 26.2$ s versus $524\,288/26\,000 = 20.2$ s: 6.05 s saved per step. $240 / 6.05 \approx$ **40 steps**. Any run longer than a quick probe should compile.

```python
import torch
w = torch.tensor(0.05, dtype=torch.bfloat16)
print(w.item(), (w + torch.tensor(1e-5, dtype=torch.bfloat16) == w).item())   # 0.050048828125 True
```

### Exercise 2: formats

Reference tables: lesson D.2.

1. fp16: 1.9766, bf16: 1.0000. fp16 is closer because it has 11 significant bits against 8. bf16 is nevertheless preferred for *computation*: it cannot overflow or underflow where fp32 would not, so no gradient scaler and no skipped steps; and its coarse resolution is harmless as long as nothing is *accumulated* in it.
2. Measured: **4.0** (fp32: 11.0002, fp16: 10.742). Near 1.0 the spacing is 0.0078, so adding 0.01 lands on the nearest representable value, a step of 0.0078: the weight creeps up, too slowly. From 2.0 the spacing is 0.0156 and each addition rounds *up* to a full step: too fast. At 4.0 the spacing becomes 0.031; 0.01 is now less than half of it and every further addition is rounded away. The result is neither "nothing happened" nor "correct": low-precision accumulation is wrong in a value-dependent way.
3. An AdamW step changes each weight by roughly the learning rate (the normalised update has magnitude near 1). The update survives if it exceeds half the spacing at the weight's magnitude; a larger learning rate clears that bar for more weights. With `--lrs 1e-4` (measured): fp32 1.5795, fp16 1.5741, **bf16 1.7900**, with 53.6 % of bf16 updates lost per step (fp16: 12.2 %). The smaller the step, the worse; and schedules end with small steps.
4. (a) nanochat gives embeddings a very large AdamW learning rate (0.3, scaled by width) against values initialised with standard deviation 0.8, whose bf16 spacing is about 0.004: each update is far larger than the spacing. (b) A row is updated only when its token occurs in the batch, so updates are sparse and individually large rather than a steady drip of tiny increments. (c) The AdamW moment estimates are separate optimizer state; what is rounded is only the stored row. The source states the trade explicitly: reduced-precision embeddings are tolerated "and it saves memory". The rule of D.2 is about *small relative updates*, not about dtype as such.
5. No reference values (not run on GPU by the author).

### Exercise 3: memory

Reference table: lesson D.1.

1. From the three points: slope $(2080 - 528)/(8192 - 2048) \approx 0.25$ MiB per token; intercept near 10 MiB (the 512-token point lies somewhat below the line). $b$ is the activation memory kept per token of the batch (here inflated by materialised attention matrices: $H\times T$ floats per token per layer, twice); $a$ is what does not scale with the batch and was not already allocated by the warm-up step.
2. For example: (i) plain, 8 rows × 4 accumulated passes: 528 MiB, 4 × 0.57 s = 2.3 s per optimizer step (the same as one batch of 32: 2.34 s). (ii) checkpointed, 8 rows × 4: 173 MiB, 4 × 0.64 s = 2.6 s. Checkpointed at 32 rows (744 MiB) does not fit. Option (i) is preferable: it fits and costs no time.
3. Yes, still equal to rounding: a mean over 32 rows is a sum of 32 terms divided by 32 however it is grouped. Dropout would differ only by the random masks drawn (statistically equivalent, not bitwise). Batch normalisation would genuinely differ: its statistics are computed per forward pass, so a micro-batch of 1 normalises differently from a batch of 32.
4. Checkpointing removes the *stored internals of the blocks*, but (a) each block's input is still stored (8 residual streams), (b) everything outside the blocks is untouched: embeddings, the final normalisation and above all the logits ($32\times256\times1024$ floats, in several copies through softmax and loss), and (c) during the backward pass one block at a time is recomputed with *all* its internals live, including its attention matrices. Peak = fixed parts + one block's internals, not total ÷ 8.

### Exercise 4: reading a training log

Reference for the miniature CPU run ([`results/reference/nanochat-local-cpu.md`](../results/reference/nanochat-local-cpu.md)):
compute dtype `float32` ("auto-detected: no CUDA (CPU/MPS)"); total batch 2 048 tokens =
device batch 16 × sequence 128, so 1 accumulation step; steady state 40.6 k tokens/s
compiled versus 31.6 k uncompiled; whole command 80 s versus 24 s; break-even about
3 900 steps.

2. MFU is meaningful only if the GPU's name matches an entry of `get_peak_flops`; otherwise the peak is infinite and the log shows 0.00. On CPU it is always meaningless.
4. No reference values (GPU). Expectation: with the total batch fixed, tokens/s should depend only weakly on the device batch size once the device is busy (larger micro-batches amortise per-pass overhead slightly); peak memory should scale almost linearly with it. A strong dependence of tokens/s on device batch size points to overhead outside the model (loader, Python, synchronisation).

### Exercise 5: the benchmark suite

Reference: [`results/reference/benchmarks-cpu.md`](../results/reference/benchmarks-cpu.md).

1. On the reference machine, sizes in bytes and all correctness columns were identical across three runs; **timings varied by up to ±15 %** (the plain training step: 2.15, 2.49, 2.88 s). A claimed 5 % speed difference is therefore not supportable from single runs; it needs repeated, interleaved measurements and an interval (Module 10).
2. Own comparison: any is acceptable if it has a correctness or quality column and states whether the optimisation is exact.
3. See the Milestone 3 rubric. The most common weakness in reports: a speed-up stated without the sequence length, batch size or model size at which it was measured.

A wrong approach worth naming: benchmarking only what is convenient to time. The suite
deliberately puts a correctness column in every row, because the cheapest way to make
code faster is to make it wrong.

### Challenge: a gradient scaler

Outline. Underflow demonstration: in fp16 the smallest positive normal number is about
6e-5 (subnormals reach 6e-8); gradients of early layers in a deep or badly scaled model
fall below that and become exactly zero, so those layers stop learning while the loss
looks merely "slow". Scaler: `(loss * S).backward()`; before the step, divide each
gradient by `S` (in fp32); if any gradient is non-finite, skip the step and set
`S ← S/2`; otherwise every $n$ steps `S ← 2S`. The chain rule makes the scaling exact
for every intermediate gradient, which is the point: intermediate values are lifted out
of the underflow range, and the final division restores the true gradient in a format
that can represent it. `torch.amp.GradScaler` implements the same logic (default growth
factor 2 every 2 000 steps, backoff 0.5).

### Written questions

1. (i) Halve the device batch size and keep the total batch: nothing changes about the experiment. (ii) Repeat down to one row. (iii) Enable checkpointing if available: same experiment, more time. (iv) Shorten the sequence length: changes the experiment (context, data packing, FLOPs per token). (v) Smaller model: a different experiment.
2. The recipe (total batch, number of steps, learning rates) is hardware-independent and transfers: accumulation replaces ranks. Wall-clock time is steps × tokens per step ÷ tokens/s, and tokens/s is a property of the hardware, precision, kernels and compiler actually used. It must be measured on the target machine.
3. Timings: everything faster in absolute terms; the KV-cache ratio at small model sizes may *shrink* (decode steps dominated by kernel-launch overhead); fused attention's advantage over materialised attention grows with $T$; bf16 becomes a real speed-up; accumulation no longer beats one large batch. Memory figures need GPU counters. Correctness columns should not change beyond rounding (bf16 differences are somewhat larger than fp32 ones).

## Quiz

1. Weights, gradients, optimizer state (independent of the batch), and activations kept for the backward pass, which grow linearly with tokens per forward pass (batch rows × sequence length), times layers and width.
2. fp16 gives up exponent range (maximum 65 504, underflow near 6e-5): it risks overflow to infinity and underflow of small gradients to zero. bf16 gives up resolution (8 significant bits): it risks losing small increments, i.e. it cannot be used to accumulate.
3. Spacing $4\times2^{-7} = 0.03125$. Each +0.01 is less than half the spacing and rounds back: the weight stays at **4.0** after 500 updates (the true value would be 9.0).
4. Compute in low precision but keep master weights and optimizer state in fp32. Also in fp32: the loss computation (softmax/cross-entropy, in nanochat also the logit soft-cap) and any long accumulation (moments, sums).
5. fp16's limited exponent range makes small gradients underflow to zero and large values overflow; scaling the loss shifts gradients into the representable range and detects overflow. bf16 has the same exponent range as fp32, so neither occurs where it would not in fp32.
6. $16\times1024 = 16\,384$ tokens per pass per GPU, 32 768 across 2 GPUs: $262\,144/32\,768 = $ **8** accumulation steps. On 1 GPU: 16. The training result is unchanged up to rounding and data order; a step takes about twice as long.
7. Without the division the accumulated gradient is the *sum* of micro-batch means, $k$ times too large: with SGD an effective learning rate $k$ times higher (adaptive optimizers hide much of this, which makes the bug easy to miss). `zero_grad()` inside the loop discards all but the last micro-batch: the step uses a gradient from $1/k$ of the data, scaled down by $k$.
8. Accumulation trades nothing in arithmetic for memory by splitting the batch across passes; it cannot go below one row. Checkpointing trades extra forward computation for memory by not storing block internals; it is needed when a single row does not fit, or when memory per row must fall further.
9. $8\times10^{8}\times5\times10^{4} = 4\times10^{13}$ FLOPs/s; MFU = **20 %**. Reasons: operations that are not matmuls (normalisation, softmax, element-wise functions) or are memory-bound; the optimizer step and data loading inside the timed step; kernel-launch and synchronisation overhead; the peak figure assumes ideal matmul shapes and precision.
10. It traces the Python model into a graph and generates fused, interpreter-free code specialised to the input shapes; it costs start-up time and, with `dynamic=False`, requires fixed shapes. Evaluation and sampling use other shapes (variable prompt lengths, different batch sizes), and checkpoints should hold plain parameter names, so the original module is kept for those.
11. (a) No synchronisation: GPU work is asynchronous, so the clock measures launch time, not execution. (b) Only `backward()` is timed, not the forward pass, optimizer step or data loading. (c) Warm-up (compilation, allocator, autotuning) is included and 20 steps are few. Also: `time.time()` rather than a monotonic clock, and no report of batch size, precision or hardware.
12. Local: forward and backward on each rank's own data. Communicated: gradients (averaged across ranks). Identical afterwards: the weights (and consistent optimizer state). `DistributedDataParallel` hooks the backward pass and all-reduces every gradient so each rank holds the full averaged gradient and a full copy of the optimizer state. nanochat synchronises inside the optimizer: for large parameters each rank receives only its slice of the averaged gradient (reduce-scatter), updates that slice with the only copy of its optimizer state, and the updated slices are gathered back, with communication overlapped with computation.
