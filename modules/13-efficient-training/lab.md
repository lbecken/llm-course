# Module 13 — Lab: precision, memory and the benchmark suite

Scripts: [`labs/optimization/13_precision.py`](../../labs/optimization/13_precision.py) ·
[`labs/optimization/13_master_weights.py`](../../labs/optimization/13_master_weights.py) ·
[`labs/optimization/13_grad_accum.py`](../../labs/optimization/13_grad_accum.py) ·
[`scripts/benchmarks/run_all.py`](../../scripts/benchmarks/run_all.py) ·
Solutions: [`solutions/13-efficient-training.md`](../../solutions/13-efficient-training.md)

## G. Exercises

### Exercise 1 (by hand): formats, batches, break-even

1. bf16 has 8 significant bits. What is the spacing between representable numbers near 1.0? Near 0.01? A weight of 0.05 receives an update of 1e-5: is it stored?
2. nanochat d20: total batch 1 048 576 tokens, sequence length 2 048. Give the number of accumulation steps for (a) 8 GPUs with device batch 32, (b) 1 GPU with device batch 32, (c) 1 GPU with device batch 8. Which of these change the training result?
3. A model has 900 M parameters, 440 M of them in block matrices trained with a momentum-only optimizer, the rest with AdamW; master weights in fp32. Estimate the memory for weights + gradients + optimizer state.
4. Compiling takes 4 minutes and raises throughput from 20 000 to 26 000 tokens/s. A step is 524 288 tokens. After how many steps has compilation paid for itself?

**Success criterion:** numerical answers; item 1 can be checked with `torch.tensor(0.05, dtype=torch.bfloat16)`.

### Exercise 2 (experimental): formats

```bash
python labs/optimization/13_precision.py
python labs/optimization/13_master_weights.py
```

1. Answer the lesson's "predict before you run" question, then record both outputs.
2. In part 2 of the first script, change the increment from 1e-3 to 1e-2. Predict the bf16 result. Explain why it is neither 1.0 nor 11.0.
3. In the second script, why does a larger learning rate reduce the fraction of lost updates? Run with `--lrs 1e-4`: predict the bf16 row first.
4. nanochat stores its embedding tables in bf16 (lesson F), contradicting the rule you have just demonstrated. Give two reasons why that can nevertheless work. (Hint: look at the AdamW learning rate nanochat uses for embeddings in Module 07, and at how often a given row is updated.)
5. *(GPU only.)* `python labs/optimization/13_precision.py --device cuda`: record step time and peak memory for fp32 and bf16.

**Success criterion:** recorded tables and written answers to 2–4. Item 5: *awaiting validation on GPU, no reference values.*

### Exercise 3 (experimental): memory

```bash
python labs/optimization/13_grad_accum.py
```

1. Record the gradient difference and the memory table. Fit a line: peak memory ≈ a + b × tokens. What are a and b, and what does each represent?
2. You must train with an effective batch of 32 rows but only 600 MiB are available. Give two configurations that fit, using the table, and the cost of each in time.
3. Extend part 1 of the script to accumulate 32 micro-batches of 1 row. Is the gradient still equal? What would make it differ in a model with dropout or batch normalisation?
4. Checkpointing reduced memory by a factor of about 2.8, not by the number of layers (8). Why not more?

**Success criterion:** a fitted line, two feasible configurations, and an explanation for item 4.

### Exercise 4 (nanochat): read a training log like a performance engineer

Use the log of your Module 08 or Milestone 2 run (or produce one: `make nanochat-local-train`).

1. Find and record: `COMPUTE_DTYPE` and its reason, total batch size, device batch size, accumulation steps, FLOPs per token, and the steady-state `tok/sec`. `python scripts/experiments/parse_train_log.py <log>` extracts the per-step values.
2. Is the reported MFU meaningful on your device? Check `get_peak_flops` for your GPU's name.
3. Re-run the miniature training with `TORCHDYNAMO_DISABLE=1` and compare steady-state throughput and whole-command time. Compute the break-even number of steps on your machine.
4. *(GPU.)* Run a short probe (`make spark-probe` on the DGX Spark, or `--num-iterations 30` on any GPU) at device batch sizes 8, 16 and 32 with the total batch fixed. Record tokens/s and peak memory. Does tokens/s depend on the device batch size? Should it?

**Success criterion:** a filled record table. Items 2 and 4 on GPU: *awaiting validation, no reference values.*

### Exercise 5 (Milestone 3): the benchmark suite

```bash
make bench            # = python scripts/benchmarks/run_all.py
```

This runs a fixed set of comparisons (naive versus cached generation; naive versus fused
attention; fp32 versus bf16 forward; plain versus checkpointed training step; fp32
versus int8 weights) and writes `results/runs/benchmarks.jsonl` and a Markdown table.
For each row it reports a **correctness** column next to the speed or memory column.

1. Run it three times. Which numbers vary by more than 10 % between runs? What does that imply for a claim such as "A is 5 % faster than B"?
2. Add one comparison of your own (suggestions: GQA 4 → 1 key/value heads on cache bytes and decode speed; tile size in the tiled attention; compiled versus eager MiniGPT forward).
3. Write the Milestone 3 report using [`results/templates/experiment-record.md`](../../results/templates/experiment-record.md): for each optimisation, what it trades, whether it is exact, the measured effect on *your* hardware, and whether you would enable it.

**Success criterion:** see [Milestone 3](../../instructor/assessments.md#milestone-3).

### Challenge (optional): a gradient scaler from scratch

Train a small MiniGPT with all activations in fp16 (`torch.autocast(..., dtype=torch.float16)`
on a GPU, or emulate by casting the loss path on CPU). Show a configuration in which
gradients underflow to zero without scaling, then implement a minimal scaler: multiply
the loss by $S$ before `backward()`, divide gradients by $S$ before the step, skip the
step and halve $S$ when a gradient is non-finite, and grow $S$ slowly otherwise.
Compare with `torch.amp.GradScaler`.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Connect number formats to observable training behaviour; verify that accumulation is exact; quantify the memory/time trade of checkpointing; produce a reproducible benchmark table |
| **Hypotheses** | (H1) Storing weights in bf16 loses a large fraction of updates at small learning rates and measurably slows training; fp32 storage does not. (H2) Accumulated gradients equal the large-batch gradient to rounding. (H3) Peak step memory is linear in tokens per pass with a small intercept. (H4) Checkpointing cuts activation memory by more than half at the cost of 20–50 % time. |
| **Hardware / software** | CPU; course package. Exercise 4 needs the nanochat environment; GPU items are optional and unvalidated |
| **Tier** | A: each script 1–2 minutes on CPU; `make bench` about 1 minute. Exercise 4.4: tier B |
| **Metrics** | validation loss; fraction of unchanged weights; maximum gradient difference; peak process memory above idle (VmHWM); seconds per step; tokens/s |
| **Independent variables** | storage format; learning rate; batch rows; checkpointing on/off |
| **Controls** | same seed, data and step count; each memory measurement in a fresh process after a warm-up step; no other load on the machine |
| **Interpretation** | CPU memory figures include allocator behaviour and are reproducible to a few percent, not exactly. One seed per configuration: treat loss differences below 0.005 as noise (Module 10). |
| **Common failure modes** | Running the memory lab while another job runs (peak memory and time both distorted). Concluding from the 1e-2 rows that storage precision "does not matter". Timing compiled code including its first step. |

## I. Results and interpretation

Reference values (CPU, 4 cores, 2026-11): lesson D.1, D.2, D.6;
[`results/reference/benchmarks-cpu.md`](../../results/reference/benchmarks-cpu.md) for the suite.

| lr | stored as | val loss | unchanged weights | predicted ordering |
|---:|---|---:|---:|---|
| 1e-3 | fp32 / fp16 / bf16 | | | |
| 1e-2 | fp32 / fp16 / bf16 | | | |

| batch rows | tokens | plain MiB | s/step | checkpointed MiB | s/step |
|---:|---:|---:|---:|---:|---:|
| 2 | 512 | | | | |
| 8 | 2 048 | | | | |
| 32 | 8 192 | | | | |

| nanochat run | value |
|---|---|
| compute dtype (reason) | |
| total batch / device batch / accumulation steps | |
| steady-state tokens/s, compiled / not compiled | |
| break-even steps for compilation | |

**Questions to answer in writing**

1. Your training run is out of memory at device batch 32. List, in order, what you try, and what each changes about the experiment.
2. Why does a published wall-clock time for an 8-GPU run tell you the *number of steps* you need but not the *time* on your machine?
3. Which numbers in your Milestone 3 report would you expect to change on a GPU, and in which direction?
