# Module 08 — Lab: from documents to a trained base model

Scripts: [`labs/pretraining/08_packing.py`](../../labs/pretraining/08_packing.py) ·
[`labs/pretraining/08_data_quality.py`](../../labs/pretraining/08_data_quality.py) ·
[`scripts/experiments/make_local_dataset.py`](../../scripts/experiments/make_local_dataset.py) ·
[`scripts/experiments/nanochat_local.sh`](../../scripts/experiments/nanochat_local.sh) ·
[`scripts/experiments/parse_train_log.py`](../../scripts/experiments/parse_train_log.py) ·
Solutions: [`solutions/08-data-and-pretraining.md`](../../solutions/08-data-and-pretraining.md)

## G. Exercises

### Exercise 1 (conceptual): design the splits

You are given 2 million customer-support conversations from 40 000 customers, collected
over three years, and asked to build a model that will answer *next* year's customers.

1. Propose train/validation/test splits. Should conversations be assigned at random? Give two reasons a random split would overestimate real performance.
2. The same boilerplate greeting and signature appear in 60 % of conversations. What does that do to the validation loss, and does deduplicating whole conversations fix it?
3. After three months of tuning against your validation set you report its loss as the system's quality. What is wrong, and what should you have kept back?

**Success criterion:** an answer that names *group* leakage and *temporal* leakage explicitly.

### Exercise 2 (experimental): packing

```bash
python labs/pretraining/08_packing.py
python labs/pretraining/08_packing.py --median 1500 --sigma 1.2
python labs/pretraining/08_packing.py --median 60 --sigma 0.6     # short, uniform documents (like chat turns)
```

1. Predict before each run which strategy wastes least at $T = 2048$.
2. For the long-document setting, what fraction of *compute* does `bestfit_pad` waste, and what fraction of *data* does `bestfit_crop` waste? Which resource is scarcer in pretraining? In fine-tuning?
3. In `simulate_packing`, when no buffered document fits, the *shortest* one is cropped. Change it to crop the *first* one in the buffer (a one-line edit) and measure the cropped fraction at $T = 2048$ for the long-document setting. Explain the difference.

**Success criterion:** a table for three length distributions and a justified choice of strategy for each.

### Exercise 3 (experimental): contamination

```bash
python labs/pretraining/08_data_quality.py
```

1. Reproduce the table. Which single number proves leakage without any reference to the training data?
2. Lower the leaked share from 50 % to 5 % (edit `leaked_val`). Is the memorising model's contaminated-validation loss still visibly better than its clean one? Does the shingle detector still find the leaked documents?
3. Make the leak a *paraphrase*: replace one word in each leaked document. What happens to exact-duplicate detection, to the shingle overlap, and to the memorising model's loss?

**Success criterion:** you can state what each of the two defences (loss sanity check, overlap detection) catches and misses.

### Exercise 4 (nanochat, CPU): the miniature pipeline

```bash
make nanochat-local-data nanochat-local-train nanochat-local-eval
python scripts/experiments/parse_train_log.py <your log file> --depth 4
```

(Redirect the training output to a file with `... 2>&1 | tee results/runs/pocket-d4.log`.)

1. Find in your log: vocabulary size; parameter counts by group; FLOPs per token; the three "Scaling …" lines; gradient-accumulation steps; the validation bpb at steps 0, 100, 200, 300.
2. Compare your final bpb with the entropy floor of 0.4528 bits per character. Why is this comparison legitimate here (think: bytes per character, and which tokens `evaluate_bpb` excludes)?
3. Open the saved `meta_000300.json`. Which fields would you need to reproduce the run exactly?
4. Run again with `TORCHDYNAMO_DISABLE=1`. Compare total wall time and steady-state tokens/s. For a 300-step run, was compilation worth it? At what run length does it pay off?

**Success criterion:** bpb at step 300 below 0.50; all items located.

### Exercise 5 (nanochat, CPU): a controlled data experiment

Using the same commands, compare three training sets for the same model and steps:

| Condition | How to create it |
|---|---|
| A: 12 000 documents (default) | `make nanochat-local-data` |
| B: 600 documents | `python scripts/experiments/make_local_dataset.py --num-docs 600 --row-group-size 64` |
| C: 600 documents, then 3 000 steps | as B, with `STEPS=3000 scripts/experiments/nanochat_local.sh train` |

Use a separate `NANOCHAT_BASE_DIR` per condition and retrain the tokenizer in each.
Predict the validation bpb ordering and what the `epoch` field will show. Record train
bpb (from `base_eval`) and validation bpb. Which condition overfits, and how can you
tell from the log alone?

**Success criterion:** a table of train/validation bpb for A, B, C, and the epoch counts.

### Exercise 6 (GPU, Tier B–C; Milestone 2 part 1): your first real base model

Follow "Scaling up to real data" in the lesson: probe, then train depth 6 (or the largest
depth whose probe estimate fits your patience) with `--window-pattern=L`. Save the log.
Record in an [experiment record](../../results/templates/experiment-record.md): hardware,
commands, tokens/s, peak memory, wall time, final validation bpb, three samples.

**Success criterion:** a checkpoint that `scripts.base_eval` can load, a parsed log, and a completed record. *Awaiting validation: the author has not run this on a GPU.*

### Challenge (optional): what did cropping remove?

Write a script that, for the local dataset, measures the average *position within its
document* of the tokens that reach training under `bestfit_crop` versus `stream`. Confirm
the log's remark that cropping "will skew the data distribution" towards document
beginnings, and quantify it for $T = 128$ and $T = 512$.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Understand the cost of each packing strategy; observe and detect contamination; run the complete nanochat pretraining pipeline and read its log; (GPU) train a first real base model |
| **Hypotheses** | (H1) Best-fit cropping discards more data as documents get longer relative to $T$; padding wastes more positions. (H2) A memorising model scores below the entropy floor on leaked documents. (H3) nanochat on the local dataset reaches < 0.50 bpb in 300 steps on CPU. (H4) With too little data the validation bpb stops improving or rises while the training bpb keeps falling. |
| **Hardware / software** | Ex. 2–3: CPU, course package. Ex. 4–5: CPU, `make env-cpu`. Ex. 6: GPU, [DGX Spark setup](../../docs/dgx-spark-setup.md) |
| **Tier** | Ex. 2–3: A (seconds). Ex. 4: A (≈ 2 min). Ex. 5: A (≈ 6 min). Ex. 6: B–C (measure first) |
| **Inputs** | synthetic length distributions; Pocket documents; (Ex. 6) ClimbMix shards |
| **Metrics** | cropped / padded fractions; validation loss; bits per byte (train and validation); tokens/s; wall time; peak memory (GPU) |
| **Independent variables** | packing strategy and length distribution; leaked share; dataset size and number of steps |
| **Controls** | model configuration, seed (nanochat fixes 42), tokenizer settings, evaluation tokens |
| **Interpretation** | Compare bpb with the floor 0.4528. A train–validation gap that grows with steps indicates overfitting; `epoch > 1` in the log confirms data is being reused. |
| **Common failure modes** | Mixing base directories (tokenizer/model vocabulary mismatch). `No dataset parquet files found`: wrong `NANOCHAT_BASE_DIR`. Tokenizer trained on one dataset and model on another. Forgetting that the first step includes ~40 s of compilation. |

## I. Results and interpretation

Reference values (author, CPU): see the lesson (D.2, D.3) and
[results/reference/nanochat-local-cpu.md](../../results/reference/nanochat-local-cpu.md).

| Exercise 4 | reference | yours |
|---|---:|---:|
| vocabulary size | 353 | |
| validation bpb @ 0 / 100 / 200 / 300 | 1.844 / 0.565 / 0.476 / 0.466 | |
| median tokens/s (compiled) | 40.6 k | |
| total wall time (compiled / not compiled) | 80 s / 24 s | |

| Exercise 5 | train bpb | validation bpb | epochs seen |
|---|---:|---:|---:|
| A: 12 000 docs, 300 steps | | | |
| B: 600 docs, 300 steps | | | |
| C: 600 docs, 3 000 steps | | | |

| Exercise 6 (GPU) | your measurement |
|---|---|
| depth / window pattern / device batch size | |
| probe: tokens/s, peak memory | |
| full run: wall time, final validation bpb | |

**Questions to answer in writing**

1. Which of your numbers are comparable with nanochat's published leaderboard values, and which are not? Give the reason for each.
2. In Exercise 5, which single log field would have warned you of overfitting before looking at validation bpb?
