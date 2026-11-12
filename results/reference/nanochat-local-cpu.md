# Reference result: unmodified nanochat on a miniature local dataset (CPU)

| | |
|---|---|
| Date | 2026-11-11 |
| Hardware | 4-vCPU x86-64 sandbox, 15 GiB RAM, **no GPU** |
| Software | Python 3.10.20, torch 2.9.1+cpu, nanochat at `92d63d4` (environment from `uv sync --extra cpu --group dev`) |
| Status | **Measured** by the course author. Single run, seed fixed by nanochat (`torch.manual_seed(42)`). |

## Commands

```bash
export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local
python scripts/experiments/make_local_dataset.py            # 4 shards x 3000 Pocket documents, 3,045,678 chars
(cd external/nanochat && python -m scripts.tok_train --vocab-size=512 --max-chars=2000000)
(cd external/nanochat && python -m scripts.tok_eval)
scripts/experiments/nanochat_local.sh train                 # depth 4, n_embd 64, 300 steps
scripts/experiments/nanochat_local.sh eval
```

(The measured run used `--eval-tokens=20480`, which is what the script passes.)

## Results

**Tokenizer.** Training stopped after 88 merges (0.12 s): Pocket has only a few dozen distinct
words, so once every word chunk is a single token there is nothing left to merge. Final
vocabulary: 256 bytes + 88 merges + 9 special tokens = 353. `tok_eval` reported 4.59
bytes/token on the validation shard, against 3.91 for GPT-2's and 4.42 for GPT-4's
tokenizer on the same text, and only 1.0–1.2 bytes/token on its out-of-domain samples
(news, code, Korean): a tokenizer is only efficient on text that resembles its training data.

**Pretraining** (`base_train.py`, compiled, fp32, SDPA attention, `--window-pattern L`):

| step | validation bits per byte |
|---:|---:|
| 0 | 1.8437 |
| 100 | 0.5652 |
| 200 | 0.4763 |
| 300 | 0.4664 |

- Source entropy of Pocket: **0.4528 bits per character** (`mlle.data.pocket.entropy_rate()`),
  and one character is one byte here. The model ended 3 % above the theoretical floor.
- Train bpb 0.4644 vs. validation bpb 0.4664: no overfitting, as expected with fresh samples.
- Throughput after warm-up: median 40.6 k tokens/s (range 22.6–54.8 k), median step 50 ms.
- Wall time for the whole command: 80 s, of which the 300 training steps took about 15 s.
  The rest is start-up and `torch.compile` (the first step alone took 41 s).
- Log line `bf16_mfu: 0.00` and `Peak memory usage: 0.00MiB`: both are meaningless on CPU.
- nanochat's automatic scaling printed: learning rates × 0.0625 for the 2 048-token batch
  (reference 524 288), AdamW learning rates × 3.46 for width 64 (reference 768), weight
  decay 0.28 → 0.426.

**Same run with `TORCHDYNAMO_DISABLE=1`** (no compilation): validation bpb 1.8437 →
0.5548 → 0.4736 → 0.4649; median 31.6 k tokens/s; whole command 24 s. For this tiny
model, compilation bought 28 % steady-state throughput and cost about a minute.

**Samples** from `base_eval.py` (temperature 1.0, unconditioned):

```text
the birds wait .
the robots see the cats .
the red child waits .
the birds that chase the children chase the children .
the children that see the dogs wait .
the small robots robots wait .
```

Agreement across relative clauses is right in the samples inspected; the last line shows
a remaining error (a repeated noun, which the grammar never produces).

## Overfitting, with the real pipeline

Same commands on a dataset of only 600 documents
(`python scripts/experiments/make_local_dataset.py --num-docs 600 --row-group-size 64`,
separate `NANOCHAT_BASE_DIR`, tokenizer retrained), measured 2026-11-12:

| Condition | train bpb | validation bpb | `epoch` at the end |
|---|---:|---:|---:|
| 12 000 documents, 300 steps (the run above) | 0.4644 | 0.4664 | 2 |
| 600 documents, 300 steps | 0.4453 | 0.4851 | 28 |
| 600 documents, 3 000 steps (`STEPS=3000`) | 0.3393 | 1.5932 | 271 |

With 3 000 steps on 600 documents the training loss falls far below the entropy of the
language (0.4528), which is only possible by memorising, and the validation loss rises
from its best value of 0.475 (step 200) to 1.59.

## What this does and does not show

- It shows that the pinned pipeline (data format, tokenizer training, pretraining,
  bits-per-byte evaluation, checkpointing, cached sampling through `Engine`) runs
  unmodified on CPU, and that its loss can be compared with a known optimum.
- It says nothing about GPU throughput, memory, bf16, FP8, Flash Attention 3 or
  multi-GPU behaviour.
