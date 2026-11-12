# Module 06 — Lab: trace it, count it, ablate it, reproduce nanochat

Scripts: [`labs/foundations/06_transformer_trace.py`](../../labs/foundations/06_transformer_trace.py) ·
[`labs/foundations/06_nanochat_parity.py`](../../labs/foundations/06_nanochat_parity.py) (needs `make env-cpu`) ·
Code under study: [`mlle/foundations/minigpt.py`](../../mlle/foundations/minigpt.py), [`mlle/nanochat_bridge.py`](../../mlle/nanochat_bridge.py) ·
Solutions: [`solutions/06-transformer-block.md`](../../solutions/06-transformer-block.md)

## G. Exercises

### Exercise 1 (mathematical): parameters by hand

For a GPT-2-style model with $V = 1000$, $T_\text{max} = 128$, $L = 4$, $C = 64$, 4 heads:

1. Compute the number of parameters of each component and the total, using the table in lesson D.6.
2. Verify with `MiniGPT(MiniGPTConfig.gpt2_style(1000, block_size=128, n_layer=4, n_head=4, n_embd=64)).num_params()`.
3. The same configuration in nanochat style (`MiniGPTConfig.nanochat_style`): predict the total, then check.
4. Double $C$ to 128. Which components grow 2×, which 4×?

**Success criterion:** your hand totals equal the program's exactly.

### Exercise 2 (tracing): predict every shape

```bash
python labs/foundations/06_transformer_trace.py
python labs/foundations/06_transformer_trace.py --style gpt2
```

1. Before running, write the shape after each of the ten stages for an input of 16 tokens, `C=48`, `H=4`, `V=29`.
2. Explain the "residual stream" printout: why is the relative change exactly 0.000 for the nanochat style and not for the GPT-2 style?
3. Modify the script to print the same residual-stream table for a model *trained* for 300 steps on Pocket (reuse `train_on_text`). Do later blocks change the stream more or less than earlier ones?

**Success criterion:** all ten shapes predicted correctly; the 0.000 explained by naming the two zero-initialised matrices.

### Exercise 3 (experimental): one switch at a time

Train character-level MiniGPTs on Pocket (10 000 sentences, 600 steps, seed 0, then seeds 1
and 2) for these configurations, starting from `gpt2_style` and changing **one** thing:

| Variant | Override |
|---|---|
| baseline | — |
| no residual connections | (requires a 2-line code change in `Block.forward`: drop the `x +`) |
| post-norm | (code change: `x = self.norm1(x + self.attn(x))`, same for the MLP) |
| ReLU² | `act="relu2"` |
| RMSNorm | `norm="rmsnorm"` |
| no MLP | (code change: skip the MLP line) |

Predict the ranking before running. Record final validation loss for three seeds each.

**Success criterion:** a table with 3 seeds per row; conclusions only where the difference exceeds the seed spread.

### Exercise 4 (nanochat): reproduce and dissect

```bash
python labs/foundations/06_nanochat_parity.py
make test-nanochat
```

1. Record the parity error and the logit change caused by each extra.
2. In `mlle/nanochat_bridge.py::minigpt_from_nanochat`, comment out the line that copies `c_k`. Predict whether the parity test still passes; run it; restore the line.
3. Change `rope_base=100000.0` to `10000.0` in `MiniGPTConfig.nanochat_style` (temporarily). Does parity survive? Why does the size of the error depend on the sequence length?
4. Write a new check: enable **only value embeddings** in nanochat (fill them with noise instead of zero) and reproduce nanochat's logits by adding value-embedding support to a *copy* of MiniGPT's attention (about 10 lines; use the nanochat code in lesson D.7 item 2).

**Success criterion:** items 1–3 answered with measurements; item 4 reaches < 1e-4.

### Challenge (optional): find the function of an addition

Using the miniature nanochat pipeline (`make nanochat-local-data`, then
`scripts/experiments/nanochat_local.sh train`), train twice for 300 steps: once
unmodified and once with **value embeddings disabled**. Without editing nanochat, you can
disable them from a small wrapper script that imports `scripts.base_train`'s pieces, or
more simply by copying `external/nanochat` to a scratch directory and making
`has_ve` return `False` there (never modify the pinned checkout itself). Compare
validation bpb over three seeds (`torch.manual_seed` in `compute_init`). Does Pocket,
with its 353-token vocabulary, benefit? Would you expect the answer to transfer to a
32 768-token vocabulary? Why or why not?

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Confirm the forward-pass description by shape tracing, parameter counting and numerical reproduction of nanochat; measure the contribution of individual architectural choices |
| **Hypotheses** | (H1) Hand-computed parameter counts equal the model's. (H2) With the five additions neutralised, MiniGPT reproduces nanochat's logits to < 1e-4; re-enabling any one breaks parity. (H3) Removing residual connections or the MLP hurts far more than swapping the activation or the normalization. |
| **Hardware / software** | CPU. Exercises 1–3: course package. Exercise 4: `make env-cpu`. |
| **Tier** | A. Exercise 3 is the longest: 6 variants × 3 seeds × ~15 s. Run sequentially. |
| **Metrics** | parameter counts; max absolute logit difference; validation loss (nats/char) |
| **Independent variables** | one architectural switch at a time |
| **Controls** | data, seed set, steps, optimizer settings, model size |
| **Interpretation** | For Exercise 3, compare the difference between variants with the spread across seeds. An ablation that changes initialisation behaviour (e.g. no residuals) may need a different learning rate to be fair; note this as a limitation rather than tuning each variant separately. |
| **Common failure modes** | Forgetting to restore temporary code edits (run `git diff` and `make test` afterwards). Parity "passes" trivially if matrices are zero: the lab randomises them for this reason. |

## I. Results and interpretation

Reference (author, CPU, 2026-11-11): parity error 4.05e-06 with largest |logit| 5.75;
logit changes when re-enabling one addition: resid 1.37, x0 1.44, smear 4.09 (with
`smear_lambda=0.5`), backout 0.74, value embeddings 0.47.

Reference for Exercise 3 (author, CPU, 2026-11-12; 2 layers, 2 heads, `n_embd=64`, 600 steps, validation loss in nats/char):

| Variant | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| baseline (GPT-2 style) | 0.3664 | 0.3722 | 0.3706 |
| no residual | 2.7497 | 2.5921 | 2.7498 |
| post-norm | 0.3988 | 0.3999 | 0.3944 |
| ReLU² | 0.3567 | 0.3550 | 0.3549 |
| RMSNorm | 0.3718 | 0.3750 | 0.3749 |
| no MLP | 0.4300 | 0.4374 | 0.4743 |

Your measurements:

| Variant (Exercise 3) | val loss seed 0 | seed 1 | seed 2 | mean | vs. baseline |
|---|---:|---:|---:|---:|---:|
| baseline (GPT-2 style) | | | | | |
| no residual | | | | | |
| post-norm | | | | | |
| ReLU² | | | | | |
| RMSNorm | | | | | |
| no MLP | | | | | |

**Questions to answer in writing**

1. Which ablations produced differences larger than the seed spread?
2. The 2-layer model without residual connections fails at this learning rate although (as Module 07, Exercise 5 measures) its gradients do not vanish at initialisation. At lower learning rates it trains, slowly. Which of the two arguments for residual connections (forward: "propose an edit"; backward: "identity path for the gradient") do these observations speak to, and what would you measure next?
3. What exactly does the parity result prove, and what does it not prove (think: training dynamics, performance, the kernels used on a GPU)?
