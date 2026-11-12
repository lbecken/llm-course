# Module 12 — Lab: exact optimisations, approximate optimisations, and telling them apart

Scripts: [`labs/optimization/12_flash_sim.py`](../../labs/optimization/12_flash_sim.py) ·
[`labs/optimization/12_gqa_window.py`](../../labs/optimization/12_gqa_window.py) ·
[`labs/optimization/12_quantize.py`](../../labs/optimization/12_quantize.py) ·
Code under study: [`mlle/inference/flash_sim.py`](../../mlle/inference/flash_sim.py), [`mlle/inference/quantize.py`](../../mlle/inference/quantize.py) ·
Solutions: [`solutions/12-inference-optimization.md`](../../solutions/12-inference-optimization.md)

## G. Exercises

### Exercise 1 (by hand): online softmax, a quantized row, a throughput bound

1. One query with scores $(0, 2, 4, 1)$ and scalar values $(1, 2, 3, 4)$. Process it in two chunks of two with the online softmax of lesson D.2: give $(m, \ell, o)$ after each chunk and the output. Check against the direct computation.
2. Quantize the row $(0.50, -0.20, 0.04, -0.01)$ symmetrically to 3 bits. Give the scale, the codes and the reconstruction. Which weights are lost?
3. A model has 2.0 GB of weights and reads 0.1 GB of cache per decode step. The device's memory bandwidth is 100 GB/s. Give the bound on tokens/s at batch 1, and (assuming the weights are read once per step regardless of batch) at batch 16. What are you assuming about compute at batch 16?

**Success criterion:** item 1 matches `attention_tiled` with `block_k=2`; item 2 matches `quantize_tensor(..., bits=3, per_channel=False)`.

### Exercise 2 (experimental): the tiled attention

```bash
python labs/optimization/12_flash_sim.py
```

1. Record the table. At which $T$ does the Python tiled version become faster than the naive one on your machine? Why is there a crossover at all, when the tiled version does strictly more bookkeeping?
2. Read `attention_tiled`. Which line implements "skip tiles above the diagonal"? How many of the $(T/128)^2$ tiles are computed for $T=1024$?
3. Change the tile size to 32 × 32 and to 512 × 512 (the constant `BLOCK` in the lab script). Predict the effect on time and on the largest intermediate before running.
4. Make the scores extreme: multiply `q` by 100 before calling all three implementations. Do they still agree? Which quantity in the online softmax prevents overflow?

**Success criterion:** differences below 1e-4 in every configuration, and a written explanation of the crossover.

### Exercise 3 (experimental): what do GQA and windows cost?

```bash
python labs/optimization/12_gqa_window.py
```

1. Answer the lesson's "predict before you run" question, then record the table.
2. The window-8 result is one seed. Run that variant with two more seeds (add `seed=1`, `seed=2` to its `TrainConfig`). Is the agreement drop robust?
3. Add a variant `dict(window=8, n_layer=4)`. Predict: does agreement recover? What does that tell you about *depth × window* as the quantity that matters? Compare its loss with a full-context 4-layer model (`dict(n_layer=4)`): is the relay free?
4. For nanochat d12 with pattern `SSSL` at a context of 2 048: how many cached positions does a decode step read, summed over the 12 layers, compared with pattern `L`?

**Success criterion:** a table with at least 9 rows and a one-paragraph recommendation: for which task properties is a short window safe?

### Exercise 4 (experimental): how many bits?

```bash
python labs/optimization/12_quantize.py
```

1. Record the table and mark the "cliff" for per-tensor and per-channel scales.
2. Write 15 lines that quantize only one *group* of matrices at 3 bits per channel, leaving the others in fp32: (a) `wte` and `lm_head`, (b) the attention matrices, (c) the MLP matrices. Predict which group is most sensitive *per parameter*, then measure.
3. From item 2, propose a mixed-precision scheme (bits per group) for this model and state its total size.
4. The lab uses "fake quantization". List two things a real int8 deployment needs that the lab does not exercise.

**Success criterion:** the three-group measurement and a mixed scheme justified by it.

### Exercise 5 (nanochat): which kernel, which window, what ceiling?

```bash
cd external/nanochat
python -c "import nanochat.flash_attention as f; print('HAS_FA3', f.HAS_FA3, 'USE_FA3', f.USE_FA3)"
python -m pytest tests/test_attention_fallback.py -q
```

1. Record which back end is active on your machine and how many tests ran versus were skipped. What do the skipped tests need?
2. Evaluate `-(-T // 4 // 128) * 128` for $T$ = 2 048, 1 024 and 256. For a sequence length of 256 (the miniature runs of Module 08), what are the windows of a depth-4 model with pattern `SSSL`?
3. Compute by hand, for d12 (793 MB of stored weights, 36 864 cache bytes per token, `SSSL`, context 1 152): the bytes read per decode step and the batch-1 ceiling on (a) an H100 (3.35 TB/s in nanochat's table) and (b) a device with 273 GB/s. State clearly that these are bounds, not predictions.
4. *(GPU only.)* `python -m scripts.infer_bench -i base -g <tag> --batch-sizes 1,8,32`. Record the table. At which batch size does tokens/s stop scaling linearly? On an unknown GPU, which columns are meaningless and why?

**Success criterion:** written answers to 1–3. Item 4: *awaiting validation on GPU, no reference values.*

### Challenge (optional): greedy speculative decoding

Train two MiniGPTs on Pocket: a small "draft" (1 layer) and a larger "target" (4 layers).
Implement greedy speculative decoding: the draft proposes $k$ tokens; the target scores
the proposal in **one** forward pass; accept the longest prefix on which the target's
arg-max agrees, then take the target's own token at the first disagreement. Verify that
the output equals the target's greedy output token for token, and report the acceptance
rate and the number of target forward passes saved as a function of $k$. Explain why the
sampling version needs an acceptance *rule* rather than an equality test.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Verify one exact optimisation by equality; quantify the quality cost of three approximate ones |
| **Hypotheses** | (H1) Tiled attention equals naive attention to rounding at every $T$; its largest intermediate does not grow with $T$. (H2) Reducing key/value heads from 4 to 1 does not measurably change validation loss on Pocket. (H3) A window shorter than half the agreement dependency (two layers) breaks agreement while changing the loss by less than 0.02. (H4) 8-bit weights change the loss by less than 0.002; per-channel scales tolerate about two bits fewer than per-tensor scales. |
| **Hardware / software** | CPU; course package. Exercise 5 needs the nanochat environment |
| **Tier** | A: 12a ≈ 30 s, 12b ≈ 3 min, 12c ≈ 1 min on CPU |
| **Metrics** | max absolute output difference; time; bytes of the largest intermediate; parameters; cache bytes per token; validation loss; agreement accuracy with attractor; relative weight error |
| **Independent variables** | sequence length and tile size; number of key/value heads; window; bit-width and scale granularity |
| **Controls** | same data, steps, learning rate and seed across variants; quantized models are copies of one trained model evaluated on the same batches |
| **Interpretation** | Differences in validation loss below about 0.005 are inside seed noise for this setup (Module 10). A single-seed capability result needs replication (Exercise 3.2). CPU timings say nothing about GPU kernels. |
| **Common failure modes** | Calling an approximate technique "lossless" from an average metric. Timing the Python tiled loop and concluding FlashAttention is slow. Comparing quantized variants across different trained models. |

## I. Results and interpretation

Reference values (CPU, 4 cores, 2026-11): lesson D.2–D.6 and the solutions file.

| $T$ | max diff | naive ms | tiled ms | fused ms | score matrix | tile |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | | | | | | |
| 1 024 | | | | | | |
| 4 096 | | | | | | |

| Variant | params | KV bytes/token | val loss | agreement | predicted agreement |
|---|---:|---:|---:|---:|---|
| baseline | | | | | |
| GQA 2 / GQA 1 | | | | | |
| window 32 / 16 / 8 | | | | | |
| window 8, seeds 1 and 2 | | | | | |
| window 8, 4 layers | | | | | |

| Bits | per-tensor Δ loss | per-channel Δ loss |
|---:|---:|---:|
| 8 | | |
| 6 | | |
| 4 | | |
| 3 | | |

**Questions to answer in writing**

1. Sort every technique of this module into "exact" and "approximate", and for each approximate one name the test that would reveal its typical failure.
2. Why is `--window-pattern L` recommended on a machine without Flash Attention 3, although `SSSL` does less arithmetic?
3. A vendor claims "4-bit quantization with no accuracy loss". Which three questions do you ask?
