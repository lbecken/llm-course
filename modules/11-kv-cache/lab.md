# Module 11 — Lab: a cache that is correct first, then fast

Scripts: [`labs/optimization/11_kv_cache.py`](../../labs/optimization/11_kv_cache.py) ·
[`labs/optimization/11_nanochat_engine.py`](../../labs/optimization/11_nanochat_engine.py) ·
Code under study: [`mlle/inference/kvcache.py`](../../mlle/inference/kvcache.py), [`mlle/foundations/minigpt.py`](../../mlle/foundations/minigpt.py) ·
Solutions: [`solutions/11-kv-cache.md`](../../solutions/11-kv-cache.md)

## G. Exercises

### Exercise 1 (by hand): sizing a cache

nanochat d20 has 20 layers, 10 heads of dimension 128 (no GQA) and caches in bf16.

1. Bytes per cached token? Megabytes for one 2 048-token conversation?
2. A serving machine has 24 GB of GPU memory; the d20 weights occupy about 1.8 GB in bf16. Ignoring every other overhead, how many full-context conversations fit at once?
3. The same model with `n_kv_head = 2` (GQA): redo items 1 and 2.
4. The naive generator needs no cache. How many positions does it process to generate 1 000 tokens after a 1 000-token prompt? How many does the cached generator process?

**Success criterion:** item 1 matches the table in lesson D.2; item 4 within 1 % of the exact sum.

### Exercise 2 (experimental): measure it

```bash
python labs/optimization/11_kv_cache.py
```

1. Answer the "predict before you run" question of the lesson, then record the table.
2. Plot or tabulate *tokens per second* against $N$ for both generators. Which is roughly constant, and why is it not exactly constant?
3. Re-run with a larger model: `--n-layer 8 --n-embd 256`. Predict first: does the speed-up at $N=512$ grow or shrink compared with the small model? Explain the result in terms of fixed per-step overhead versus arithmetic.
4. If you have a GPU, run with `--device cuda`. The cached tokens/s may be *lower* than you expect, and the naive generator may be surprisingly competitive at small $N$. Why?

**Success criterion:** `max |logit diff| < 1e-4`, identical tokens, and a table for two model sizes with a written explanation of the trend.

### Exercise 3 (coding): break it on purpose

Each of these is a real bug class. Introduce them one at a time in a scratch copy of the
course code (or by monkey-patching), run part 1 of the lab or
`pytest tests/test_minigpt.py -k cached`, observe, then revert.

1. **Wrong positions.** In `CausalSelfAttention.forward` replace `offset=pos0` with `offset=0`. Predict: which positions' logits stay correct? Does the output look obviously broken?
2. **Advance per layer.** Move `cache.advance(T)` so that it runs after *every* layer instead of once. What happens, and at which layer does it first go wrong?
3. **Forgetting the mask in multi-token appends.** Replace the mask in the cached path with all-True. Which of the two operations is still correct: single-token decode, or a 7-token prefill? Why?

**Success criterion:** for each bug, a sentence stating which test catches it and what the symptom is. A cache bug that no test catches is the one that matters; note any you find.

### Exercise 4 (nanochat): the engine

Needs the miniature checkpoint from Module 08 (`make nanochat-local-data nanochat-local-train`).

```bash
export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local
python labs/optimization/11_nanochat_engine.py --model-tag pocket-d4
cd external/nanochat && python -m pytest tests/test_engine.py -q
```

1. Record `Match`, both times and the cache shape. Map each of the five dimensions of `k_cache` to a model quantity. Which two axes are swapped relative to the course's `KVCache`, and why did nanochat choose that order?
2. In [`Engine.generate`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py#L176-L275), find the two `KVCache(...)` constructions. What are their batch sizes and sequence lengths, and why are two caches used instead of one?
3. `KVCache.get_pos()` returns `cache_seqlens[0]`. State the assumption this makes. Give a serving scenario in which it would be false, and what would have to change.
4. Read `test_multi_sample_first_token_diversity` in `tests/test_engine.py`. What bug does it guard against, and why would that bug be invisible in a single-sample test?
5. *(GPU only.)* Run `python -m scripts.infer_bench --model-tag <tag> --decode-batch-sizes 1,4,16`. Record TTFT, TPOT and tokens/s per batch size. Does total throughput scale linearly with batch size?

**Success criterion:** `Match: True`, the engine tests pass, and written answers to items 1–4. Item 5: *awaiting validation on GPU, no reference values.*

### Challenge (optional): prefix sharing

Many requests to a deployed assistant start with the same long system prompt. Implement
`generate_many(model, prefix_ids, list_of_suffix_ids, ...)` for MiniGPT that prefills the
shared prefix **once**, then serves each suffix from a copy of that cache (suffixes have
different lengths: decide how to handle that without padding bugs). Verify token-exact
agreement with running each full prompt separately, and measure the saving as a function
of prefix length. State what must be true of the tokenisation for a prefix cache to be
valid.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Establish that cached generation is exact, measure its memory, and measure how its advantage depends on sequence length and model size |
| **Hypotheses** | (H1) Cached and full logits agree to rounding; greedy tokens are identical. (H2) Allocated cache bytes equal $L\cdot2\cdot H_{kv}\cdot D\cdot$ bytes/number per token. (H3) The speed-up grows with $N$. (H4) The speed-up is far below the ratio of positions processed. (H5) It is closer to that ratio for a larger model. *(H5 was the author's expectation and was not confirmed on the reference machine; see the solutions.)* |
| **Hardware / software** | CPU is sufficient; course package; Exercise 4 needs the nanochat environment (`make env-cpu`) |
| **Tier** | A: about 1 minute per script on CPU |
| **Metrics** | maximum absolute logit difference; token equality; cache bytes; wall time (median of 3 after 1 warm-up); tokens/s; positions processed |
| **Independent variables** | number of generated tokens; model size; device |
| **Controls** | same weights, prompt and greedy decoding for both generators; nothing else running on the machine (timings on a shared CPU vary by tens of percent) |
| **Interpretation** | Report the speed-up *with* $N$ and model size; a single "KV caching is $x$ times faster" is not a meaningful statement. |
| **Common failure modes** | Timing without warm-up. On GPU, timing without synchronisation (the harness does it for you). Comparing sampled rather than greedy outputs and concluding the cache is wrong. Generating past `block_size` (the cache has a fixed capacity; the lab stays inside it). |

## I. Results and interpretation

Reference values (CPU, 4 cores, 2026-11): lesson D.5. Your absolute times will differ; the pattern should not.

| $N$ | naive s | cached s | speed-up | naive tok/s | cached tok/s | your prediction of speed-up |
|---:|---:|---:|---:|---:|---:|---|
| 32 | | | | | | |
| 128 | | | | | | |
| 512 | | | | | | |

| Model | params | bytes/token (formula) | allocated for 1 024 tokens | speed-up at 512 |
|---|---:|---:|---:|---:|
| 4 layers × 128 | | | | |
| 8 layers × 256 | | | | |

| nanochat miniature | value |
|---|---|
| `Match` | |
| naive / Engine time for 64 tokens | |
| `k_cache` shape | |
| `kv_bytes_per_token()` | |

**Questions to answer in writing**

1. Which line of the cached forward pass would you check first if a model produced good text for the first sentence and nonsense afterwards?
2. Why does the cache make *serving* memory planning depend on the number of concurrent users, when training memory did not?
3. The cached keys in the course implementation are stored *after* RoPE rotation and QK-norm. Why is that correct, and what would go wrong if they were stored before rotation and re-rotated at read time with the reader's position?
