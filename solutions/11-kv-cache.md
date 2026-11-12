# Solutions — Module 11

## Lab exercises

### Exercise 1: sizing a cache

1. $20 \times 2 \times 10 \times 128 \times 2 = 102\,400$ bytes per token. For 2 048 tokens: 209 715 200 bytes = **209.7 MB** (200 MiB).
2. $(24 - 2.7)\ \text{GB} / 0.2097\ \text{GB} = 101.6$, so **101** conversations. The weights are shared by all of them; the caches are not. (The d20 weight figure: 461 M embedding and value-embedding parameters × 2 bytes + 435 M matrix parameters × 4 bytes = 2.66 GB.)
3. With two key/value heads: $20\times2\times2\times128\times2 = 20\,480$ bytes per token, 41.9 MB per conversation, **507** conversations. A fivefold reduction in the quantity that limits concurrency, at a quality cost that must be measured (Module 12 does, on a small model).
4. Naive: step $i$ (from 0) processes $1000 + i$ positions: $\sum_{i=0}^{999}(1000+i) = 1\,000\,000 + 499\,500 = $ **1 499 500**. Cached: the prompt once, then each sampled token fed back except the last: $1000 + 999 = $ **1 999**. Ratio 750.

A wrong approach worth naming: sizing the cache from the *parameter count*. The cache does
not depend on the MLP, the vocabulary or the embedding tables at all; it depends on
depth, key/value width, context length and batch.

### Exercise 2: measure it

Reference (CPU, 4 cores):

| Model | $N$ | naive | cached | speed-up | naive tok/s | cached tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 4 × 128 (0.9 M) | 32 | 0.068 s | 0.054 s | 1.3× | 467 | 596 |
| | 128 | 0.522 s | 0.268 s | 1.9× | 245 | 478 |
| | 512 | 5.497 s | 0.803 s | 6.8× | 93 | 637 |
| 8 × 256 (6.6 M) | 32 | 0.173 s | 0.107 s | 1.6× | 185 | 298 |
| | 128 | 1.196 s | 0.408 s | 2.9× | 107 | 314 |
| | 512 | 12.529 s | 2.023 s | 6.2× | 41 | 253 |

1. Closer to 7×. The cached generator does 260 times fewer position-computations, but each of its 512 steps is a separate trip through the whole network for *one* position; the naive generator's passes are few and large.
2. The cached generator is roughly constant. Not exactly: each step's attention reads a cache that grows by one token (the $LCt$ term of D.5), and on a shared CPU timings fluctuate.
3. The author's own prediction was that the larger model would show a larger speed-up (less dominated by Python overhead). **Measured: about the same (6.2× versus 6.8×).** Dividing the times: one decode step costs as much as about 40 positions inside a batched pass, for both sizes (1.57 ms versus 41 µs; 3.95 ms versus 93 µs). The naive pass is not merely "less overhead per position": it runs large matrix products that use all cores and vectorise well, whereas a single-position step is a sequence of small matrix-vector products that do not. That factor of about 40 is a property of this machine and these sizes, and it did not shrink with the model. The asymptotic argument still holds (the speed-up grows with $N$ without bound); the constant is what you measure.
4. On a GPU a decode step is dominated by launch overhead of many small kernels, not arithmetic, so cached tokens/s for a tiny model can be *lower* than on CPU. And the naive generator's big passes are precisely what a GPU does well, so for short sequences it is competitive. The cache wins as $N$ grows, and for real model sizes; batching the decode (Exercise 4.5) is how the GPU's capacity is actually used. *No reference values: not run on GPU by the author.*

### Exercise 3: break it on purpose

Measured on the lab model (random weights, logit magnitudes up to 4.7):

1. **Wrong positions** (`offset=0`). The 16 prefilled positions stay exact (difference 0): in prefill the true offset *is* 0. Every decoded position is wrong: maximum logit difference 2.6, and the arg-max token agrees with the correct model at only 45 % of positions. Nothing crashes and the output still looks like model output. Each new token is rotated as if it sat at position 0, so its query sees every cached key at the wrong relative distance. Caught by `test_cached_logits_equal_full_forward` and by the token-equality test.
2. **Advance per layer.** With a cache sized exactly (as in the lab), it fails loudly: `AssertionError: KV cache is full`, because four layers each consume 16 slots. With a larger cache it is silent and wrong (differences up to 2.8, including the prefill positions): layer 0 writes slots 0–15, then layer 1 believes its tokens sit at positions 16–31 and attends over 16 slots of zeros that layer 1 never wrote. The first layer is correct; the error starts in the second. The lesson: exact-size allocation turned a silent bug into a crash, which is the better failure.
3. **All-True mask.** Single-token decode is unaffected (difference 0 when the prompt is fed one token at a time): the one query is the newest position and may see everything. A 16-token prefill is wrong everywhere (difference 4.6 in the prefill body): earlier positions of the block see later ones. And the damage persists: the *last* prefill position and all later decode steps are also wrong (0.6), although their own masks are fine, because they read cached keys and values that deeper layers computed from contaminated residual streams. A corrupted cache stays corrupted.

Which tests catch what: the logit-equivalence test catches all three, *because it prefills
several tokens and then decodes*. A test that only decodes token by token misses bug 3; a
test that only compares the prefill misses bug 1. A bug none of the course tests would
catch: one that appears only beyond the test's sequence length, for example a sliding
window larger than the test sequence. The parametrised test therefore includes a window
shorter than its 20 tokens.

### Exercise 4: the engine

Reference (CPU, miniature `pocket-d4`): `Match: True`; naive 0.663 s, Engine 0.110 s
(6.0×) for 64 tokens; `k_cache` shape `(4, 1, 128, 2, 32)`; `kv_bytes_per_token()` =
2 048; allocated 262 144 bytes.

1. `(num_layers=4, batch=1, seq_len=128, n_kv_head=2, head_dim=32)`. The course cache is `(L, B, H, T, D)`; nanochat is `(L, B, T, H, D)`. That is the layout Flash Attention 3's `flash_attn_with_kvcache` takes, so on the fast path the kernel reads and writes the cache in place with no transposes; the SDPA fallback transposes on the way in and out.
2. First: batch 1, `seq_len = len(tokens)` (the prompt exactly), for the prefill. Second: batch `num_samples`, `seq_len = len(tokens) + max_tokens` (or the model's `sequence_len` when no limit is given), filled by `prefill(...)` from the first. Two caches so that the prompt is processed once regardless of the number of samples, and so that the prefill kernel sees a batch-1, exactly-sized problem.
3. It assumes all rows hold the same number of tokens. That is true here by construction: all rows share one prompt and advance one token per step (a finished row keeps being fed tokens whose output is ignored). It is false when a server batches *different* requests with different prompt lengths: then each row needs its own position (for RoPE and for where to write), which is why the field is a per-row tensor although the code reads only element 0. Supporting that needs per-row position offsets in the rotary lookup and padding-aware attention.
4. The first token after prefill was once sampled a single time and copied to every row, so all "independent" samples began with the same token. With one sample this is invisible; with greedy decoding it is the correct behaviour. Only a multi-sample, stochastic test with a distribution flat enough to make coincidence impossible ($(1/262)^{15}$) exposes it. Bugs in the *sampling* path need statistical tests, not equality tests.
5. No reference values (not run on GPU by the author). Expected pattern: TTFT roughly independent of decode batch size (prefill is batch 1); TPOT rising slowly with batch size; total tokens/s rising nearly linearly at first, then flattening as the step becomes limited by compute or by reading ever more cache.

### Challenge: prefix sharing

Outline. Prefill the prefix with batch 1; for each suffix, `expand_batch(1)` (a copy) of
the prefix cache with capacity for the suffix and the generation, feed the suffix as one
multi-token block (the mask handles a block appended to a longer cache), then decode.
Handling different suffix lengths: the simplest correct design serves each suffix as its
own batch-1 sequence. Batching suffixes of unequal length needs per-row positions and
padding masks; padding on the *left* is not an option, because it would shift absolute
positions and invalidate the shared prefix.

Validity condition: the token sequence of the full prompt must begin with exactly the
token sequence of the prefix. With BPE that is not automatic: merging can cross the
boundary (the prefix `"Answer:"` followed by `" yes"` may tokenise differently from the
concatenated string). Chat formats avoid the problem with special tokens at message
boundaries, across which no merge can occur (Module 03); that is one reason prefix
caching works cleanly at message granularity.

Expected saving: the prefill cost of the prefix, once per request; in tokens processed,
$(R-1)\,P$ for $R$ requests sharing a prefix of length $P$.

### Written questions

1. The position offset of the rotary embedding (or of learned positions). "Good, then nonsense" means prefill is right and decode is wrong, or the sequence exceeded a limit (window, cache capacity, trained context length).
2. Training memory is fixed by batch × sequence length, chosen by you. In serving, each concurrent sequence owns a cache proportional to its current length, and arrivals are not under your control. Capacity becomes a queueing question.
3. A key's rotation depends only on the key's own position, which never changes, so rotating once at write time is correct; likewise QK-norm is a function of the key alone. Storing unrotated keys and rotating "at read time with the reader's position" would give every key the reader's angle: all relative angles would be zero and position information would vanish. (Storing unrotated keys and rotating each by *its own* position at read time would be correct but wasteful.)

## Quiz

1. Causality: the output at position $j$ does not depend on tokens after $j$. Enforced by the causal mask in attention (every other operation acts on positions independently).
2. A query is used once, by its own position, at the step in which it is computed. Keys and values are read by every *later* position. Caching queries would store something nobody reads again.
3. $12\times2\times6\times128\times2 = 36\,864$ bytes per token; $\times\,8 \times 4096 = 1\,207\,959\,552$ bytes ≈ **1.21 GB**.
4. Naive: $\sum_{i=0}^{199}(100+i) = 20\,000 + 19\,900 = $ **39 900**. Cached: $100 + 199 = $ **299**.
5. Prefill: one parallel pass over the prompt, filling the cache; it determines TTFT. Decode: one token per step. Decode is bandwidth-limited: each step must read all the weights and the whole cache from memory to perform a small amount of arithmetic, so the time to move bytes, not to multiply them, dominates.
6. The position offset: new tokens are encoded as position 0 (or relative to the current call) instead of their absolute position.
7. The mask for a multi-token append: within the block, earlier tokens can see later ones (or the mask is built for a square $T\times T$ case and misapplied to $T\times(\text{pos}+T)$).
8. nanochat's "smear" mixes the previous token's embedding into the current one before the first block. In decode the previous token is not part of the input, so its (normalised) embedding must be carried in the cache. General rule: cache whatever earlier state the new position reads.
9. Prefill the prompt once with batch 1; allocate a cache for batch 16 with room for the generation; copy the prefilled keys/values (and position counters, and any extra state) into each row; decode with batch 16, sampling each row independently, including the very first token.
10. Any two of: sequence length (the speed-up grows with it); model size and hardware; batch size; the baseline (naive full recomputation?); whether prefill is included; decoding settings and stopping rule.
11. No. With teacher forcing every position's target is known, so one parallel forward pass computes all positions exactly once; there is nothing to reuse. (It does matter for training procedures that *generate*, such as the reinforcement learning rollouts of Module 15.)
12. Not necessarily. Cached and uncached computation group floating-point operations differently; in bf16 the logit differences can reach about 1e-2, enough to flip an arg-max between two nearly tied tokens, after which the sequences legitimately diverge. Decide by (a) comparing *logits* at the first differing position: a small difference with a near-tie indicates rounding, a large one a bug; (b) repeating in fp32, where tokens should match; (c) checking that the divergence position is not special (window length, cache capacity, a power of two).
