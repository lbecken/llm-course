# Solutions — Module 12

## Lab exercises

### Exercise 1: by hand

1. Scores $(0, 2, 4, 1)$, values $(1, 2, 3, 4)$.
   - Chunk $(0,2)$: $m=2$; $\ell = e^{-2} + 1 = 1.1353$; $o = 0.1353\cdot1 + 1\cdot2 = 2.1353$.
   - Chunk $(4,1)$: $m'=4$, rescale by $e^{2-4} = 0.1353$. $\ell = 0.1353\cdot1.1353 + (1 + e^{-3}) = 0.1537 + 1.0498 = 1.2034$; $o = 0.1353\cdot2.1353 + 1\cdot3 + 0.0498\cdot4 = 0.2890 + 3 + 0.1991 = 3.4881$.
   - Output $3.4881/1.2034 = $ **2.8985**. Direct: weights $(0.0152, 0.1125, 0.8310, 0.0414)$ give $0.0152 + 0.2249 + 2.4929 + 0.1655 = 2.8985$.
2. $q_{\max} = 3$, scale $0.5/3 = 0.1667$. Codes $(3, -1, 0, 0)$; reconstruction $(0.5, -0.1667, 0, 0)$. The two small weights vanish and $-0.20$ is off by 17 %. Only the largest weight is exact, by construction.
3. Batch 1: $100 / 2.1 = $ **47.6** tokens/s. Batch 16: one step moves $2.0 + 16\cdot0.1 = 3.6$ GB, so 27.8 steps/s × 16 = **444** tokens/s. Assumptions: 16× the arithmetic is still negligible against the GPU's compute (false at some batch size), and the full nominal bandwidth is achievable.

```python
import torch
from mlle.inference.flash_sim import attention_tiled
from mlle.inference.quantize import quantize_tensor, dequantize_tensor
q = torch.ones(1, 1); k = torch.tensor([[0.], [2.], [4.], [1.]]); v = torch.tensor([[1.], [2.], [3.], [4.]])
print(attention_tiled(q, k, v, causal=False, block_k=2)[0])          # D = 1, so scores = q*k
codes, s = quantize_tensor(torch.tensor([[0.50, -0.20, 0.04, -0.01]]), bits=3, per_channel=False)
print(codes, s, dequantize_tensor(codes, s))
```

### Exercise 2: the tiled attention

Reference table: lesson D.2.

1. On the reference CPU the crossover is near $T=2000$. Below it the naive version wins because it is one large, well-optimised matrix product while the tiled version runs hundreds of small operations from Python. Above it the naive version's intermediates (64 MiB and up, several of them) no longer fit in the processor's caches and every pass over them goes to main memory, while each 0.25 MiB tile stays in cache. The crossover is the memory hierarchy becoming visible.
2. `k_end = qe if causal else Tk`: for a block of queries ending at position `qe`, key tiles beyond `qe` are never visited. For $T=1024$ with 128-wide tiles there are 8 query blocks; block $i$ visits $i+1$ key tiles: $1+2+\dots+8 = $ **36 of 64**.
3. Measured at $T=2048$: 32 × 32 tiles: 2 080 tiles, 279 ms, largest intermediate 0.016 MiB; 128 × 128: 136 tiles, 44 ms, 0.25 MiB; 512 × 512: 10 tiles, 45 ms, 4 MiB. All agree with naive attention to 4.8e-07. Smaller tiles mean less memory and more loop overhead; beyond the size that fits in cache there is nothing more to gain. A real kernel chooses the tile to fit on-chip memory.
4. They still agree (differences below 1e-6 with scores up to ±620 and ±6 200). The running maximum $m$: every exponent is $s - m \le 0$, so `exp` never overflows. A softmax computed as `exp(s) / exp(s).sum()` overflows to infinity for the same inputs.

### Exercise 3: GQA and windows

Reference (2 layers unless stated, width 64, 1 500 steps):

| Variant | params | KV bytes/token | val loss | agreement |
|---|---:|---:|---:|---:|
| baseline 4/4 heads, full context | 101 504 | 1 024 | 0.3475 | 1.00 |
| GQA 2 kv heads | 93 312 | 512 | 0.3471 | 1.00 |
| GQA 1 kv head | 89 216 | 256 | 0.3472 | 1.00 |
| window 32 | 101 504 | 1 024 | 0.3474 | 1.00 |
| window 16 | 101 504 | 1 024 | 0.3500 | 1.00 |
| window 8, seed 0 | 101 504 | 1 024 | 0.3607 | 0.86 |
| window 8, seed 1 | | | 0.3605 | 0.88 |
| window 8, seed 2 | | | 0.3596 | 0.80 |
| window 8, **4 layers** | | | 0.3514 | 1.00 |
| full context, 4 layers | | | 0.3470 | 1.00 |

1. Window 8 breaks agreement; 16 and 32 do not. The loss barely shows it (+0.013).
2. Robust: 0.86, 0.88, 0.80 across three seeds, with losses within 0.001 of each other.
3. Agreement recovers to 1.00 with four layers: four hops of 8 reach 32 positions. What matters for reach is roughly depth × window. The relay is not free: the windowed 4-layer model's loss is 0.3514 against 0.3470 for full context at the same depth (about three seed standard deviations: suggestive, one seed). Layers spent relaying are not available for other computation.
4. Pattern `SSSL` over 12 layers: 9 short layers (512) and 3 long ones (2 048): $9\cdot512 + 3\cdot2048 = 10\,752$ cached positions per step, against $12\cdot2048 = 24\,576$ for `L`: 44 %.

Recommendation: a short window is safe when the dependencies the task needs are either
shorter than the window or can be relayed in the available depth, **and** some layers
keep full context for the rest. Verify with a test that targets the longest dependency
you care about, not with average loss.

A wrong conclusion to avoid: "GQA with one key/value head is free". It is free *on
Pocket with this model*. The table cannot tell you what a 20-layer model on natural
language loses.

### Exercise 4: how many bits?

Reference table: lesson D.6. The cliff is between 6 and 4 bits per tensor and between 4
and 3 (gently), then 3 and 2 (fatally) per channel.

2. Measured, 3 bits per channel on one group only (base loss 0.3483):

   | Group quantized | parameters | val loss | Δ |
   |---|---:|---:|---:|
   | `wte` + `lm_head` | 3 200 | 0.3657 | +0.0174 |
   | attention matrices | 32 768 | 0.3488 | +0.0005 |
   | MLP matrices | 65 536 | 0.3581 | +0.0098 |

   At 2 bits: 1.508, 0.458 and 0.931 respectively. The embedding and output matrices,
   3 % of the parameters, are the most sensitive by a wide margin; the attention
   matrices the least. Plausible reasons: every token passes through one embedding row
   and every prediction through the output matrix with no redundancy, whereas attention
   scores pass through QK-normalisation and a softmax that tolerate perturbation.
3. For example: embeddings and head 8 bits, MLP 4 bits, attention 3 bits:
   $3200 + 65536/2 + 32768\cdot3/8 = 48\,256$ bytes, 12 % of fp32, slightly below
   uniform 4-bit (50 752) and avoiding its most damaging component. (To claim it is
   *better* than uniform 4-bit you must measure it; the group effects need not add.)
4. Any two of: integer (or mixed) matmul kernels that consume the codes; quantization of *activations*, with calibration data to set their ranges; handling of the scales in the arithmetic (re-quantization between layers); storage format and loading; a decision for layers that are not plain matrices (normalisation, embeddings as lookups).

### Exercise 5: nanochat

1. On a CPU machine: `HAS_FA3 False USE_FA3 False`; `tests/test_attention_fallback.py`: 5 passed, 10 skipped. The skipped tests compare FA3 with SDPA and need a GPU for which an FA3 kernel loads.
2. 2 048 → 512; 1 024 → 256; 256 → 128. Depth 4 with `SSSL` at sequence length 256: windows (128, 128, 128, 256).
3. Bytes per step: $793 + 24.8 = 818$ MB (cache: $9\cdot512 + 3\cdot1152 = 8\,064$ positions × 3 072 bytes). Bounds: (a) $3.35\times10^{12} / 8.18\times10^{8} \approx$ **4 100** tokens/s; (b) $2.73\times10^{11} / 8.18\times10^{8} \approx$ **330** tokens/s. These are ceilings from one formula with a conservative byte count; measured throughput at batch 1 will be a fraction of them, and neither figure has been measured by the author.
4. No reference values (not run on GPU by the author). On a GPU missing from nanochat's tables the MBU and MFU columns print 0 % (the peak is taken as infinite) and carry no information; TTFT, TPOT, tokens/s and memory remain valid.

### Challenge: greedy speculative decoding

Outline. Loop: draft generates $k$ tokens greedily (with its own cache); the target runs
one forward pass over prefix + proposal (or the $k$ new tokens against its cache) and
yields its arg-max at each proposed position; accept proposal tokens while they equal
the target's arg-max; at the first mismatch emit the target's token instead; roll both
caches back to the accepted length (with the course `KVCache`: set `pos`; stale entries
beyond `pos` are overwritten later). Equality with the target's greedy output holds by
construction, and is the test.

Savings: target passes per emitted token ≈ $1/(\text{accepted} + 1)$. On Pocket a
1-layer draft agrees with a 4-layer target on most characters (inside words nearly
always), so acceptance is high. The saving in *time* additionally requires the draft to
be much cheaper than the target and the verification pass to be parallel, which on a
small CPU model it barely is.

For sampling, "equal to the target's sample" is undefined: the target has a distribution,
not a token. The rule of Leviathan et al. accepts a drafted token $x$ with probability
$\min(1, p_\text{target}(x)/p_\text{draft}(x))$ and otherwise resamples from the
normalised positive part of $p_\text{target} - p_\text{draft}$, which makes the emitted
token exactly distributed as $p_\text{target}$.

### Written questions

1. Exact: FlashAttention, decode batching, the KV cache, continuous batching, paged caches, prefix caching, speculative decoding (in distribution). Approximate: sliding windows (test: a dependency longer than the reach), GQA (test: quality at the target scale; tasks needing many distinct lookups per position), quantization (test: full evaluation at the deployed bit-width, especially rare tokens and long generations where small errors compound), distillation (test: everything).
2. Without FA3 a windowed layer needs an explicit $T\times T$ boolean mask and leaves SDPA's fast causal path; nanochat itself warns that utilisation "will be terrible". Fewer FLOPs on paper, more time in practice: the binding resource was never the arithmetic.
3. For example: (a) no loss on *which* metric and task, with what uncertainty; (b) which method (round-to-nearest, calibrated, trained) and are activations quantized too; (c) compared with which baseline precision, and does the claim hold for the model size I care about. Also reasonable: is the speed-up measured or inferred from the size.

## Quiz

1. Arithmetic (FLOPs): training, prefill, large decode batches. Memory held (bytes): many concurrent sequences or long contexts, large attention intermediates. Memory moved (bandwidth): decode at small batch.
2. Exact with respect to the mathematical result of softmax attention: outputs equal the naive implementation's up to floating-point rounding. It never stores the $T\times T$ score/weight matrix, only one tile at a time plus per-query running statistics.
3. $m' = 4$; rescale $e^{-2} = 0.1353$. $\ell = 0.1353\cdot1.5 + (1 + 0.1353) = 1.3383$; $o = 0.1353\cdot6 + 10 + 0 = 10.812$; output so far $10.812/1.3383 = 8.08$.
4. The benefit is not fewer operations but keeping each tile in fast memory within a single kernel, avoiding traffic to main memory. A Python loop adds interpreter overhead per tile and gives no control over where data lives; only at large $T$ does the cache effect show even there.
5. $6 \times 256 = 1\,536$ positions. Each hop must be learned as an explicit relay, consuming capacity in every intermediate layer and position; if training provides weak signal for that, the model does not learn it, and information relayed through several layers is also more exposed to interference.
6. It reduces the number of key/value heads: smaller key and value projections, a proportionally smaller KV cache and less cache traffic per decode step. The number of query heads and the size of the score computation are unchanged. $H_{kv}=1$: multi-query attention.
7. Batch 1: $200/4.5 = 44$ tokens/s. Batch 8: $200/(4 + 8\cdot0.5) = 25$ steps/s → 200 tokens/s.
8. MBU = (bytes a decode step must move ÷ step time) ÷ peak memory bandwidth. MFU = (FLOPs a step must perform ÷ step time) ÷ peak FLOPs. At batch 1 the step is bandwidth-bound, so MBU says how close the implementation is to the physical limit; MFU is tiny by nature and says little.
9. $q_{\max}=7$, scale $1/7 = 0.1429$; codes $(7, 2, -2)$; reconstruction $(1.0, 0.2857, -0.2857)$.
10. The scale is set by the largest magnitude in the group. With one scale per row, rows whose weights are all small get a proportionally finer grid; a single outlier only coarsens its own row.
11. (a) The model was trained with full attention; cutting its context at inference is a distribution shift, and a model *trained* with the window would behave differently (possibly better, by learning relays). (b) Short documents never exercise dependencies longer than 128, so the test cannot detect the loss; an average loss would under-weight it even on long documents. Also: one evaluation, no uncertainty.
12. Only when a compiled FA3 kernel loads for the GPU (Hopper build, or a community build if one exists for the device) *and* the compute dtype is bf16. Otherwise attention goes through PyTorch SDPA, where a sliding window requires an explicit boolean mask instead of the fast `is_causal` path; training still works but slowly, which is why `--window-pattern L` is recommended there.
