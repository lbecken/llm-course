# Solutions — Module 09

## Lab exercises

### Exercise 1: size a model by hand (depth 16)

1. Width $= 16\times64 = 1024$ (already a multiple of 128). Heads $= 1024/128 = 8$.
2. Block matrices: $12 \times 16 \times 1024^2 = 201{,}326{,}592$ (the script adds $8$ layers $\times\,12\times 8 = 768$ gate weights: 201,327,360). `lm_head`: $32768\times1024 = 33{,}554{,}432$. Scaling parameters: 234,881,024 (script: 234,881,792).
3. Tokens $= 12 \times 234{,}881{,}792 = 2{,}818{,}581{,}504$; the batch size is $2^{19}$ and the step count is rounded down to 5 376 steps, i.e. 2,818,572,288 tokens.
4. $6N = 1.41\times10^9$ FLOPs per token; total $\approx 1.41\times10^9 \times 2.82\times10^9 = 3.97\times10^{18}$.
5. The script reports 1.59e9 FLOPs/token (SSSL) and 4.47e18 total: the attention term adds about 13 %. With `--window-pattern L` it is 1.78e9 (+26 % over $6N$).
6. Depth 32: width doubles, so block parameters grow $2\times 2^2 = 8\times$ (the head only 2×); tokens grow about 8× (slightly less, because of the head); compute about 64×.

### Exercise 2: does the FLOP rule predict time?

1. Reference (4 CPU cores): 1.04e11, 2.11e11, 2.65e11, 2.32e11, 2.70e11 FLOPs/s. Not constant: the smallest model achieves less than half the rate of the others. Its matrices are small, so per-operation overheads (Python, dispatch, thread start-up, memory allocation) are a large part of the time, and BLAS cannot use all cores effectively on tiny matmuls. Once matrices are large enough, time per FLOP is roughly constant.
2. Attention's term is proportional to the context length, while the $6N$ term is not: quadrupling `block_size` from 256 to 1 024 roughly quadruples the attention FLOPs per token. For the 4-layer, width-256 model the share rises from 11 % to about 33 %.
3. Depth 4: 68.6 % of parameters are lookup tables (`wte` + value embeddings); depth 26: 45.4 %. Tables grow linearly with width (and with the number of value-embedding layers), block matrices quadratically with width and linearly with depth, so blocks win as models grow.

### Exercise 3: the frontier

Reference table in the lesson (D.6).

2. Matmul parameters per width (2 layers, vocabulary 25): about 1.7 k (8), 6.5 k (16), 25 k (32), 100 k (64). Tokens = steps × 2 048. Best models: budget 1e10 → width 8, 0.44 M tokens, ratio ≈ 260; 4e10 → width 16, 0.62 M tokens, ratio ≈ 95; 1.6e11 → width 16, 2.5 M, ≈ 380 (or width 32, 0.79 M, ≈ 32); 6.4e11 → width 32, 3.2 M, ≈ 127. Not constant at this scale, and far from 10–20: these models are tiny relative to their vocabulary-independent task, and Pocket's hard floor distorts the picture. The qualitative law (optimum moves right) survives; the quantitative constant does not transfer from a toy. That is itself the lesson of D.8.
3. Expect the 16-vs-32 ranking to flip between seeds at 1.6e11 and 6.4e11 (differences below 0.003). What is stable: width 8 loses clearly at every budget except the smallest; width 64 loses clearly except at the largest.
4. Width 8: 1.05 → 0.77 → 0.65 → 0.57. It is still improving slowly but is far above the entropy floor (0.314) and above what width 16 reaches with a fraction of the compute. Its floor is the best loss achievable *by a model of that capacity*: an approximation error that data cannot remove.

### Exercise 4: from probe to plan

No reference values exist (no GPU was available to the author). What a good answer
contains: both probe JSON files; achieved FLOPs/s computed as tokens/s × FLOPs-per-token
for each depth (expect the larger model to achieve more, as in Exercise 2);
hours = tokens / tokens-per-second / 3600 using each depth's own throughput; and a choice
that fits the available time with margin for evaluation and at least one repeat.

### Exercise 5: Milestone 2 experiment

Graded by the rubric in `instructor/grading-rubrics.md`. The most common flaw: comparing
at equal steps when the variable changes the cost per step (depth, window pattern, batch
size), so that one arm simply received more compute. State what is held equal: FLOPs,
tokens, or steps, and why that is the right control for the question.

### Challenge

Reasons your exponent can differ from 0.5: the range of budgets is tiny (less than two
decades); Pocket's loss has a hard floor, so large models cannot keep improving;
quadratic fits to four or five noisy points are unstable; one seed per cell; depth is
held fixed while only width varies; the learning rate was not tuned per width.

## Quiz

1. Each matmul weight is used in one multiply and one add per token in the forward pass (2 FLOPs). The backward pass does two matmuls for each forward one (the gradient with respect to the weights and the gradient with respect to the inputs): 4 FLOPs. Total 6.
2. $N = 70$ M matmul parameters (blocks + head). FLOPs per token $= 6 \times 70\text{M} = 4.2\times10^8$. For 1 B tokens: $4.2\times10^{17}$.
3. A lookup performs no arithmetic on the table: it copies one row per token, whatever the table's size, so it adds (almost) no FLOPs. But every row must be stored (weights), and during training its gradient and optimizer state too, so tables count fully towards memory.
4. Width 1 280, 10 heads. From depth 10 to 20, width doubles and depth doubles: $2 \times 2^2 = 8\times$.
5. For a fixed compute budget, the loss-minimising model size and number of training tokens both grow about as the square root of the budget, i.e. in equal proportion (about 20 tokens per parameter in their setting). nanochat's fits also give nearly equal exponents (0.54 and 0.49), but a ratio of about 10–12 tokens per (matmul) parameter, and show that the ratio depends on which parameters are counted and on the architecture.
6. A: capacity-limited (many steps, loss stuck). B: undertrained (40 steps). C: near-optimal. At 16× the budget, A barely improves; C improves somewhat; B gets 640 steps and improves a lot, probably overtaking C or coming close. The optimum moves towards larger models.
7. $8^{0.383} = 2^{1.149} \approx 2.22$; $\log_2$ rounds to 1, so the batch is $2B_\text{ref}$. Powers of two keep the batch divisible by `device_batch_size × sequence_len × world_size` for the usual power-of-two device batch sizes, so gradient accumulation works out to an integer.
8. $1.32\times10^9 / 40{,}000 = 33{,}000$ s ≈ 9.2 hours. Slower in practice because of: periodic evaluation (validation bpb every 250 steps, CORE, sampling), which is not counted in tokens/s; gradient accumulation overhead if the probe used a different batch configuration; thermal throttling or other processes over a long run; data loading if shards are on a slow disk.
9. (i) Which parameters are counted (all, or non-embedding)? (ii) Trained on how many tokens and with how much compute each? A smaller model trained far longer is not "more efficient" in compute. (iii) "Matches" on which evaluation, with what variance, and was the baseline tuned? Also: same tokenizer and data?
10. Re-run with several seeds at depth 4 to see whether 0.002 exceeds the seed spread; if it does, test at one or two larger depths (e.g. 8 and 12) to see whether the effect persists or grows; check that the comparison is at equal compute; only then propose it, stating the scales at which it was verified.
