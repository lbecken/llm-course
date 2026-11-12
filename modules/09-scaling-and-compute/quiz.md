# Module 09 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/09-scaling-and-compute.md`](../../solutions/09-scaling-and-compute.md).

1. **(Concept)** Where does the factor 6 in $C \approx 6ND$ come from?

2. **(Calculation)** A model has 50 M block-matrix parameters, a 20 M output head and 200 M of embedding tables. Estimate its training FLOPs per token (ignore attention's own term). How many FLOPs to train on 1 B tokens?

3. **(Concept)** Why are embedding parameters excluded from the FLOP estimate but not from the memory budget?

4. **(Calculation)** In nanochat, what are the width and head count for `--depth=20`? By what factor do block parameters grow from depth 10 to depth 20?

5. **(Concept)** State the Chinchilla result in one sentence. How does nanochat's own measurement agree and differ?

6. **(Interpretation)** In a sweep at a fixed FLOP budget, model A (small) took 12 000 steps and reached loss 0.60; model B (large) took 40 steps and reached 0.95; model C (medium) took 900 steps and reached 0.37. Label each as capacity-limited, undertrained or near-optimal, and predict what happens to the ranking at 16× the budget.

7. **(Code reading)** `total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))`. If the token budget is 8× that of depth 12, what batch size results (in units of $B_\text{ref}$)? Why round to a power of two?

8. **(Planning)** A probe at depth 12 reports 40 000 tokens/s. How long would the default depth-12 run (1.32 B tokens) take? Name two things that could make the real run slower than this estimate.

9. **(Critical reading)** A paper says "our 300 M-parameter model matches a 1 B-parameter baseline". List three questions you must ask before interpreting the claim.

10. **(Judgement)** You find a change that lowers validation loss by 0.002 at depth 4 in one run. What do you do before proposing it for depth 20?

## Mastery criteria

- [ ] Size a nanochat model of any depth by hand: width, heads, scaling parameters, tokens, steps.
- [ ] Explain compute-optimal training and recognise undertrained and capacity-limited models in a table.
- [ ] Produce a time and memory plan for a run from a probe.
- [ ] Explain what does and does not transfer from small to large models, with an example of each.
- [ ] Complete Milestone 2's controlled experiment with a filled-in record.
- [ ] At least 8 of 10 quiz questions correct.
