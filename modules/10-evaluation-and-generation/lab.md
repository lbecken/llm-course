# Module 10 — Lab: decoding against ground truth, and honest uncertainty

Scripts: [`labs/pretraining/10_decoding.py`](../../labs/pretraining/10_decoding.py) ·
[`labs/pretraining/10_eval_methodology.py`](../../labs/pretraining/10_eval_methodology.py) ·
Code under study: [`mlle/inference/sampling.py`](../../mlle/inference/sampling.py), [`mlle/evaluation.py`](../../mlle/evaluation.py) ·
Solutions: [`solutions/10-evaluation-and-generation.md`](../../solutions/10-evaluation-and-generation.md)

## G. Exercises

### Exercise 1 (by hand): decoding arithmetic

Next-token logits over five tokens: $(3, 2, 2, 0, -1)$.

1. Compute the probabilities at temperature 1.
2. Which tokens survive top-k with $k=2$? (Note the tie.) What does `top_k_filter` do with ties, and is that the only reasonable choice?
3. Which tokens survive top-p with $p = 0.9$? With $p=0.5$?
4. At which temperature does the most likely token have probability exactly 0.5? (Solve numerically or by trial with `torch.softmax`.)

**Success criterion:** items 1–3 match `mlle.inference.sampling` to 3 decimals.

### Exercise 2 (experimental): the trade-off

```bash
python labs/pretraining/10_decoding.py
```

1. Record the table. Plot or sketch *valid* against *unique*: where is the "knee"?
2. Add two settings to the `settings` list: `temperature=1.0, top_k=5` and `temperature=0.3`. Predict their rows first.
3. Add a repetition penalty of 1.3 to greedy decoding (`dict(temperature=0.0, rep_penalty=1.3)`). The character-level penalty hits *every previously used character*. Predict what happens to validity, then explain the result. What does this tell you about applying token-level heuristics blindly?
4. Find an invalid sentence produced at temperature 1.0 and classify its error (agreement, structure, spelling).

**Success criterion:** the table extended by three rows and a written choice of settings for (a) answering a factual question and (b) writing a varied story, each justified by your numbers.

### Exercise 3 (experimental): how sure are you?

```bash
python labs/pretraining/10_eval_methodology.py
```

1. From the five-seed table: how large must a difference in validation loss be before you would call it real with this setup? In agreement accuracy?
2. Your colleague evaluates on 50 items and reports "72 % vs 64 %, so the new model is better". Compute the standard error of each and of the difference (assume independent samples). Is the claim supported?
3. Explain in your own words why the paired interval in part 3 of the output excludes zero while the unpaired intervals overlap.
4. Change model B to 590 steps (almost identical to A). What do you expect for P(A better)? Run it.

**Success criterion:** numerical answers for item 2 and a correct explanation for item 3.

### Exercise 4 (nanochat): evaluate your Milestone 2 model

On the machine with your trained base model:

```bash
cd external/nanochat
python -m scripts.base_eval --model-tag <your tag> --eval bpb,sample --device-batch-size 16
python -m scripts.base_eval --model-tag <your tag> --eval core --max-per-task 100
python -m scripts.base_eval --model-tag <your tag> --eval core --max-per-task 500
```

1. Record train and validation bpb. Is there a gap?
2. Record CORE at 100 and 500 items per task. How far apart are they? Using the per-task accuracies in the CSV, estimate the standard error of one task's accuracy at 100 items.
3. Read the 7 conditioned samples. Which are factually right? Which are fluent and wrong?
4. Generate from one prompt at temperatures 0, 0.6 and 1.2 using `scripts.chat_cli`-style settings on the base model (write a 10-line script with `Engine.generate_batch`). Describe the differences.

On a CPU-only machine use the miniature Pocket model for items 1 and 4 (CORE is not meaningful there).

**Success criterion:** a completed record. *Awaiting validation on GPU: no reference values.*

### Challenge (optional): beam search, and why LLMs do not use it

Implement beam search (keep the $b$ highest-scoring partial sequences) for MiniGPT.
Compare beam width 4 with greedy and with $\tau=1$, top-p 0.9 on the Pocket metrics.
Explain, with your numbers, why approximately maximising sequence probability gives
worse *text* than sampling, although it gives higher-probability sequences.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Quantify the effect of decoding settings with an exact validity oracle; measure seed variance, sampling error and the benefit of paired comparison |
| **Hypotheses** | (H1) Validity falls and diversity rises with temperature. (H2) Greedy decoding yields near-zero diversity. (H3) Nucleus truncation restores validity at high temperature. (H4) Agreement accuracy varies across seeds by more than 0.1 while the loss varies by less than 0.005. (H5) A paired comparison detects a difference that unpaired intervals do not. |
| **Hardware / software** | CPU; course package. Exercise 4: your nanochat environment |
| **Tier** | A: decoding lab ≈ 1 min, methodology lab ≈ 2 min |
| **Metrics** | valid fraction, unique fraction, nats/char of valid sentences under the true grammar; validation loss; agreement accuracy; standard errors; bootstrap intervals |
| **Independent variables** | decoding settings; seed; number of evaluation items; training length of model B |
| **Controls** | one trained model for all decoding settings; same prompts and generator seed; same evaluation sentences for both models |
| **Interpretation** | "Valid" and "unique" are computed only over complete sentences; with 64 × 60 characters there are about 100 sentences per setting, so a validity of 0.96 has a standard error near 0.02. |
| **Common failure modes** | Comparing decoding settings on different models or prompts. Reading a single seed as a result. Forgetting that the penalty and filters act on *tokens* (characters here). |

## I. Results and interpretation

Reference values: lesson D.4 and D.6.

| Decoding | valid | unique | nats/char | your prediction |
|---|---:|---:|---:|---|
| greedy | | | | |
| $\tau=0.3$ | | | | |
| $\tau=1.0$ | | | | |
| $\tau=1.0$, top-k 5 | | | | |
| $\tau=1.0$, top-p 0.9 | | | | |
| greedy + repetition penalty 1.3 | | | | |

| Seeds (5) | mean | std |
|---|---:|---:|
| validation loss | | |
| agreement with attractor | | |

**Questions to answer in writing**

1. Which of your course results so far were single-seed? Which conclusions would you now re-check?
2. Why is "the model's own distribution" ($\tau=1$, no truncation) not automatically the best decoding setting, although it is what the model was trained to match?
