# Module 02 — Lab: counting models against a known optimum

Script: [`labs/foundations/02_ngrams_and_entropy.py`](../../labs/foundations/02_ngrams_and_entropy.py) ·
Code under study: [`mlle/foundations/ngram.py`](../../mlle/foundations/ngram.py), [`mlle/data/pocket.py`](../../mlle/data/pocket.py) ·
Solutions: [`solutions/02-language-modeling.md`](../../solutions/02-language-modeling.md)

## G. Exercises

### Exercise 1 (mathematical): the chain rule on a new sentence

Using the grammar in the docstring of `mlle/data/pocket.py`:

1. Compute the probability of `the old dogs run .` (followed by a newline) as a product of grammar choices, and its surprise in nats.
2. Compute $P(\texttt{o} \mid \texttt{the␣})$, the probability that the fifth character is `o`.
3. How many characters of this sentence have surprise exactly 0? Name the positions that carry information.
4. Check item 1 in Python:
   ```python
   import random, math
   from mlle.data import pocket
   rng = random.Random(0)
   seen = {}
   while "the old dogs run .\n" not in seen:
       s = pocket.sample_sentence(rng); seen[s.text] = s.logprob
   print(-seen["the old dogs run .\n"])
   ```

**Success criterion:** your hand computation and the program agree to 3 decimals.

### Exercise 2 (experimental): how much data does a context length need?

Predict first: with only 200 training sentences, which context length gives the lowest
validation loss: 2, 4, 6 or 10?

```bash
for n in 200 2000 20000; do
  python labs/foundations/02_ngrams_and_entropy.py --sentences $n --orders 2 4 6 10 | sed -n '/context/,/floor/p'
done
```

Fill in the table in section I. Then answer: for each corpus size, which order is best,
and why does the best order grow with the amount of data?

**Success criterion:** table complete; you can state the bias–variance trade-off in terms of "context length" and "examples per context".

### Exercise 3 (coding): temperature for a counting model

Add a `temperature` argument to `NGramLM.generate` so that it samples from
$p_i^{1/\tau} / \sum_j p_j^{1/\tau}$. Generate 200 characters from the 6-character model
at $\tau = 0.2$, $1.0$ and $2.0$.

1. Which temperature produces the most grammatical-looking text? The most varied?
2. At $\tau \to 0$ the model always picks the most likely character. Predict what the output looks like, then try `temperature=0.01`.
3. Is raising $p$ to the power $1/\tau$ the same as dividing logits by $\tau$? Show it in one line.

**Success criterion:** three samples, and a one-sentence explanation of the degenerate output at very low temperature.

### Exercise 4 (conceptual): teacher forcing and exposure bias

1. For a training sequence of 2 048 tokens, how many (context, next-token) examples does one forward pass provide? What is the length of the shortest and the longest context among them?
2. Describe a concrete situation in which a 6-character Pocket model, during *generation*, ends up in a context that never occurred in its training data. What does `probs` do then, and what would a model without back-off or smoothing do?

### Challenge (optional): an exact floor for a fixed context

The entropy floor 0.3139 is for unlimited context. Estimate the best achievable loss for
a model limited to $k$ characters of context, for $k = 1 \dots 8$, by training n-gram
models on a very large corpus (e.g. 200 000 sentences) where sparsity is negligible.
Plot loss against $k$. Where does the curve flatten, and what does the remaining gap to
0.3139 correspond to in the grammar?

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Measure how cross-entropy and a targeted grammatical test depend on context length and training-set size, against the known entropy of the source |
| **Hypotheses** | (H1) Validation cross-entropy decreases with context length when data is ample, and never goes below 0.3139 nats/char. (H2) With little data, long contexts are worse than short ones. (H3) Agreement with an attractor is at or below chance for every fixed context shorter than the dependency. |
| **Hardware / software** | Any CPU; `pip install -e .` |
| **Tier** | A: about 10 s for the default run; up to 2 minutes for contexts ≥ 10 |
| **Inputs** | Pocket corpus from seed 0 (training) and a fixed set of 600 fresh sentences from seed 999 (validation, identical for every corpus size) |
| **Command** | `python labs/foundations/02_ngrams_and_entropy.py [--sentences N] [--orders ...]` |
| **Metrics** | validation cross-entropy (nats/char), perplexity, agreement accuracy on 200 minimal pairs without and with an attractor |
| **Independent variables** | context length; number of training sentences (one at a time) |
| **Controls** | validation set, smoothing constant α = 0.01, seeds |
| **Interpretation** | Compare each loss with the floor (what is achievable) and with ln 25 = 3.219 (knowing nothing). An agreement score *below* 0.5 is informative: it means the model follows a systematic wrong rule. Standard error of an accuracy on 200 items near 0.5 is about 0.035, so differences of a few points are noise. |
| **Common failure modes** | Slow run: contexts ≥ 14 with 20 000 sentences build large tables (minutes). Agreement of exactly 0.42–0.53 for short contexts is not a bug: with too little context the two verb forms are scored only by their own character statistics. |

## I. Results and interpretation

Reference values (author, CPU sandbox, 2026-11-11):

| training sentences | best order among {2, 4, 6, 10} | losses at orders 2 / 4 / 6 / 10 |
|---:|---:|---|
| 200 | 4 | 0.6090 / 0.4476 / 0.4668 / 0.6231 |
| 2 000 | 6 | 0.5974 / 0.3961 / 0.3761 / 0.3915 |
| 20 000 | 10 | 0.5966 / 0.3938 / 0.3711 / 0.3627 |

Your measurements:

| training sentences | order 2 | order 4 | order 6 | order 10 | best |
|---:|---:|---:|---:|---:|---:|
| 200 | | | | | |
| 2 000 | | | | | |
| 20 000 | | | | | |

**Questions to answer in writing**

1. The loss at order 2 barely changes with 100× more data, while the loss at order 10 changes a lot. Why?
2. Agreement with attractor at order 10 is 0.00, not 0.50. What rule is the model following?
3. Which numbers in your table are noisy? How would you find out how noisy?
