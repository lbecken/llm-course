# Solutions — Module 02

## Lab exercises

### Exercise 1: the chain rule on a new sentence

Sentence: `the old dogs run .` + newline.

1. Choices: plural (0.5) × adjective present (0.3) × adjective "old" (1/5) × noun "dogs" (1/6) × no relative clause (0.75) × intransitive (0.4) × verb "run" (1/4) = 0.5 · 0.3 · 0.2 · 0.16667 · 0.75 · 0.4 · 0.25 = **0.000375**. Surprise: $-\ln 0.000375 = 7.889$ nats (19 characters, 0.415 nats/char: a slightly "surprising" sentence, above the average 0.314, mainly because adjectives are the less likely branch).
2. After `the␣` comes an adjective (0.3) or a noun (0.7). `o` starts only the adjective "old": $P = 0.3 \times 1/5 = 0.06$. No noun starts with `o`.
3. Information is carried by: `o` (adjective, and which one: 0.06), `d` after `old␣` (noun "dog(s)": 1/6), `s` after `dog` (plural: 0.5), `r` after `dogs␣` (no relative clause × intransitive × verb starting with r: $0.75 \times 0.4 \times 1/4 = 0.075$… the only verb starting with `r` is "run", so this single character settles everything). Check: $0.06 \times \tfrac16 \times 0.5 \times 0.075 = 0.000375$. ✓. The other 15 characters have surprise 0.
4. The program prints 7.8888.

**Common mistake:** forgetting that not choosing an optional element is also a choice with a probability (0.7 for "no adjective", 0.75 for "no relative clause").

### Exercise 2: data versus context length

Reference table is in the lab. Explanation:

- A short context has few distinct values (86 two-character contexts), each seen thousands of times even in a small corpus: its counts are accurate (low variance) but the context is uninformative (high bias). More data cannot help it: order 2 stays at ≈ 0.60.
- A long context is informative (low bias) but there are thousands of distinct ones, each seen rarely; estimates are noisy and many validation contexts were never seen (high variance). More data helps enormously: order 10 goes from 0.62 to 0.36.
- The best order is where the two errors balance, and it moves to longer contexts as data grows. This is the bias–variance trade-off, and it is the n-gram version of "bigger models need more data" (Module 09).

### Exercise 3: temperature

```python
def generate(self, prompt, num_tokens, seed=0, temperature=1.0):
    g = torch.Generator().manual_seed(seed)
    out = list(prompt)
    for _ in range(num_tokens):
        p = self.probs(out) ** (1.0 / temperature)
        out.append(torch.multinomial(p / p.sum(), 1, generator=g).item())
    return out
```

1. Low temperature gives the most "correct-looking" text and the least variety; high temperature the reverse, with misspelt words appearing at τ = 2.
2. At τ → 0 the model deterministically follows the single most likely path, which quickly enters a cycle: the same most-probable sentence repeated forever (for example "the cat sees the cat sees the cat …" or one short sentence over and over). Greedy decoding of a language model tends towards repetition, a theme of Module 10.
3. Yes: $p_i \propto e^{z_i}$, so $p_i^{1/\tau} \propto e^{z_i/\tau}$.

### Exercise 4: teacher forcing and exposure bias

1. 2 048 examples from one forward pass. Contexts have lengths 1 (only the first token) up to 2 048.
2. Example: the model samples "the dogs helps " (number disagreement, which it produces because it cannot see far enough). Suppose it later samples a rare continuation such as "the robots robots". The 6-character context "obots " followed by "r…" never occurred in training. `probs` backs off to shorter contexts until it finds one that was seen, so generation continues plausibly. Without back-off and smoothing, the count table has no row for the context: the estimate is 0/0. The neural analogue is milder but real: an input unlike anything seen in training yields an output that is not constrained by the training objective.

### Challenge

The curve drops steeply to about $k = 4$–6 and then flattens near 0.36. The remaining
gap to 0.3139 is the subject–verb agreement across intervening material (and, to a small
degree, sentence-length effects): information that sits more than $k$ characters back.
Closing that last 0.05 nats requires looking back up to ~30 characters.

## Quiz

1. $P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_1,x_2)\,P(x_4\mid x_1,x_2,x_3)$. It is exact. The approximation enters only when a model limits or misestimates the conditionals.
2. Perplexity $e^{1.386} = 4.0$. Bits per token $1.386/0.693 = 2.0$. Bits per byte $2.0/4 = 0.5$.
3. Entropy $= \tfrac12\cdot1 + \tfrac14\cdot2 + \tfrac18\cdot3 + \tfrac18\cdot3 = 1.75$ bits per symbol; that is the floor. The uniform model scores $\log_2 4 = 2$ bits; the difference 0.25 bits is its KL divergence from the truth.
4. Feeding the true previous tokens as context during training instead of the model's own predictions. Advantage: all positions of a sequence can be trained in parallel in one pass, and the learning signal is stable. Consequence: at generation time the model conditions on its own samples, which it never practised on; errors can compound (exposure bias).
5. To make $T$ input–target pairs you need $T+1$ tokens: the target of the last input position is the token after it. `[:, :-1]` are the inputs (tokens 0…T−1) and `[:, 1:]` the targets (tokens 1…T).
6. Positions before the last predict tokens that are already known (they are in the context). Only the last position predicts something new. Recomputing the earlier outputs each step is pure waste, which the KV cache removes (Module 11).
7. It has learned the regularity "a verb agrees in number with the noun that most recently precedes it", which is correct whenever subject and verb are adjacent and systematically wrong when another noun intervenes. Scoring 0 % rather than 50 % shows a consistent rule, not ignorance. The model lacks access to the relevant evidence (the subject lies outside its window), not the concept of number.
8. (i) Sparsity: the number of possible contexts grows exponentially with $n$, so most long contexts are never observed and the rest too rarely for reliable counts. (ii) No sharing: each context is an independent table row; similar contexts teach each other nothing, so the model cannot generalise to unseen ones.
9. It is below the entropy of the source (0.3139), which is impossible in expectation. Something is wrong: validation data leaked into training (the model memorised it), the validation set is not from the same distribution (for example easier or shorter sentences), the loss is computed over fewer characters than believed, or the validation set is too small and the difference is noise. Investigate before celebrating. "Too good to be true" results are a signal, a point Module 08 returns to.
10. The application stores the conversation as text and, at every turn, re-sends the entire history (formatted with role markers) as the model's context. The model re-reads it from scratch each time. When the history exceeds the context window, the application must truncate or summarise it; the model itself retains nothing between calls.
