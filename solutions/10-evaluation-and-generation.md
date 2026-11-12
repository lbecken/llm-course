# Solutions — Module 10

## Lab exercises

### Exercise 1: decoding arithmetic

Logits $(3, 2, 2, 0, -1)$.

1. Exponentials: 20.086, 7.389, 7.389, 1.000, 0.368; sum 36.232. Probabilities: **0.554, 0.204, 0.204, 0.028, 0.010**.
2. The second-largest logit is 2 and occurs twice. `top_k_filter` keeps every logit that is not *below* the k-th largest value, so **three** tokens survive. Alternatives: break the tie arbitrarily (keep exactly k). Keeping ties is the more defensible choice (the result does not depend on token order) and is what nanochat's naive `generate` does as well; its `Engine` uses `torch.topk`, which returns exactly k.
3. $p=0.9$: cumulative mass 0.554, 0.758, 0.962. A token is dropped if the mass *before* it already reaches $p$: the fourth token (0.962 before it) and the fifth are dropped, **three** survive. $p=0.5$: the mass before the second token is 0.554 ≥ 0.5, so only **one** survives.
4. $\tau \approx 1.21$ (at 1.2 the top probability is 0.503, at 1.22 it is 0.499).

```python
import torch
from mlle.inference.sampling import top_k_filter, top_p_filter
z = torch.tensor([[3., 2., 2., 0., -1.]])
print(torch.softmax(z, -1), top_k_filter(z, 2), top_p_filter(z, 0.9), top_p_filter(z, 0.5))
```

### Exercise 2: the trade-off

Reference table in the lesson (D.4).

1. The knee is around $\tau = 0.5$–$1.0$: diversity has already risen to ~0.95 while validity is still ≥ 0.96. Beyond 1.0 validity falls quickly with no further gain in diversity.
2. Expect `temperature=1.0, top_k=5` close to the top-p 0.9 row (validity ≈ 0.99–1.00, unique ≈ 0.96): at most positions fewer than five characters have meaningful probability. Expect `temperature=0.3` between greedy and 0.5: validity 1.00, diversity well below 0.94 (few distinct sentences).
3. Validity collapses. With character tokens, every sentence reuses the same 25 characters; after a few words *all* of them have been "used" and are penalised, most strongly exactly the frequent ones (space, `e`, `t`) the text needs next. The penalty no longer discourages repeating a *phrase*; it distorts the whole distribution. A repetition penalty presumes tokens large enough that repeating a token means repeating content. Heuristics designed for one tokenisation or domain can be harmful in another: check them against a measurement.
4. Typical errors at $\tau=1$: agreement errors across a relative clause ("the cats that like the fox that see…" is also structurally invalid, since relative clauses do not nest in Pocket) and incomplete structures ("…helps the cat that .").

Choices: (a) factual question: greedy or $\tau\le0.3$; there is one right answer and diversity has no value. (b) Varied story: $\tau\approx1$ with top-p 0.9 (100 % valid, 97 % unique in the reference run).

### Exercise 3: how sure are you?

1. With a seed standard deviation of 0.0016, a difference in validation loss should exceed roughly 0.005 (about three standard deviations of a single run; less if you average several seeds per arm). For agreement accuracy (std 0.40 at this training length) no single-run difference is meaningful at all: you would need many seeds, or to evaluate later in training when the metric has stabilised.
2. $a=0.72$: SE $=\sqrt{0.72\cdot0.28/50}=0.063$. $a=0.64$: SE $=0.068$. Difference 0.08 with SE $\sqrt{0.063^2+0.068^2}=0.093$: less than one standard error. Not supported.
3. Each sentence's negative log-probability is dominated by the sentence itself (its length and how many rare choices it contains): the per-sentence standard deviation is 3.8 nats. That variation is the same for both models, because they score the same sentences. In the unpaired analysis it widens both intervals to ±0.43. In the paired analysis it cancels: the per-sentence *difference* B − A has a tiny spread, so a mean difference of 0.04 is resolved clearly.
4. P(A better) moves towards 0.5 and the paired interval straddles zero: with nearly identical models, even a paired test finds nothing, which is the correct answer.

### Exercise 4: your Milestone 2 model

No reference values (GPU run not performed by the author). Sanity anchors: train and
validation bpb within a few thousandths of each other when `epoch` stayed at 1; CORE at
100 and at 500 items per task typically differ in the second decimal; with 100 items a
task accuracy near 0.5 has a standard error of 0.05, so individual task scores are very
noisy and only the 22-task mean is usable; at temperature 0 the samples are the model's
single most likely continuation and tend to be generic or repetitive, at 1.2 varied and
error-prone.

### Challenge

Beam search returns sequences of higher total probability than sampling and typically
*all beams are near-copies of the same most-probable sentence*: validity 1.00, diversity
near zero, nats/char well below the language's 0.314. High-probability sequences are not
*typical* sequences: a typical sample contains its share of medium-probability choices.
For tasks with a single correct output (translation in the pre-LLM era) maximising
probability was sensible; for open-ended generation it produces bland, repetitive text.

## Quiz

1. $1.2\times10^6 / 6.0\times10^5 = 2.0$ nats per token. Perplexity $e^{2}=7.39$. Bits per byte $= 1.2\times10^6 / (0.6931 \times 2.5\times10^6) = 0.693$.
2. Special tokens are not part of the text: they have no bytes in the original document, so they must not add to the denominator. Their loss is dropped too, because otherwise the numerator would include predictions (such as "a new document starts here") that have no corresponding bytes, and the ratio would depend on document length rather than on how well text is modelled.
3. For each answer option, a prompt "few-shot examples + question + that option" is built; all prompts are run through the model in one batch; the mean loss over the option's tokens is computed for each; the option with the lowest mean loss is the prediction, compared with the gold index. The model's preference is read from its probabilities, so nothing needs to be sampled.
4. $(0.40-0.25)/(1-0.25)=0.20$. An accuracy of 0.25 gives 0: chance.
5. Whether the model, unprompted, answers in the required format at all (a single letter and nothing else), and whether it would have preferred some other token entirely. A model that puts 2 % on "B", 1 % on the other letters and 96 % on unrelated tokens is scored as answering "B".
6. $\tau=1$: $(0.25, 0.25, 0.5)$. $\tau=0.5$ doubles the logits to $(0,0,\ln4)$: $(1/6, 1/6, 2/3)$.
7. Far fewer: when the model is confident (one or two tokens already hold 90 % of the mass), top-p keeps only those while top-k still admits 40, most of them tail tokens. Far more: when the distribution is flat (hundreds of plausible tokens, e.g. the first word of a story), 90 % of the mass spans many more than 40 tokens.
8. Deterministic or very-low-temperature decoding following the single most probable path into a cycle. Fixes: sample at a moderate temperature (with top-p/top-k); apply a repetition penalty or no-repeat constraint; also valid: stop conditions and a maximum length in the application.
9. Mitigated by lower temperature: random sampling errors, i.e. unlikely tokens drawn from the tail that derail the text. Not mitigated: systematic errors, where the model's most probable continuation is itself false (it never learned the fact, or learned a wrong association); greedy decoding reproduces those with full confidence.
10. X: 0.65, SE $=\sqrt{0.65\cdot0.35/200}=0.034$, interval about [0.58, 0.72]. Y: 0.71, SE 0.032. Difference 0.06 with SE $\sqrt{0.034^2+0.032^2}=0.047$: 1.3 standard errors. Not established.
11. The per-question outcomes of both models: on how many questions did only X succeed, and on how many only Y? If the models agree on most questions and Y additionally solves 14 that X misses while X solves only 2 that Y misses, the paired difference is highly significant although the totals are the same as before. Pairing removes the variance due to question difficulty.
12. Baseline and variant, identical in everything else (data, tokenizer, depth, steps or FLOPs budget stated, evaluation pipeline, decoding at temperature 0 for benchmarks); at least three seeds each; both validation bpb and CORE on the full task set; repeated at a second (and ideally third) depth to check that the effect is not specific to one size; compared at equal compute if the activation changes cost per step; negative or null outcomes reported. This is the standard nanochat's leaderboard applies.
