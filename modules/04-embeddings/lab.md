# Module 04 — Lab: looking inside embeddings and positions

Scripts: [`labs/foundations/04_positions.py`](../../labs/foundations/04_positions.py) ·
[`labs/foundations/04_representations.py`](../../labs/foundations/04_representations.py) ·
Code under study: [`mlle/foundations/positional.py`](../../mlle/foundations/positional.py), [`mlle/foundations/minigpt.py`](../../mlle/foundations/minigpt.py) ·
Solutions: [`solutions/04-embeddings.md`](../../solutions/04-embeddings.md)

## G. Exercises

### Exercise 1 (mathematical): RoPE in two dimensions

Let $\mathbf{q} = (1, 2)$, $\mathbf{k} = (3, 0)$, one frequency $\theta = \pi/2$ per position,
and use nanochat's convention $y_1 = x_1\cos + x_2\sin$, $y_2 = -x_1\sin + x_2\cos$.

1. Compute the rotated query at position $m = 1$ and the rotated key at positions $n = 0$ and $n = 1$.
2. Compute the scores for $(m, n) = (1, 0)$ and $(1, 1)$.
3. Without computing: what is the score for $(m, n) = (5, 4)$? For $(7, 7)$?
4. For which distance does the score equal $-\mathbf{q}\cdot\mathbf{k}$?

**Success criterion:** matches the solution; item 3 answered by argument, not arithmetic.

### Exercise 2 (experimental): the position ablation

```bash
python labs/foundations/04_positions.py --steps 800
```

1. Write your predictions (see the lesson's "predict" box) before running.
2. Record the table. Compare your agreement column with the three-seed reference in the lesson. Is your run consistent with it?
3. Why can a model with `pos=none` still do far better than a unigram model? Name the two sources of order information it retains.

**Success criterion:** your loss for `none` is at least 0.1 nats above the others; you can explain item 3.

### Exercise 3 (experimental): embeddings under two initialisations

```bash
python labs/foundations/04_representations.py
```

1. For the small-init model, list the four nearest neighbours of `cats` in the input space and in the output space. Explain the difference using the grammar.
2. In the output space `sleeps` is close to `runs` (0.99) but farther from `helps` (0.60). Which property of the grammar makes the model need to tell these apart?
3. The nanochat-init model has unstructured-looking input embeddings and the same loss. Propose an experiment (do not run it) that would reveal whether "number" is nevertheless represented as a direction *after the first block*.

**Success criterion:** explanations refer to specific grammar rules, not to "the model understands".

### Exercise 4 (coding): add an embedding-norm probe

Write a function that returns, for a trained `MiniGPT`, the L2 norm of each row of `wte`
together with the token's frequency in the training text. Run it on the small-init
word-level model from Exercise 3 (copy the training lines from the lab script).
Do frequent tokens have larger or smaller norms? Give a reason based on section D.1
(which rows receive gradient, and how often).

**Success criterion:** a table of (token, count, norm) sorted by count, and one sentence of interpretation.

### Challenge (optional): extrapolating beyond the training length

Train two character-level MiniGPTs on Pocket with `block_size=128` but **only ever feed
windows of 48 characters** during training (modify `get_batch` or crop the batch), one
with `pos="learned"` and one with `pos="rope"`. Evaluate the loss on windows of 48 and of
120 characters. Predict which model degrades at 120 and why. Run 3 seeds before you
conclude anything.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Test whether and how position information affects a small language model; inspect what token embeddings and intermediate states encode |
| **Hypotheses** | (H1) Removing position information raises the loss and prevents learning the agreement dependency. (H2) Learned, sinusoidal and rotary encodings reach the same loss on this task. (H3) With a small initialisation, input embeddings of words of the same grammatical category are close and number is approximately a direction; with a large initialisation the raw rows show no such structure at equal loss. (H4) Loss decoded from the residual stream decreases layer by layer. |
| **Hardware / software** | CPU; course package |
| **Tier** | A: `04_positions.py` about 2 minutes (4 trainings of 800 steps); `04_representations.py` about 20 s |
| **Metrics** | validation loss (nats per character or word), agreement accuracy with attractor, cosine similarities, per-layer decoded loss |
| **Independent variables** | `pos` ∈ {none, learned, sinusoidal, rope}; embedding init scale |
| **Controls** | data, seed, architecture, optimizer, steps |
| **Interpretation** | Compare losses with the entropy floor (0.3139 nats/char; 1.2718 nats/word-token). Treat single-seed differences in agreement as unreliable (see the three-seed reference). |
| **Common failure modes** | Running several trainings in parallel on a small CPU makes each much slower (thread oversubscription). Comparing a word-level loss with the character-level floor: units differ. |

## I. Results and interpretation

Reference values are in the lesson (sections D.3, D.4, D.9) and in
[results/reference/position-ablation-seeds.md](../../results/reference/position-ablation-seeds.md).

| `pos` | your val loss (800 steps) | gap to floor | your agreement + attractor |
|---|---:|---:|---:|
| none | | | |
| learned | | | |
| sinusoidal | | | |
| rope | | | |

| stage | reference decoded loss | yours |
|---|---:|---:|
| embedding | 2.187 | |
| block 0 | 1.621 | |
| block 1 | 1.515 | |
| block 2 | 1.292 | |

**Questions to answer in writing**

1. Which of the hypotheses H1–H4 did your runs support? For which would you want more seeds?
2. The decoded loss after the embedding (2.19) is much better than uniform (3.61). What can be predicted from the current token alone?
3. The model's final loss (1.291) is 0.02 above the floor (1.272). Name one possible source of the gap.
