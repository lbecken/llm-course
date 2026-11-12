# Solutions — Module 04

## Lab exercises

### Exercise 1: RoPE in two dimensions

$\theta=\pi/2$, so at position 1: $\cos = 0$, $\sin = 1$; at position 0: $\cos=1$, $\sin=0$.

1. Query at $m=1$: $y_1 = 1\cdot0 + 2\cdot1 = 2$, $y_2 = -1\cdot1 + 2\cdot0 = -1$, so $(2,-1)$. Key at $n=0$: unchanged, $(3,0)$. Key at $n=1$: $y_1 = 3\cdot0+0\cdot1 = 0$, $y_2 = -3\cdot1 + 0\cdot 0 = -3$, so $(0,-3)$.
2. $(m,n)=(1,0)$: $(2,-1)\cdot(3,0) = 6$. $(m,n)=(1,1)$: $(2,-1)\cdot(0,-3) = 3$, which equals the unrotated $\mathbf{q}\cdot\mathbf{k} = 3$, as it must at distance 0.
3. $(5,4)$ has the same distance as $(1,0)$: the score is 6. $(7,7)$ has distance 0: the score is 3. No arithmetic is needed; that is the point of RoPE.
4. A rotation by $\pi$ maps $\mathbf{k}$ to $-\mathbf{k}$. Since $2\theta=\pi$, this happens at distance 2 (and at 6, 10, …): the score is $-3$.

**Common mistake:** rotating only the key (or only the query). Both are rotated, each by its own position; the score then depends on the difference. Rotating one of them by its absolute position alone would make the score depend on absolute position.

### Exercise 2: the position ablation

3. Two sources. (i) The causal mask: position $t$ attends to $t+1$ tokens, so the result of averaging depends on how many tokens precede, which leaks a noisy sense of absolute position. (ii) Content: characters are informative about local order through repeated application of attention across layers only weakly, but the *set* of characters seen so far already narrows down the possible words ("bag of characters since the last space" is not directly available, but frequency information is). This is enough for 0.39–0.53 nats, far better than the unigram's 2.75, and far from the 0.35 of position-aware models.

Your single-seed agreement values at 800 steps can land anywhere between 0 and 1 for the position-aware variants; that *is* consistent with the reference.

### Exercise 3: embeddings under two initialisations

1. Input space: `dogs, children, foxes, robots`: other plural nouns. Output space: nouns of both numbers at similarity ≈ 1.00. Reading a noun, its number matters for the upcoming verb, so singular and plural must be distinguished on the input side. Predicting a noun, the grammar makes all 12 noun forms equally likely wherever a noun is possible (number is a fresh 50/50 choice for every noun phrase), so the output side never needs to separate them.
2. Intransitive versus transitive. They occur in different contexts with different probabilities: after a subject, 40 % intransitive / 60 % transitive; inside a relative clause ("that …") only transitive verbs occur. The model must give them different logits, so their output vectors differ. `sleeps`/`runs` are always equally likely together, so they coincide.
3. Record the residual stream after block 0 for many occurrences of each noun (e.g. at the noun position in random sentences), average per noun form, and repeat the difference-vector analysis on those averages; or train a linear classifier ("probe") to read number from the block-0 stream and check its accuracy on held-out nouns. If number is linearly decodable there, the information is present although invisible in raw `wte`.

### Exercise 4: embedding norms

```python
from collections import Counter
counts = Counter(tok.encode(text))
rows = sorted(((counts[i], tok.itos[i], model.wte.weight[i].norm().item()) for i in range(tok.vocab_size)), reverse=True)
```

With small initialisation, rows start with norm ≈ $0.02\sqrt{48} \approx 0.14$ and grow
through updates. Rows of frequent tokens (`the`, `.`, the newline) are looked up, and
therefore updated, in nearly every batch, so they typically move furthest from their
initial values. With AdamW the *per-step* movement is roughly the learning rate
regardless of gradient size, so the number of steps in which a row appears is what
matters. Rare tokens stay closer to initialisation: the trained-rarely effect of D.1.

### Challenge

Expected: the learned-position model degrades sharply beyond 48 characters, because the
rows of `wpe` for positions 48–127 were never updated (they are still random), so the
model receives noise as position codes there. The RoPE model should degrade much less:
relative offsets up to 48 behave identically at any absolute position. It may still lose
some accuracy because it never saw attention over more than 48 keys. Report the three
seeds.

## Quiz

1. `(16, 2048, 768)`. At most $16 \times 2048 = 32\,768$ distinct ids occur, which happens to equal the vocabulary size, but real text repeats tokens heavily: typically a few thousand distinct rows per batch get a gradient; all others get exactly zero.
2. Multiplying a one-hot vector by a matrix produces one row of the matrix and adds zeros for all others. The implementation reads that row from memory directly; no multiplications or additions are performed.
3. Input: what the token's presence in the context implies for what follows. Output: in which contexts the token is a likely next token. Tying assumes the two are similar enough that one vector per token can serve both, which saves $V\times C$ parameters and shares statistical strength.
4. (i) With a large random initialisation the learned structure is a small perturbation on large random vectors; raw cosine similarity mostly measures the random part (the nanochat-init model in D.3 shows exactly this at equal loss). (ii) The knowledge can live in how later layers read the embeddings, in the output embeddings, or in value embeddings; similarity of input rows is neither necessary nor sufficient. A behavioural test (does the model predict similar continuations after "cat" and "dog"?) answers the actual question.
5. Because attention, the only operation that mixes positions, is permutation-equivariant: without position information, permuting the input tokens permutes the outputs identically, so the model cannot distinguish orderings of the same tokens (apart from weak effects of the causal mask).
6. Rotations satisfy $R(\alpha)^\top = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha+\beta)$. So $\langle R(m\theta)\mathbf{q}, R(n\theta)\mathbf{k}\rangle = \mathbf{q}^\top R(-m\theta)R(n\theta)\mathbf{k} = \mathbf{q}^\top R((n-m)\theta)\mathbf{k}$, a function of $n-m$ only.
7. Channel 64 (`d = 128 // 2`; `x1` is the first half, `x2` the second; pair $j$ is $(j, j+64)$). The sign does not matter functionally: queries and keys are rotated the same way, and only their relative rotation enters the score. It matters only for loading weights trained under the other convention.
8. Advantage (any one): relative offsets are represented identically at every absolute position; no parameters; no hard maximum length from a table; re-applied in every layer. No difference expected: short, fixed-length contexts on a task where the training data covers all positions densely, such as the Pocket ablation at 2 000 steps.
9. Can conclude: in that run, at that step, one model had resolved the dependency and the other had not. Cannot conclude: that RoPE learns it faster in general. Next: repeat with several seeds and several training lengths; the three-seed reference shows each scheme ranging from near 0 to near 1 at 800 steps, so the single-seed difference is within seed variance.
10. At the embedding layer the vector identifies the current token (decoding it gives a bigram-quality prediction, loss 2.19). Each block adds information gathered from earlier positions and transforms it, moving the vector towards a representation of the *next* token's distribution; after the last block the output head reads the final prediction (1.29). The logit-lens table in D.4 shows the decoded loss falling stage by stage.
