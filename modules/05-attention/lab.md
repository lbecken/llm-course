# Module 05 — Lab: compute it, verify it, train it, and intervene on it

Scripts: [`labs/foundations/05_attention.py`](../../labs/foundations/05_attention.py) ·
[`labs/foundations/05_attention_in_a_model.py`](../../labs/foundations/05_attention_in_a_model.py) ·
Code under study: [`mlle/foundations/attention.py`](../../mlle/foundations/attention.py), [`mlle/foundations/experiments.py`](../../mlle/foundations/experiments.py) ·
Solutions: [`solutions/05-attention.md`](../../solutions/05-attention.md)

## G. Exercises

### Exercise 1 (by hand): one row of attention

One query $\mathbf{q} = (2, 0)$. Three keys $\mathbf{k}_0 = (1,0)$, $\mathbf{k}_1 = (0,1)$, $\mathbf{k}_2 = (-1,0)$
with values $\mathbf{v}_0 = (1,0)$, $\mathbf{v}_1=(0,1)$, $\mathbf{v}_2=(1,1)$. $D = 2$.

1. Compute the scaled scores, the attention weights and the output (no mask).
2. Repeat **without** the $1/\sqrt{D}$ factor. Which way do the weights move?
3. Suppose this query sits at position 1 of a causal model (it may see keys 0 and 1 only). Compute weights and output.
4. Double the query: $\mathbf{q} = (4, 0)$. Without recomputing everything, say what happens to the weights and why. What does this tell you about how a model makes a head sharper?

Check with `mlle.foundations.attention.attention`.
**Success criterion:** your weights match to 3 decimals.

### Exercise 2 (coding): test a modification properly

`causal_mask(T, window=w)` implements sliding-window attention. Write three pytest tests
for `MiniGPT` built with `MiniGPTConfig.nanochat_style(50, block_size=32, n_layer=2, n_head=2, n_embd=32, window=4)`:

1. **Intended behaviour present:** with `record=True`, every recorded attention weight at distance greater than 4 is exactly 0.
2. **Unintended change absent:** for an input of length 5, the logits equal those of the same weights in a model with `window=None`.
3. **The change has an effect:** for an input of length 20, the logits differ from the `window=None` model.

Hint: build both models, then `windowed.load_state_dict(full.state_dict())`. Zero-initialised
projections make everything trivially equal; fill the 2-D parameters with noise first.

**Success criterion:** three passing tests; you can say which bug each one would catch.

### Exercise 3 (experimental): the capacity of one head

```python
from mlle.data.tasks import LookupSpec
from mlle.foundations.experiments import train_lookup
train_lookup(LookupSpec(), n_embd=32)                       # baseline
train_lookup(LookupSpec(), n_embd=4)
train_lookup(LookupSpec(), n_embd=2)
train_lookup(LookupSpec(num_keys=32, num_values=8, num_pairs=20), n_embd=32)
```

Predict first: (a) will `n_embd=2` reach 100 % accuracy? (b) with 20 pairs and 32
possible keys, will attention on the answer be higher or lower than in the baseline?
Record final accuracy and attention on the answer. Explain each result in terms of how
many distinct key directions the head must keep apart in its embedding space.

**Success criterion:** table complete, with a geometric explanation for the `n_embd=2` result.

### Exercise 4 (experimental): interventions

```bash
python labs/foundations/05_attention_in_a_model.py
```

1. Before reading the intervention results, write down which single edge you expect to matter most, based on the attention table alone.
2. Record the results. Was your expectation right?
3. Modify the script: block the two edges (`last → cats`, `last → chase`) in **layer 0 only**, then in **layer 1 only**. Predict the outcome from the attention table first.
4. Blocking `last → that` (in all layers) also reduces P(plural). Propose a reason. (Hint: what can the token `that` have picked up in layer 0?)

**Success criterion:** results recorded; a written explanation of why the layer matters, using "information is relocated by earlier layers".

### Exercise 5 (measurement): the quadratic

Extend part 5 of `05_attention.py` to $T = 4096$ and fit the exponent $p$ in
$\text{time} \propto T^p$ from your measurements (slope of $\log$ time against $\log T$).
Is it 2? Give two reasons a measured exponent can differ from the theoretical one at
small $T$.

### Challenge (optional): a two-token lookup that needs two layers

Change the lookup task so that keys and values are **separate tokens**
(`k3 v5 k0 v2 … ? k0`) and the answer is the value token that *follows* the queried key.
Argue why a single attention layer without position information cannot solve this, then
build the smallest model that does (hint: MiniGPT with RoPE, 2 layers, 1 head each) and
describe what each layer's attention does. Verify your description with interventions.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Verify attention numerically; observe a head learning content-based lookup; test a hypothesis about a trained model's attention by intervention |
| **Hypotheses** | (H1) Loop, matrix and fused implementations agree to ~1e-6. (H2) One head learns the lookup task to ≈ 100 % and its weight on the answer rises from 1/5 to > 0.9. (H3) In the Pocket model, removing the attention edge with the largest weight changes the prediction less than removing a specific pair of edges. (H4) Naive attention time grows about 4× per doubling of $T$. |
| **Hardware / software** | CPU; course package |
| **Tier** | A: `05_attention.py` ≈ 30–60 s; `05_attention_in_a_model.py` ≈ 20 s |
| **Metrics** | max absolute difference between implementations; accuracy; mean attention weight on the answer pair; P(plural verb); time per call |
| **Independent variables** | embedding width and number of pairs (Ex. 3); set of blocked edges and layer (Ex. 4); sequence length (Ex. 5) |
| **Controls** | seeds; training steps; the probe sentence; a control intervention on an irrelevant edge |
| **Interpretation** | An intervention result is evidence about *dependence* (does the output need this edge?). Always include a control edge; a model can be disturbed by any change, and the control shows the size of that non-specific effect (here: none). |
| **Common failure modes** | Timing noise at small $T$ (use several repetitions). Interpreting P(plural) on one sentence as a general property: repeat on other sentences before generalising. |

## I. Results and interpretation

Reference values (author, CPU, 2026-11-11, seed 0):

| Lookup configuration | accuracy | attention on answer |
|---|---:|---:|
| 8 keys, 5 pairs, `n_embd=32` | 1.000 | 0.977 |
| 8 keys, 5 pairs, `n_embd=4` | 0.820 | 0.741 |
| 8 keys, 5 pairs, `n_embd=2` | 0.250 | 0.361 |
| 32 keys, 20 pairs, `n_embd=32` | 0.836 | 0.648 |

| Intervention | P(plural verb) |
|---|---:|
| none | 0.999 |
| last → `cats`, all layers | 0.970 |
| last → `chase`, all layers | 0.826 |
| both, all layers | 0.008 |
| both, layer 0 only | 0.999 |
| both, layer 1 only | 0.010 |
| last → `that`, all layers | 0.698 |
| control: last → first `the` | 0.999 |

Your results:

| Experiment | prediction | measured |
|---|---|---|
| | | |

**Questions to answer in writing**

1. In the lookup model, attention weights are a faithful explanation; in the Pocket model they were misleading. What structural difference between the two models accounts for this?
2. Which numbers in your tables would you expect to change with a different training seed: the implementation agreement, the lookup accuracy, the specific attention pattern of "layer 1 head 0", the effect of blocking both edges?
