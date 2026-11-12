# Module 04 — Embeddings, representation spaces and position

**Part I · Rebuilding the Foundations** | Tier A (CPU, 1–3 minutes) | Study time: 4–5 hours

> Token ids are just labels: id 317 is not "larger" than id 12. The first thing a model
> does is replace each id with a learned vector, and everything afterwards is geometry on
> those vectors. This module covers what those vectors are, what training does to them,
> how the model's working representation evolves from layer to layer, and how the order
> of tokens (which the core mechanism of a Transformer ignores) is put back in.

## A. Learning objectives

After this module you can:

1. Explain an embedding as a row lookup, show it equals a one-hot matrix product, and describe its gradient.
2. Describe what input and output embeddings each come to encode, and interpret an embedding-similarity measurement with appropriate caution.
3. Explain what the residual stream is and how predictions sharpen across layers.
4. Explain why attention needs position information, and compare learned, sinusoidal and rotary encodings.
5. Derive the relative-position property of RoPE and connect each term to nanochat's `apply_rotary_emb`.

## B. Prerequisites

- [Module 01](../01-neural-network-refresher/lesson.md) §D.7 (one-hot inputs select rows of the first weight matrix).
- [Module 03](../03-tokenization/lesson.md) (token ids, vocabulary size).
- [Mathematics primer](../../docs/mathematics-primer.md) §1 (dot products, cosine similarity) and §8 (rotations).

## C. Motivation and intuition

In Module 01 the window-MLP received one-hot vectors, and we noticed that multiplying a
one-hot vector by a weight matrix simply selects one row. Modern models make that
explicit: keep a table with one row per vocabulary entry and **look rows up by token id**.

```text
token ids:   [ 17,  4,  29 ]                 shape (T,)      T = 3 tokens

wte (V x C):  row 0  [ 0.31 -1.20  0.05  0.77 ]
              ...
              row 4  [-0.42  0.18  1.03 -0.66 ]   <--- id 4
              ...
              row 17 [ 0.90  0.02 -0.35  0.48 ]   <--- id 17
              ...
              row 29 [-1.11  0.64  0.27  0.09 ]   <--- id 29

output:      [[ 0.90  0.02 -0.35  0.48 ],
              [-0.42  0.18  1.03 -0.66 ],
              [-1.11  0.64  0.27  0.09 ]]     shape (T, C)    C = 4 here; 768 in nanochat d12
```

The rows start random. Training moves them. Because the only pressure on them is the
prediction loss, they end up encoding whatever about a token helps predict what follows.
Tokens that behave alike in text end up with related vectors, not because anyone defined
"alike", but because the same gradients push them the same way.

The width $C$ (`n_embd`) is the size of the model's working memory per position. Every
layer reads and writes vectors of this size. More dimensions mean more room to keep
different kinds of information apart.

## D. Rigorous technical explanation

### D.1 The embedding layer

Parameters: $E \in \mathbb{R}^{V\times C}$. Forward, for ids $x_1,\dots,x_T$:
$\;\mathbf{h}_t = E[x_t]$, giving a tensor of shape `(B, T, C)`.

Equivalence with a matrix product: if $\mathbf{o}_t$ is the one-hot row vector for $x_t$,
then $\mathbf{o}_t E = E[x_t]$. The lookup is a matmul that skips the multiplications by
zero (verified in `labs/foundations/04_positions.py`, part 1). This is why nanochat's
FLOP count ignores the embedding: it costs a memory read, not arithmetic.

Gradient: $\partial L/\partial E$ is zero everywhere except in the rows that were looked
up, where it equals the gradient arriving at that position (summed if a token occurs
several times). Consequences:

- A token that never appears in a batch receives no update. Rare tokens are trained rarely.
- Embedding gradients are sparse and have different statistics from those of dense matrices, one reason nanochat uses a separate optimizer group and learning rate for them (Module 07).

### D.2 The other end: the output head as a second embedding

The final layer maps the last hidden state $\mathbf{h}\in\mathbb{R}^C$ to logits:
$z_v = \mathbf{h}\cdot\mathbf{u}_v$, where $\mathbf{u}_v$ is row $v$ of the output
matrix $U\in\mathbb{R}^{V\times C}$ (`lm_head`). So there are *two* vectors per token:

| | Matrix | Role | Asked question |
|---|---|---|---|
| input embedding | `wte` | what the model reads when token $v$ is in the context | "given that $v$ occurred here, what does that imply about what comes next?" |
| output embedding | `lm_head` | what the hidden state is compared with to score $v$ as the next token | "in which contexts is $v$ a likely continuation?" |

GPT-2 used one matrix for both (**weight tying**): fewer parameters, and a reasonable
prior that the two roles are related. nanochat keeps them separate (and initialises
them very differently: `wte` with standard deviation 0.8, `lm_head` with 0.001).

### D.3 What the vectors learn: a measurement, and a caution

`python labs/foundations/04_representations.py` trains two word-level models on Pocket
that are identical except for the initial scale of `wte`, and inspects both matrices.
Both reach the same validation loss (1.291 nats/word; entropy floor 1.272). Measured
on CPU, seed 0:

**Model with small initial embeddings (std 0.02):**

```text
input (wte)     nearest neighbours by cosine similarity
    cat     -> bird (0.93), robot (0.93), dog (0.92), fox (0.91)          singular nouns
    cats    -> dogs (0.95), children (0.94), foxes (0.92), robots (0.91)   plural nouns
    sees    -> helps (0.88), likes (0.88), chases (0.88), help (0.51)      singular transitive verbs
    small   -> quiet (0.89), red (0.85), old (0.78), quick (0.77)          adjectives
    number direction: cos(cats - cat, dogs - dog) etc.: mean +0.81, min +0.73 over 15 noun pairs

output (lm_head)
    cat     -> foxes (1.00), bird (1.00), robot (1.00), child (1.00)       ALL nouns, both numbers
    sleeps  -> sings (0.99), waits (0.99), runs (0.99), helps (0.60)       singular intransitive verbs
```

Three findings:

1. **Grammatical categories emerge as clusters.** Nobody told the model what a noun is.
2. **A feature can be a direction.** The vector from `cat` to `cats` is nearly parallel to
   the vector from `dog` to `dogs` (cosine 0.81 on average): "plural" is approximately
   *one direction* in the space, added to any noun. This linear structure is what people
   mean by statements like king − man + woman ≈ queen.
3. **Input and output spaces differ, for a reason.** In the *input* space `cat` and
   `cats` are apart: number matters for what follows (the verb must agree). In the
   *output* space all twelve noun forms are almost the same vector: in Pocket every noun,
   singular or plural, is an equally likely continuation wherever a noun can occur, so
   the model never needs to score them differently. Each space encodes exactly the
   distinctions its role requires.

**Model with nanochat's initial embeddings (std 0.8), same loss:**

```text
input (wte)
    cat     -> child (0.30), old (0.18), birds (0.18), help (0.12)         no visible structure
    number direction: mean +0.03, min -0.16                                indistinguishable from random
output (lm_head)
    cat     -> birds (1.00), dogs (1.00), bird (1.00), robots (1.00)       same clean structure as above
```

The input embeddings of this model look random, yet it predicts exactly as well. The
explanation: 37 random vectors in 48 dimensions are already almost orthogonal, i.e.
perfectly *distinguishable*, and the layers that read them can learn any needed grouping
through their own weights. Training changed these rows only slightly relative to their
large random starting values, so the structure exists in how the network *uses* the
vectors, not in their raw cosine similarities. With a small initialisation the learned
component dominates and becomes visible.

> **Caution: similarity of raw embedding rows is not a reliable window into a model.**
> The same function can be implemented with clean-looking or random-looking embeddings.
> An absence of visible structure does not show an absence of knowledge, and (as Module
> 05 argues for attention weights) a visible pattern does not by itself show how the
> model computes. Use such measurements as hypotheses to test, not as explanations.

### D.4 The residual stream: representations evolve by addition

After the lookup, the tensor `(B, T, C)` flows through the layers. Each layer computes
something from it and **adds** the result back (Module 06). This running sum is called
the **residual stream**. The vector at position $t$ starts as "which token is here" and
is gradually rewritten into "what comes after here", which the output head then reads.

One way to watch this happen is the *logit lens*: take the stream after each layer,
apply the final normalization and the output head as if the model ended there, and
measure the loss. Measured on the nanochat-init model above:

| Residual stream decoded after | Loss (nats/word) | |
|---|---:|---|
| embedding only | 2.187 | knows the current token: a bigram-quality guess |
| block 0 | 1.621 | |
| block 1 | 1.515 | |
| block 2 (final) | 1.292 | |
| *(entropy floor)* | *1.272* | |
| *(uniform guess, ln 37)* | *3.611* | |

The prediction is refined step by step; no single layer "contains" it. (The logit lens is
a heuristic: intermediate layers were never trained to be decoded this way, so it can
understate what they hold.)

### D.5 Why position must be added: attention is blind to order

Attention (Module 05) computes, for each position, a weighted average over other
positions with weights that depend only on the *contents* of the vectors. Shuffle the
input tokens and each output is the same as before, just moved to its token's new place.
Formally the operation is **permutation-equivariant**; the test
`test_attention_without_positions_is_permutation_equivariant` checks it to 1e-5.

A model built only from such operations treats "the dog chases the cats" and "the cats
chases the dog" as the same bag of words. Order information must be injected.

*(A causal mask breaks the symmetry a little: position $t$ averages over $t+1$ tokens, so
some positional information leaks in. The ablation in D.9 shows this is far from enough.)*

### D.6 Option 1: learned absolute positions (GPT-2)

A second table $P\in\mathbb{R}^{T_\text{max}\times C}$, one row per position, added to the
token embedding: $\mathbf{h}_t = E[x_t] + P[t]$.

Simple and effective, with three weaknesses: it cannot represent positions beyond
$T_\text{max}$; each position's vector is learned separately, so nothing learned about
"3 tokens back" at position 10 transfers to position 500; and the model must *derive*
relative distance from two absolute codes.

### D.7 Option 2: sinusoidal positions (the original Transformer)

A fixed code with no parameters:

$$PE[t, 2i] = \sin\!\big(t\,\omega_i\big),\qquad PE[t, 2i+1] = \cos\!\big(t\,\omega_i\big),\qquad \omega_i = 10000^{-2i/C}$$

Each pair of channels is a clock hand rotating at its own speed $\omega_i$: fast for small
$i$ (wavelength $2\pi \approx 6$ positions), slow for large $i$ (wavelength up to
$2\pi\cdot 10000$). Like the digits of a number, fast channels distinguish neighbours and
slow channels distinguish distant regions. Because $\sin$ and $\cos$ of $(t+k)\omega$ are
a fixed rotation of those of $t\omega$, a shift by $k$ positions is a linear function of
the code, which makes relative offsets easy to compute in principle.

### D.8 Option 3: rotary position embedding (RoPE), used by nanochat

RoPE takes the clock-hand idea and applies it where it matters: **directly to the queries
and keys inside attention**, instead of adding anything to the residual stream.

**In two dimensions.** Let a query $\mathbf{q}$ at position $m$ and a key $\mathbf{k}$ at
position $n$ be 2-D vectors. Rotate each by an angle proportional to its position, then
take the dot product:

$$\langle R(m\theta)\,\mathbf{q},\; R(n\theta)\,\mathbf{k}\rangle = \mathbf{q}^\top R(m\theta)^\top R(n\theta)\,\mathbf{k} = \mathbf{q}^\top R\big((n-m)\theta\big)\,\mathbf{k}$$

because rotations compose by adding angles and $R(\alpha)^\top = R(-\alpha)$. **The
attention score depends on the positions only through their difference $n - m$.** The
absolute positions have cancelled.

**A number to hold on to.** Take $\mathbf{q}=\mathbf{k}=(1,0)$ and $\theta = 0.5$ rad. The
score is $\cos(0.5\,(m-n))$:

| distance $m-n$ | 0 | 1 | 2 | 3 | 6 |
|---|---:|---:|---:|---:|---:|
| score | 1.000 | 0.878 | 0.540 | 0.071 | −0.990 |

Identical contents, different distances, different scores. With a single frequency the
dependence on distance is a plain cosine; with many frequencies the model can combine
them into sharply tuned preferences ("exactly one token back", "anywhere within the last
20").

**In $D$ dimensions.** Split the head dimension $D$ into $D/2$ pairs. Pair $i$ rotates
with its own frequency $\theta_i = \text{base}^{-2i/D}$, by the angle $t\,\theta_i$ at
position $t$. nanochat's implementation
([`gpt.py` L57–L65](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L57-L65)):

```python
def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]     # split up last dim into two halves
    y1 = x1 * cos + x2 * sin            # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

Details that differ between implementations and matter when porting weights:

| Choice | nanochat at the pin | Common alternative |
|---|---|---|
| Which channels form a pair | channel $j$ with channel $j + D/2$ (two halves) | adjacent channels $(2j, 2j+1)$ |
| Direction | rotation by $-t\theta_i$ | $+t\theta_i$ (equivalent: only relative rotation matters) |
| Base | 100 000 ([`gpt.py` L270](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L270-L285)) | 10 000 |
| Applied to | queries and keys, in every layer, before QK-norm | same |

The course's `mlle/foundations/positional.py` follows nanochat's convention exactly; the
parity test of Module 06 would fail otherwise.

Why RoPE won:

- **Relative by construction.** "Attend to the token 3 back" is the same computation at position 10 and at position 5 000.
- **No parameters, no table size limit.** Tables of cos/sin are computed, not learned.
- **Re-applied in every layer**, so position information cannot be washed out of the residual stream as depth grows.
- **Rotations preserve length**, so they do not disturb the scale of queries and keys.

Measured (`labs/foundations/04_positions.py`, part 3): for random 8-dimensional $\mathbf{q},\mathbf{k}$,
the score at $(m,n) = (5,2)$, $(25,22)$ and $(60,57)$ is identical (+1.53657), and differs
at distance 2 (+1.41848) and 0 (+1.65659).

### D.9 Does it matter? A controlled ablation, three seeds

Same model, data and training budget; only `pos` changes
([full results](../../results/reference/position-ablation-seeds.md)). "Agreement" is the
attractor test of Module 02 (chance = 0.5).

| `pos` | val loss, 2 000 steps (3 seeds) | agreement, 2 000 steps | agreement, 800 steps |
|---|---|---|---|
| none | 0.395, 0.386, 0.386 | 0.32, 0.39, 0.35 | 0.37, 0.42, 0.43 |
| learned | 0.3475, 0.3475, 0.3474 | 1.00, 1.00, 1.00 | 0.04, 0.59, 0.92 |
| sinusoidal | 0.3467, 0.3471, 0.3474 | 1.00, 1.00, 1.00 | 0.22, 0.70, 1.00 |
| rope | 0.3469, 0.3475, 0.3474 | 1.00, 1.00, 1.00 | 0.99, 0.18, 1.00 |

What the data support:

- **Without position information the model is clearly worse** (all seeds), and its
  agreement stays below chance: like the n-gram model, it sides with the nearest noun.
- **On this small task the three position schemes end up equivalent.** RoPE's advantages
  (relative offsets at any absolute position, long contexts) are not exercised by
  64-character windows of a toy language.
- **The 800-step column is a warning.** The first run of this experiment used one seed
  and showed rope 0.99 versus learned 0.04, a tempting story about RoPE learning
  long-range structure faster. Two more seeds dissolved it: every scheme swings between
  failure and success at that point in training. *One seed is an anecdote.* Module 10
  turns this into method.

## E. Minimal implementation

[`mlle/foundations/positional.py`](../../mlle/foundations/positional.py) implements all
three schemes in about 70 lines. The RoPE pair:

```python
def rope_cos_sin(seq_len, head_dim, base=10000.0, offset=0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # (D/2,)
    t = torch.arange(offset, offset + seq_len, dtype=torch.float32)                             # (T,)
    freqs = torch.outer(t, inv_freq)                                                            # (T, D/2)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):               # x: (..., T, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
```

`offset` exists for cached decoding: when only the newest token is processed, it must be
rotated by the angle of its true position in the sequence, not of position 0 (Module 11).

In MiniGPT ([`mlle/foundations/minigpt.py`](../../mlle/foundations/minigpt.py)) the three
options enter at two different places, which is the conceptual difference between them:

```python
# learned / sinusoidal: ADDED to the residual stream, once, at the input
x = self.wte(idx)
if self.wpe is not None:
    x = x + self.wpe(torch.arange(pos0, pos0 + T, device=idx.device))

# rope: applied to q and k INSIDE every attention layer
if self.config.pos == "rope":
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

> **Predict before you run** `python labs/foundations/04_positions.py --steps 800`:
> which variant will have the highest loss? Will the three variants *with* position
> information differ by more or less than 0.01 nats?

## F. nanochat code walkthrough

**Token embedding and its initialisation.** `wte` is an `nn.Embedding(padded_vocab_size,
n_embd)` ([`gpt.py` L173–L177](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L173-L177)).
The vocabulary is padded up to a multiple of 64 for efficiency; the extra rows are never
used and the corresponding logits are sliced away. Initialisation
([L219–L221](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L219-L221)):

```python
torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
```

Large, well-separated input vectors; a near-zero output head, so the first predictions
are almost uniform and the initial loss is close to $\ln V$ (Module 01's sanity check).
This is the configuration in which section D.3 found raw input embeddings uninformative
to inspect.

**Embedding, then normalise** ([L470–L473](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L470-L473)):

```python
x = self.transformer.wte(idx)   # embed current token
x = x.to(COMPUTE_DTYPE)
x = norm(x)
```

The RMS normalisation right after the lookup removes the overall scale, so the std-0.8
choice affects the optimizer's relative step size on the embeddings rather than the
magnitude seen by the first block.

**Embeddings are stored in the compute dtype** (bf16 on modern GPUs) to save memory
([L262–L268](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L262-L268)),
while the matrices inside the blocks are stored in fp32 (Module 13).

**No positional table.** The config has no position-embedding entry. Position enters
only through RoPE: tables are precomputed for ten times the training sequence length
([L197–L201](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L197-L201)),
sliced to the current positions, with an offset when a KV cache is in use
([L466–L468](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L466-L468)):

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

and applied to queries and keys in every attention layer
([L99–L101](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L99-L101)).

**A third kind of embedding: value embeddings.** nanochat has additional token-indexed
tables, `value_embeds`, one for every second layer
([L189–L192](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L189-L192)).
Their rows are added (through a learned gate) to the attention *values* of that layer,
giving deep layers direct access to "which token is at this position" without that
information having to survive in the residual stream. They are cheap in compute (a
lookup) but large in parameters: at depth 12 they hold 151 M of the model's 286 M
parameters. Module 06 shows where they enter the attention computation.

| | MiniGPT (course) | nanochat |
|---|---|---|
| Token embedding | `wte`, init per chosen style | `wte`, std 0.8, vocabulary padded to a multiple of 64, stored in bf16 on GPU |
| Output head | tied or untied (switch) | untied, std 0.001 |
| Position | learned / sinusoidal / rope / none (switch) | RoPE only, base 100 000, tables for 10× sequence length |
| Extra embeddings | none | value embeddings on alternating layers |

## M. Key takeaways

1. An embedding is a learned lookup table; the lookup is a one-hot matmul without the wasted arithmetic. Only looked-up rows receive gradient.
2. Every token has an input vector (what its presence implies) and an output vector (when it is likely); they encode different distinctions. Tying them is a design choice.
3. Features can appear as clusters and as directions, but whether you can *see* them in raw embedding rows depends on initialisation. Looking is not explaining.
4. The model's state is the residual stream: one vector per position, rewritten additively from "current token" into "prediction of the next".
5. Attention ignores order, so position must be injected: by adding a position vector to the stream (learned, sinusoidal) or by rotating queries and keys (RoPE), which makes scores depend on relative position.
6. A difference observed with one seed can vanish with three.

**Where this sits in the full system:** ids → vectors is the entry to the network, and
vectors → logits its exit. Between them, the only operation that moves information
between positions is attention, the subject of Module 05.

Next: [lab](lab.md) · [quiz](quiz.md) · [references](references.md) · then [Module 05 — Attention](../05-attention/lesson.md)
