# Solutions — Module 06

## Lab exercises

### Exercise 1: parameters by hand

$V=1000$, $T_\text{max}=128$, $L=4$, $C=64$.

1. GPT-2 style: `wte` $= 64{,}000$; `wpe` $= 128\times64 = 8{,}192$; per block $12C^2+13C = 49{,}152 + 832 = 49{,}984$, four blocks $= 199{,}936$; final LayerNorm $2C = 128$; head tied (0). **Total 272,256.**
2. `num_params()` returns 272256.
3. nanochat style: no position table, no biases, no norm parameters, untied head. `wte` 64,000 + `lm_head` 64,000 + $4\times12\times64^2 = 196{,}608$. **Total 324,608.** It is *larger*: untying the head adds $VC$, which outweighs everything removed.
4. $C=128$: embeddings and position table grow 2× (linear in $C$); block matrices grow 4× (quadratic); biases and norm parameters grow 2×. New GPT-2-style total: 937,728. As width grows, the blocks dominate; as vocabulary grows, the tables do.

The number of heads does not appear anywhere: heads slice the same matrices.

### Exercise 2: predict every shape

1. `(1,16)` → `(1,16,48)` → `(1,4,16,12)` → `(1,4,16,16)` → `(1,16,48)` → `(1,16,192)` → `(1,16,48)` → `(1,16,48)` → `(1,16,29)` → `(29,)`.
2. In the nanochat-style initialisation the two matrices that *write into the residual stream*, `attn.c_proj` and `mlp.c_proj`, are exactly zero. Each block therefore adds zero, and the stream after every block equals the embedding. In the GPT-2 style those matrices are small random values, so each block adds a small random vector.
3. In a trained model every block makes a substantial change (relative changes of order 0.5–2 at this size). There is no universal ordering; typically the last block's contribution is large because it must produce the direction the output head reads. The point of the exercise is that "each block adds a correction" is observable.

### Exercise 3: one switch at a time

Reference numbers are in the lab (three seeds each). Reading them:

| Variant | mean | verdict |
|---|---:|---|
| baseline | 0.370 | seed spread ≈ ±0.003 |
| **no residual** | **2.70** | catastrophic: at or near the unigram loss (2.75). At this learning rate (3e-3) the model collapses and never recovers. Follow-up runs by the author: at lr 1e-3 it reaches 0.77 after 2 000 steps and at 3e-4 0.98, against 0.35 for the residual model after 600. So the network *can* learn without residuals, but only with a much smaller step size and far more slowly. |
| post-norm | 0.398 | clearly worse (+0.028, ten times the seed spread) at equal steps |
| ReLU² | 0.355 | clearly *better* than GELU here (−0.015) |
| RMSNorm | 0.374 | within about one seed spread of the baseline: no reliable difference |
| no MLP | 0.447 | clearly worse, and noisier across seeds: attention can move information but the model cannot compute enough with it |

Note what this does and does not show. Gradient norms at initialisation are similar with and without residual connections in this pre-norm model (Module 07, Exercise 5), so "vanishing gradients" is not the explanation here; that argument applies to deep plain networks *without* normalization. What the residual path provides in this experiment is robustness and speed: each block starts as a small edit to a state that already carries the token's identity, instead of having to re-transmit everything through attention's averaging and the MLP.

So H3 holds: the structural elements (residual path, MLP) matter by an order of magnitude
more than the choice of activation or normalization. Limitations to state with these
results: one model size, one learning rate (ablations that change the scale of
activations may want a different one), 600 steps (differences in *speed* of learning and
differences in *final* quality are not separated).

### Exercise 4: reproduce and dissect

1. Parity 4.05e-06; changes: resid 1.37, x0 1.44, smear 4.09, backout 0.74, value embeddings 0.47.
2. It fails. MiniGPT's own `c_k` stays at its initial values, so its keys differ from nanochat's and the logits differ by order 1. (A parity test that still passed would mean `c_k` did not influence the output, which would itself be a bug.)
3. It fails, and the error grows with sequence length: the rotation angle is $t\cdot\text{base}^{-2i/D}$, so two bases agree at $t=0$ and drift apart as $t$ grows, most for the slowly rotating pairs (large $i$), where the exponent of the base matters most. For very short sequences the discrepancy can be small enough to slip under a loose tolerance, a reminder to test with realistic lengths.
4. Sketch of the change inside the attention forward, mirroring nanochat:
   ```python
   if ve is not None:                                   # ve: (B, T, Hkv*D) looked up from a table by token id
       ve = ve.view(B, T, Hkv, D).transpose(1, 2)       # MiniGPT layout (B, Hkv, T, D)
       gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))   # (B, T, Hkv)
       v = v + gate.transpose(1, 2).unsqueeze(-1) * ve
   ```
   with `ve_gate = nn.Linear(12, n_kv_head, bias=False)` and tables on the layers where
   `layer_idx % 2 == (n_layer - 1) % 2`. Copy `value_embeds[str(i)].weight` and
   `ve_gate.weight` from nanochat. Note where `x` comes from: it is the *normalised*
   block input (`norm(x)`), exactly what nanochat's attention receives.

### Challenge

Expected: little or no benefit on Pocket. Value embeddings add capacity that is indexed
by token identity; with 353 tokens and a language whose difficulty is structural
(agreement), there is little token-specific knowledge to store. In a 32 768-token model
of real text, much of what must be learned *is* token-specific (what follows this word,
what it means), which is where upstream found large gains. A result on one data
distribution does not transfer automatically to another: state the regime with the result.

## Quiz

1. $\mathbf{x}\leftarrow\mathbf{x}+\mathrm{Attn}(\mathrm{norm}(\mathbf{x}))$; $\mathbf{x}\leftarrow\mathbf{x}+\mathrm{MLP}(\mathrm{norm}(\mathbf{x}))$. Post-norm: $\mathbf{x}\leftarrow\mathrm{norm}(\mathbf{x}+\mathrm{Attn}(\mathbf{x}))$, likewise for the MLP. In pre-norm the residual path is a pure identity from input to output, which keeps gradients well-scaled in deep stacks; post-norm puts a normalization on that path and typically needs warm-up and care to train (and was measurably worse in the lab ablation).
2. Forward: each layer only needs to compute a correction to a shared state, so information is preserved by default and the layer can start as the identity. Backward: the Jacobian of each layer is $I + \partial F/\partial\mathbf{x}$, so the gradient reaches every earlier layer through the identity term without being multiplied by a long product of matrices.
3. Mean square $= (1+1+1+1+4)/8 = 1$, so the RMS is 1 and the output equals the input. Output RMS is 1 by construction. What is lost in general is the overall magnitude of the vector (here nothing changes because it was already 1); direction is preserved, and unlike LayerNorm the mean is not removed.
4. From the MLP (and, more weakly, from the softmax that computes attention weights and from normalization). Attention's output is a weighted average of linearly projected inputs: given the weights, it cannot compute non-linear functions of the content it gathers, such as "plural AND end of clause".
5. $12C^2L = 12\times512^2\times8 = 25{,}165{,}824$. MLPs hold $8C^2$ of each block's $12C^2$: two thirds.
6. Only inside attention: the score and weight tensors `(B, H, T, T)`. Their size and the work to fill them grow with $T^2$, while everything else grows with $T$; for long contexts this term dominates time and, unless avoided by the kernel, memory.
7. Any six of: learned positions → RoPE (fewer parameters); LayerNorm with parameters → parameter-free RMSNorm (fewer); norm after embedding (no change); GELU → ReLU² (no change); biases → none (fewer); tied → untied head (more); plain scores → QK-norm (no change); unbounded → soft-capped logits (no change); residual scalars (a few more); value embeddings (many more); smear, backout (a few more); zero-initialised output projections (no change); dropout removed (no change).
8. Before block $i$, the residual stream is rescaled by a learned scalar and a learned fraction of `x0` is added. `x0` is the normalised token embedding (after smear), i.e. the stream's initial state; re-adding it gives deep layers direct access to the original token identity.
9. Extra embedding tables indexed by token id, present on alternating layers. Their rows are added, through a learned per-head gate, to the attention *values* of that layer, so any position attending to position $j$ can directly receive a learned vector for the token at $j$. A lookup and an addition cost almost no FLOPs, but each table has $V\times C$ entries, and with several tables they become the majority of the model's parameters.
10. nanochat initialises its residual-writing projections (and gates) to zero, so an unmodified fresh model's logits depend on almost none of its matrices; two quite different implementations would "agree". Randomising makes every weight matter. Testing that each addition breaks parity shows the neutralisation was real (each addition does influence the output when on) and that the comparison is sensitive enough to detect a missing component.
11. The un-normalised stream is a sum of many block outputs and can have any scale, often large. The output head was trained to read a unit-RMS vector; fed a vector several times larger, its logits are scaled up by the same factor, making the softmax extremely confident (limited only by the soft-cap), and confident errors are heavily penalised by cross-entropy.
12. ids `(B,T)` → embedding lookup `(B,T,C)` → (+ positions, or norm) `(B,T,C)` → for each block: norm `(B,T,C)`; q,k,v projections and head split `(B,H,T,D)`; scores `(B,H,T,T)`; mask + softmax `(B,H,T,T)`; weights @ values `(B,H,T,D)`; merge + `c_proj` `(B,T,C)`; add to stream; norm; `c_fc` `(B,T,4C)`; activation; `c_proj` `(B,T,C)`; add → final norm `(B,T,C)` → `lm_head` `(B,T,V)` → (soft-cap) → softmax `(B,T,V)`; the next-token distribution is row `[:, -1, :]`, shape `(B,V)`.
