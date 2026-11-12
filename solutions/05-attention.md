# Solutions — Module 05

## Lab exercises

### Exercise 1: one row of attention

Raw dot products $\mathbf{q}\cdot\mathbf{k}_j = (2, 0, -2)$.

1. Scaled by $1/\sqrt2$: $(1.414, 0, -1.414)$. Exponentials: $4.113,\ 1,\ 0.243$; sum 5.356. Weights $(0.768,\ 0.187,\ 0.045)$. Output $= 0.768\,(1,0) + 0.187\,(0,1) + 0.045\,(1,1) = (0.813,\ 0.232)$.
2. Unscaled scores $(2, 0, -2)$: exponentials $7.389,\ 1,\ 0.135$; sum 8.524; weights $(0.867,\ 0.117,\ 0.016)$; output $(0.883,\ 0.133)$. The distribution is **sharper**: larger score differences concentrate the softmax.
3. Only keys 0 and 1: scores $(1.414, 0)$ → weights $(0.804,\ 0.196)$; output $(0.804,\ 0.196)$. The weights are renormalised over the visible keys.
4. Doubling $\mathbf{q}$ doubles every score, which is the same as halving the temperature: weights $\approx (0.941,\ 0.056,\ 0.003)$. A model makes a head sharper by increasing the norm of its queries/keys (through $W_Q$, $W_K$). Under QK-norm this route is closed: the norms are fixed, and sharpness is limited by the constant scale (1.44√D in nanochat).

```python
import torch
from mlle.foundations.attention import attention
q = torch.tensor([[2., 0.]]); k = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]); v = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
print(attention(q, k, v))            # item 1
print(attention(q, k, v, scale=1.0)) # item 2
```

### Exercise 2: three tests for a sliding window

```python
import torch
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig

def make_pair():
    torch.manual_seed(0)
    kw = dict(block_size=32, n_layer=2, n_head=2, n_embd=32)
    full = MiniGPT(MiniGPTConfig.nanochat_style(50, **kw)).eval()
    for p in full.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.2)       # zero-initialised projections would hide everything
    win = MiniGPT(MiniGPTConfig.nanochat_style(50, window=4, **kw)).eval()
    win.load_state_dict(full.state_dict())
    return full, win

def test_no_weight_beyond_window():
    _, win = make_pair()
    win(torch.randint(0, 50, (1, 20)), record=True)
    for att in win.internals["attention"]:           # (B, H, T, T)
        T = att.shape[-1]
        dist = torch.arange(T)[:, None] - torch.arange(T)[None, :]
        assert torch.all(att[..., dist > 4] == 0)

def test_short_inputs_unchanged():
    full, win = make_pair()
    idx = torch.randint(0, 50, (1, 5))
    assert torch.allclose(full(idx), win(idx), atol=1e-6)

def test_long_inputs_differ():
    full, win = make_pair()
    idx = torch.randint(0, 50, (1, 20))
    assert not torch.allclose(full(idx), win(idx), atol=1e-3)
```

What each catches: test 1, a window that is too wide or not applied in some layer;
test 2, a window that is off by one (too narrow) or a change that disturbed something
unrelated (for example position handling); test 3, a "window" option that is silently
ignored. A modification that passes only test 1 is not verified.

### Exercise 3: the capacity of one head

Reference numbers are in the lab. Explanations:

- **`n_embd=32`: 100 %.** Eight key directions fit comfortably; queries can match their own key and be nearly orthogonal to the other seven.
- **`n_embd=4`: 82 %**, attention 0.74. Eight keys in four dimensions cannot all be mutually orthogonal; the query for one key necessarily has a positive dot product with some others, so attention leaks to wrong pairs and blends their values.
- **`n_embd=2`: 25 %**, attention 0.36. In a plane, eight directions are at most 45° apart; the best achievable score gap between the right key and its neighbours is small, the softmax stays blurry, and the 2-D output is a mixture of values. (a) No.
- **20 pairs from 32 keys, `n_embd=32`: 84 %**, attention 0.65. (b) Lower. There are four times as many keys to keep apart, and the softmax now has 19 distractors instead of 4: with the same score gap, the weight on the correct pair is smaller. More training or a larger width raises it.

General point: a head separates items by direction, so the number of things it can
select among cleanly is tied to its dimension. This is one reason head width is kept
substantial (nanochat: 128) rather than increasing the number of heads indefinitely.

### Exercise 4: interventions

1. Most people pick `last → cats` (weight 0.76 in layer 1 head 0).
2. That edge alone moves P(plural) only from 0.999 to 0.970. `last → chase` alone: 0.826. Both: 0.008.
3. **Layer 0 only: no effect (0.999). Layer 1 only: 0.010.** The attention table predicts this: in layer 0 the last position attends to `that` and to the newline, not to `cats` or `chase`, so there is nothing to block there. The number information is read in layer 1.
4. Blocking `last → that` gives 0.698. In layer 0, the position of `that` can attend to its left neighbour `cats` and write the subject's number into its own residual stream. In layer 1, head 1 puts 0.52 of its weight on `that` and reads that relocated copy. So there are at least three routes for the same bit: directly from `cats`, from `chase` (which agrees with the subject), and from `that` (which copied it). An attention weight on a position tells you the reader took *something* from that position's stream, not which original token the information came from.

### Exercise 5: the quadratic

From the reference timings: 256 → 2048 is 8× in $T$ and 55× in time, an overall exponent
of $\log 55/\log 8 \approx 1.9$; step by step the factors are 2.5×, 5.4×, 4.0×. Reasons
measured exponents deviate: (i) fixed overheads (Python dispatch, allocation, thread
start-up) dominate at small $T$ and flatten the curve; (ii) memory hierarchy: when the
$T\times T$ matrix outgrows a cache level the cost per element jumps, giving a locally
steeper slope; (iii) the BLAS library switches strategies and thread counts with size.

### Challenge: two-token lookup

A value token does not "know" which key precedes it: its own embedding contains only the
value. A single attention layer lets the query match keys (it can find the queried key's
position) but then it would copy *the key's* value vector, not the following token's.
Without position information it cannot even express "the token after".

Two layers solve it by composition. Layer 1: every value token attends to the token
immediately before it (a fixed relative offset of −1, expressible with RoPE) and copies
the key's identity into its own residual stream. Layer 2: the query attends to the
positions whose stream now contains "preceded by key $k$" matching its own key, and
copies the value identity from there. This previous-token-then-match pattern is the same
composition that underlies "induction heads" in language models. Verify by intervention:
blocking the offset −1 edges in layer 1 should destroy accuracy; blocking them in layer 2
should not.

## Quiz

1. Each position needs information from other positions whose identity depends on content (the subject of this verb, the referent of this pronoun). Attention gives every position direct, parallel, content-addressed access to all earlier positions in one step. A recurrent network instead carries a fixed-size state through the sequence one token at a time, so distant information must survive many updates and positions cannot be processed in parallel.
2. `q`: `(8, 6, 100, 64)`. Scores: `(8, 6, 100, 100)`. Weights: `(8, 6, 100, 100)`. Output after merging heads: `(8, 100, 384)`; after `c_proj`: `(8, 100, 384)`.
3. Exponentials are $1, 2, 5$; weights $(0.125,\ 0.25,\ 0.625)$. Output $= 1.25 + 5 + 18.75 = 25$.
4. Zeroing after the softmax would leave the remaining weights summing to less than 1 (the masked positions took part of the probability mass), and the amount missing would differ per row, changing the scale of the output from position to position. It would also let future scores influence the normaliser, leaking information about the future. $-\infty$ before the softmax removes those positions from the computation entirely.
5. Dot products of $D$-dimensional vectors with unit-variance entries have standard deviation $\sqrt D$; for large $D$ the softmax is nearly one-hot at initialisation and its gradient nearly zero, so heads are hard to train. nanochat RMS-normalises queries and keys and multiplies each by 1.2, so scores are $1.44\sqrt{D}\cos\theta$, bounded and independent of the learned norms.
6. $4 \times 768^2 = 2\,359\,296$. Unchanged by the number of heads (the same matrices are sliced differently). Unchanged by the context length.
7. (a) ×4: projections are per position, linear in $T$. (b) ×16: the tables have $T^2$ entries.
8. $1.44 \times \sqrt{64} = 11.52$, attained when query and key point in exactly the same direction. After RMS-normalisation each vector has norm $\sqrt D$; the dot product is $1.44\,D\cos\theta$, and the kernel divides by $\sqrt D$. Since $\cos\theta \le 1$, no choice of weights can exceed it.
9. The causality test: run the model on a sequence; change only the token at position $i+1$; assert the outputs at positions $\le i$ are unchanged. With an off-by-one mask, the output at position $i$ would change. (Checking that recorded weights are zero above the diagonal also works for an explicit implementation.)
10. (i) The weight shows that the head reads from that position, not what is copied (the OV map may ignore number) or whether the result influences the logits. (ii) Other routes may carry the same information (another token that agrees with the subject, a copy relocated by an earlier layer), so the edge may be unnecessary. Experiment: block that edge (and candidate alternative edges, separately and together, with a control edge) and measure the change in the probability of the correct verb form.
11. The loss gradient with respect to an attention weight $A_{ij}$ is $(\partial L/\partial\mathbf{y}_i)\cdot\mathbf{v}_j$: positive or negative according to whether reading more from $j$ would raise or lower the loss. It propagates through the softmax to the scores and on to $W_Q$ and $W_K$, which are adjusted so that helpful positions score higher. Simultaneously $W_V$ learns to carry what is useful from the positions being read.
12. Of an implementation. The mathematics defines each output as a weighted sum over keys, which can be computed by streaming over the keys in blocks while keeping only running sums and a running normaliser: $O(T)$ extra memory per query block. Materialising the full $T\times T$ matrix is a convenient but unnecessary strategy; Flash Attention avoids it and returns the same result up to rounding.
