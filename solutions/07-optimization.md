# Solutions — Module 07

## Lab exercises

### Exercise 1: Adam by hand

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | update $-\eta\hat m/\sqrt{\hat v}$ | $\theta_t$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.2 | 0.004 | 2.0 | 4.0 | −0.1 | 0.9 |
| 2 | −2 | −0.02 | 0.007996 | −0.10526 | 4.0 | +0.005263 | 0.905263 |
| 3 | 2 | 0.182 | 0.011988 | 0.67159 | 4.0 | −0.033579 | 0.871684 |

2. The first update is exactly $\eta$. When the gradient flips sign, the momentum average nearly cancels and the step shrinks 19-fold; it grows again as the sign repeats. Adam takes full-size steps only where recent gradients agree. $\hat v$ stays at 4 because $g^2$ is constant: the denominator tracks magnitude, not sign.
3. SGD: $1 \to 0.8 \to 1.0 \to 0.8$: steps of full size 0.2 back and forth.
4. `AdamW([w], lr=0.1, betas=(0.9, 0.999), eps=0.0, weight_decay=0.0)` gives 0.9, 0.905263, 0.871684.

### Exercise 2: the milestone run

1. Reference: nanochat style 101,504 parameters, 0.3481, gap 0.034, agreement 1.00 / 1.00, about 100 k tokens/s on 4 CPU cores. GPT-2 style 105,792 parameters, 0.3492.
2. $\ln 25 = 3.219$. The nanochat style starts at 3.2203 (its head is initialised at std 0.001, so predictions are almost exactly uniform); the GPT-2 style at about 3.25 (a tied head with std 0.02 gives slightly non-uniform, slightly wrong initial predictions).
3. (i) Training windows start at random offsets, mostly in the middle of a sentence. The first characters of such a window must be predicted without knowing where in the sentence they are, which costs extra loss that no model with this window can avoid; the entropy floor assumes full context. (ii) Finite training: 1 500 steps of a decaying learning rate; the curve is still falling slowly. (Also valid: finite validation sample noise, about ±0.003.)
4. For example "the fox that chases the child helps the robots ." has singular subject "fox", singular relative-clause verb "chases", singular main verb "helps": correct, despite the intervening "child"/"robots".

### Exercise 3: the learning-rate landscape

See the table in lesson D.6. The GPT-2-style model at lr 0.3 ends at ≈ 2.75. That is the
unigram loss from Module 02 (2.749): the model outputs the same distribution
(character frequencies) at every position, regardless of context. Mechanism: an
oversized early step drives activations into a saturated regime (for example enormous
pre-normalization values or attention scores); gradients through the context-dependent
path become negligible while the output bias-like directions still learn the marginal
frequencies. The model then sits in a state from which the remaining gradient signal
cannot rebuild the context pathway.

### Exercise 4: comparing optimizers fairly

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Muon 0.02 + AdamW 3e-3 | 0.3499 | 0.3487 | 0.3525 |
| AdamW 3e-3 | 0.3574 | 0.3553 | 0.3542 |
| AdamW 1e-2 | 0.3501 | (run it) | (run it) |

A defensible conclusion: "Muon + AdamW is better than AdamW at lr 3e-3 on all three
seeds (by 0.002–0.008), but AdamW at lr 1e-2 matches Muon on seed 0; on this task there
is no evidence that tuned Muon beats tuned AdamW." An indefensible one: "Muon is better",
based on the first two rows alone. Note also that the Muon configuration has two
learning rates, one of which (the AdamW part, 3e-3) was not tuned either.

### Exercise 5: the gradient's path

```python
import types, torch
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig

def no_resid(self, x, cache=None, record=False):
    x = self.attn(self.norm1(x), cache, record)
    return self.mlp(self.norm2(x))

for patch in (False, True):
    torch.manual_seed(0)
    m = MiniGPT(MiniGPTConfig.gpt2_style(50, block_size=32, n_layer=6, n_head=2, n_embd=64))
    if patch:
        for b in m.h:
            b.forward = types.MethodType(no_resid, b)
    idx = torch.randint(0, 50, (8, 32))
    m(idx, idx).backward()
    print([f"{b.attn.c_proj.weight.grad.norm():.2f}" for b in m.h])
```

Measured (untrained model, one random batch):

| layer | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| with residual connections | 0.32 | 0.27 | 0.31 | 0.31 | 0.28 | 0.25 |
| without | 1.43 | 1.73 | 1.80 | 1.68 | 1.71 | 1.91 |

**The expected answer "gradients vanish without residuals" is wrong here.** All six
layers receive gradient of similar size in both cases; without residual connections the
gradients are even larger. The per-sub-layer normalization keeps scales under control in
both directions. So the dramatic failure of the no-residual model in Module 06's
ablation is not explained by vanishing gradients at initialisation. What the
measurements do show: the residual model's gradients are uniform across depth (each
layer sees the same error signal through the identity path), and the no-residual model
is far more sensitive to the learning rate and slower to train. The textbook
vanishing-gradient argument describes deep networks *without* normalization, the
situation of the 1990s.

A lesson in method: an explanation that sounds right (and is right in another setting)
still has to be checked against a measurement in the setting at hand.

### Challenge

Expect warm-up to matter mainly for the GPT-2 style at the higher rate: at $t=1$ Adam's
bias-corrected step is ±η for *every* coordinate regardless of gradient size, and the
model's first gradients come from a random state. One full-size step in an arbitrary
direction is harmless at 3e-3 and can be destabilising at 3e-2 for a model without
bounded activations. The nanochat style, whose blocks start as the identity and whose
activations are bounded, should be nearly indifferent (the lesson's sweep found 0.355
both with and without warm-up at 3e-2).

## Quiz

1. Along the identity (residual) path, unchanged, and through the block's own computation (attention and MLP), where it is multiplied by that block's Jacobian and also produces the gradients of the block's weights. The identity path guarantees every layer and the embeddings receive an error signal that has not been attenuated or amplified by the layers above.
2. One-hot row: $a_s = 1$ for the selected $s$, $a_j = 0$ otherwise. Then $\sum_k a_k g_{A,k} = g_{A,s}$. For $j = s$: $g_{S,s} = 1\cdot(g_{A,s} - g_{A,s}) = 0$. For $j\ne s$: $g_{S,j} = 0\cdot(\dots) = 0$. No gradient reaches the scores, hence none reaches the queries and keys: a saturated head cannot change where it looks.
3. $\hat m_1 = g_1$, $\hat v_1 = g_1^2$, so the update is $-\eta\,g_1/|g_1| = +0.01$ in both cases. The first Adam step has magnitude exactly $\eta$ whatever the gradient's size.
4. L2 regularisation adds $\lambda\theta$ to the gradient, which then passes through Adam's division by $\sqrt{\hat v}$: parameters with large gradients are regularised less, and the effective decay is entangled with the gradient statistics. Decoupled decay multiplies the parameter by $(1-\eta\lambda)$ directly, the same relative shrinkage for every parameter, independent of the adaptive scaling.
5. It replaces the (momentum-averaged) update $USV^\top$ by approximately $UV^\top$: same singular directions, all singular values 1. It uses a few iterations of an odd matrix polynomial (Newton–Schulz / Polar Express) that needs only matrix products. Embedding tables are not linear maps applied to the stream but collections of independent rows, most of which receive no gradient in a step; orthogonalising such an update has no justification, and nanochat keeps them on AdamW.
6. A given embedding row is updated only when its token occurs, a handful of times per batch at most, and affects only those positions: it can and must take large steps to learn in reasonable time. The output head participates in every prediction at every position; its gradient is dense and a large step changes all predictions at once, so it needs a small rate.
7. Linear rise from 1/40 = 0.025 at step 0 to 1.0 at step 39; flat at 1.0 until step 350 (= 1000 − 650); then linear decay to 0.05 at step 1000.
8. Every block adds zero, so the network is `wte → norm → (resid/x0 scaling) → norm → lm_head`: a function of the current token only. With the head near zero the logits are near zero, so the prediction is uniform and the loss is $\ln V$.
9. 2.75 is the unigram cross-entropy of Pocket: the model predicts character frequencies and ignores context. Typical cause: a learning rate too high for the architecture (or another instability) destroyed the context pathway early. 3.22 = ln 25 is the uniform distribution: nothing is being learned at all. Typical causes: gradients not reaching the parameters (optimizer not stepping, learning rate zero, parameters not registered), targets disconnected from inputs, or a loss mask that ignores every position.
10. Whether AdamW was tuned (3e-4 is a default, not an optimum); whether X's own hyperparameters were tuned and how much effort went into each; the number of seeds and the seed-to-seed spread; whether the comparison is at equal steps, equal wall-clock time or equal compute; the model size and data (does it hold at other scales?); and whether the final metric is held-out loss.
11. Parameter-free RMSNorm before every sub-layer and before the head (prevents scale drift of the residual stream from reaching the layers); QK-norm with a fixed scale (prevents attention softmax saturation and the resulting dead gradients); logit soft-cap at ±15 in fp32 (prevents extreme logits, enormous losses and overflow). Also acceptable: zero-initialised projections, Adam's ε.
12. A batch `(x, y)` is already on the device (prefetched). For each of `grad_accum_steps` micro-batches: forward `loss = model(x, y)`; divide the loss by the accumulation count; `backward()` (gradients add up); fetch the next batch. Then compute the schedule values (learning-rate multiplier, Muon momentum, weight decay) and write them into the parameter groups; `optimizer.step()` (AdamW groups and Muon groups); `model.zero_grad(set_to_none=True)`; read the loss value for logging and measure the step time.
