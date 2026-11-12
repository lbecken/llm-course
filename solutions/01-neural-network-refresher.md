# Solutions — Module 01

Try the exercises first. Reference numbers were measured by the course author on a CPU
sandbox (2026-11-11, seed 0); yours can differ in the last digits.

## Lab exercises

### Exercise 1: the gradient by hand

Logits $\mathbf{z} = [1, 3, 0]$, correct class 2.

1. $e^{\mathbf{z}} = [2.718, 20.086, 1.000]$, sum 23.804. $\mathbf{q} = [0.114, 0.844, 0.042]$. Loss $= -\ln 0.042 = 3.170$.
2. $\partial L/\partial\mathbf{z} = \mathbf{q} - \mathbf{p} = [+0.114, +0.844, -0.958]$.
3. New logits $= \mathbf{z} - 1\cdot\text{gradient} = [0.886, 2.156, 0.958]$. New probabilities: $e^{\cdot} = [2.425, 8.638, 2.607]$, sum 13.670, so $q_2 = 0.191$ and the loss is $-\ln 0.191 = 1.657$. One step nearly halved the loss.
4. Smaller. With correct class 1 the gradient would be $[0.114, -0.156, 0.042]$. The gradient is the *error* $\mathbf{q}-\mathbf{p}$: a model that is nearly right receives a small correction, a confidently wrong one a large correction.

**Why a common wrong approach fails.** Writing the gradient as $-1/q_y$ (the derivative of $-\log q_y$ with respect to $q_y$) and stopping there forgets that the logits act on $q_y$ through the softmax, including through the *other* classes' terms in the denominator. The full chain gives $q_i - p_i$ for every $i$, not only for the correct class.

### Exercise 2: break the backward pass

- `dW2` and `db2` are computed *before* the modified line, so their checks still pass. `dW1` and `db1` depend on `dh_pre` and fail: the script reports differences around 1e-2 instead of 1e-11.
- The network still trains, a little worse or slower. Reason: the correct factor is $1-h^2 = (1-h)(1+h)$ and the typo uses $(1-h)$. The two differ by the factor $(1+h)$, which is positive because $h=\tanh(\cdot)\in(-1,1)$. So every entry of `dh_pre` keeps its sign and is only rescaled: the update is still a descent direction, just not the steepest one.
- The lesson: **"it trains" is not evidence that a gradient is correct.** Many backward-pass bugs produce a degraded but working learner. Only the gradient check distinguishes them.

### Exercise 3: capacity and noise

| hidden | test acc (2000 fresh digits) |
|---:|---:|
| 2 | 0.5045 |
| 4 | 0.9130 |
| 8 | 0.9845 |
| 32 | 0.9925 |
| 128 | 0.9920 |

(a) Eight hidden units already exceed 95 %. Two units cannot separate ten classes: the
hidden layer is a 2-dimensional bottleneck through which all information must pass, and
ten clusters do not fit separably into a $\tanh$-squashed plane with these few steps.
(c) The train/test gap grows with capacity relative to data, but here it stays tiny at
every size because the training set (4 000 examples of 10 simple templates) is large for
the problem. Going from 32 to 128 buys nothing: the remaining errors are caused by the
noise, not by lack of capacity.

| flip prob | test acc | comment |
|---:|---:|---|
| 0.0 | 1.0000 | only Gaussian noise: trivially separable |
| 0.08 | 0.9925 | default |
| 0.2 | 0.8800 | 3↔8 and 9→8 dominate: these templates differ in few pixels |
| 0.3 | 0.6255 | |
| 0.4 | 0.3195 | |

(b) About 32 %. This is mostly not the network's fault. At flip probability 0.5 every
pixel would be a fair coin, the image would be independent of the label, and no
classifier could exceed 10 %. At 0.4 each pixel still carries a little information, and
35 such pixels support only a weak inference. The best possible classifier (pick the
template with the smallest Hamming distance, which is the maximum-likelihood decision
for independent flips) also has a high error rate. The irreducible error of a task is
the classification analogue of the entropy floor in Module 02.

### Exercise 4: context length in the window model

Reference point: K = 3 gives MLP 0.4597, count-based 0.4381, floor 0.3139. Expect:

- For small K both models are far above the floor and nearly equal: the context simply does not contain the needed information, and counting is already the optimal use of it (with ample data per context).
- As K grows the count-based model needs a table entry for every distinct K-character context, and each entry is supported by fewer examples; its estimates get noisy and unseen contexts appear. The MLP has to squeeze all contexts through shared weights, which forces it to treat similar contexts similarly: it *generalises*. Neural models overtake counting when contexts become too numerous to observe individually.
- Parameters in $W_1$ grow linearly with K ($K \cdot V \cdot H$); the number of possible contexts grows exponentially ($V^K$). That asymmetry is the reason neural language models exist.

### Challenge: a second hidden layer

Forward: $H_1 = \tanh(XW_1+\mathbf{b}_1)$, $H_2 = \tanh(H_1W_2+\mathbf{b}_2)$, $Z = H_2W_3+\mathbf{b}_3$.
Backward, with $G_Z = (\mathbf{q}-\mathbf{p})/B$:

```python
dW3 = h2.T @ dz;            db3 = dz.sum(0)
dh2 = dz @ W3.T;            dh2_pre = dh2 * (1 - h2**2)
dW2 = h1.T @ dh2_pre;       db2 = dh2_pre.sum(0)
dh1 = dh2_pre @ W2.T;       dh1_pre = dh1 * (1 - h1**2)
dW1 = x.T @ dh1_pre;        db1 = dh1_pre.sum(0)
```

The pattern "multiply by the transposed weights, then by the local slope" repeats once
per layer. That repetition is all that "deep" backpropagation is.

## Quiz

1. A **logit** is an unnormalised real-valued score for one class. **Softmax** exponentiates the logits and divides by their sum, producing positive numbers that sum to 1. **Cross-entropy** is the average negative log-probability the model assigned to the correct class: its average surprise at the truth.
2. $\ln 4096 = 8.318$. The first model starts exactly where an uninformed model should. The second starts 3.6 nats *worse than guessing uniformly*: its initial logits are large, so it makes confident wrong predictions. Look for an output layer (or embedding) initialised with too large a scale. It may still train, but it wastes early steps unlearning its initial confidence.
3. $\mathbf{q} = [1/4, 3/4]$. Loss $= -\ln(1/4) = 1.386$. Gradient $= \mathbf{q}-\mathbf{p} = [-0.75, +0.75]$.
4. `dW1: (35, 32)`, `dh: (64, 32)`, `dlogits: (64, 10)`. Rule: a gradient always has the shape of the tensor it is the gradient of.
5. The local derivatives depend on forward values: the weight gradient of a linear layer is (input)ᵀ × (output error), and the slope of `tanh` is $1-h^2$. `backward` reads `x` and `h` (and the logits). Consequence: all intermediate activations must be kept in memory until the backward pass, so training needs far more memory than inference; techniques such as activation checkpointing and smaller batches trade against this (Module 13).
6. It computes the mean cross-entropy over every position of every sequence in the batch. `logits` is `(B, T, V)` and `targets` is `(B, T)`; `F.cross_entropy` expects `(N, V)` and `(N,)`, so `.view` flattens the batch and time dimensions into $N = B\cdot T$ independent classification examples. `ignore_index=-1` skips positions whose target is −1 (used for masking in fine-tuning, Module 14).
7. Not literally. There are labels; they are not *human-provided*. The target at each position is the next token of the same text, obtained by shifting the sequence by one. "Self-supervised" means the data supervises itself.
8. Switch the check to float64 and reduce $\epsilon$ (e.g. to 1e-5). In float32 the loss carries about 7 significant digits; subtracting two nearly equal losses and dividing by $2\epsilon = 2\times10^{-3}$ leaves an error of order 1e-4 from rounding alone, regardless of whether the gradient is right. Only if the discrepancy survives in float64 is the backward pass wrong.
9. Any three of: (i) fixed context length → attention over the whole context; (ii) no parameter sharing across positions (the same word at a different offset is a different input) → shared token embeddings plus position information; (iii) first-layer size grows with the context → attention parameters are independent of sequence length; (iv) character or word units → subword tokenization; (v) shallow → deep residual stacks with normalization.
10. Identical: the structure linear → non-linearity → linear, trained by backpropagation of a cross-entropy loss. Different: nanochat's version has no biases, uses $\mathrm{relu}(x)^2$ rather than $\tanh$, has a hidden layer four times wider than its input and an output the same size as its input (so it can be added back to the residual stream), is applied independently at every sequence position, and gets its gradients from autograd.
