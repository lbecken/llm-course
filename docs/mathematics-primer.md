# Mathematics primer

A refresher of exactly the mathematics the course uses, in the order it is needed. If you
trained multilayer perceptrons in the 1990s, most of this is a change of notation rather
than new material. Each section ends with where it is used.

Notation used throughout the course: scalars are lowercase italic ($x$), vectors are bold
lowercase ($\mathbf{x}$), matrices are uppercase ($W$). Shapes are written like code:
`(B, T, C)`. Vectors are **rows** unless stated otherwise, matching PyTorch, where a
linear layer computes $\mathbf{y} = \mathbf{x} W^\top + \mathbf{b}$.

## 1. Vectors, dot products, and what "similar" means

The dot product of two vectors of length $d$:

$$\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{d} a_i b_i = \lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert \cos\theta$$

It is large and positive when the vectors point the same way, zero when they are
perpendicular, negative when they point in opposite directions. Dividing out the lengths
gives the cosine similarity $\cos\theta \in [-1, 1]$.

One fact about high dimensions that matters later: two *random* vectors with independent
entries of variance 1 have a dot product with mean 0 and standard deviation $\sqrt{d}$.
That $\sqrt{d}$ is why attention divides its scores by $\sqrt{d}$ (Module 05).

*Used in:* attention scores (05), embeddings (04), QK-norm (06).

## 2. Matrices as functions; shapes

A matrix $W$ of shape `(out, in)` maps a vector of length `in` to one of length `out`.
Matrix multiplication $AB$ requires the inner sizes to agree: `(m, n) @ (n, p) -> (m, p)`.

PyTorch extends this to batches: `@` multiplies the **last two** dimensions and treats all
earlier dimensions as independent copies.

```python
q.shape == (B, H, T, D)
k.shape == (B, H, T, D)
(q @ k.transpose(-2, -1)).shape == (B, H, T, T)   # for each batch item and head: a T x T table
```

**Broadcasting**: when shapes differ, dimensions of size 1 (or missing leading dimensions)
are repeated as needed. `(B, T, C) + (T, C)` adds the same `(T, C)` table to every batch
item. Most shape bugs are unintended broadcasts; the lessons print shapes for this reason.

*Used in:* everywhere. Module 06's trace lab prints every shape of a full forward pass.

## 3. Derivatives, gradients, the chain rule

For a function $L(\theta)$ of many parameters, the gradient $\nabla_\theta L$ collects the
partial derivatives $\partial L/\partial\theta_i$: how much $L$ changes per unit change of
each parameter, holding the others fixed. Gradient descent moves against it:

$$\theta \leftarrow \theta - \eta\,\nabla_\theta L$$

The chain rule is the whole of backpropagation. If $L$ depends on $\mathbf{y}$ and
$\mathbf{y} = f(\mathbf{x})$, then

$$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i}\,\frac{\partial y_i}{\partial x_j}$$

Two cases cover almost every layer in this course. Let $G_Y = \partial L/\partial Y$ be the
gradient arriving from above, with the same shape as $Y$.

| Forward | Backward | In words |
|---|---|---|
| $Y = XW$ | $\dfrac{\partial L}{\partial W} = X^\top G_Y$, $\quad\dfrac{\partial L}{\partial X} = G_Y W^\top$ | each weight's gradient is (its input) × (its output's error); errors flow back through the same weights |
| $Y = f(X)$ element-wise | $\dfrac{\partial L}{\partial X} = G_Y \odot f'(X)$ | scale each error by the local slope |

A consequence worth remembering: adding is free for gradients. If $Y = X + F(X)$ (a
residual connection), then $\partial L/\partial X = G_Y + (\text{gradient through } F)$:
the incoming gradient passes through unchanged, plus a correction. This is why very deep
residual networks can be trained (Module 06).

*Used in:* Module 01 derives and checks these by hand; Module 07 follows them through attention.

## 4. Probability: distributions, conditionals, the chain rule again

A discrete distribution over $V$ outcomes is a vector $\mathbf{p}$ with $p_i \ge 0$ and
$\sum_i p_i = 1$. A conditional distribution $P(y\mid x)$ is one such vector for each $x$.

The probability chain rule factorises any joint distribution without approximation:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

A language model *is* the family of conditionals on the right-hand side (Module 02).

## 5. Softmax and the log-sum-exp

Softmax turns $V$ arbitrary real scores (**logits**) $\mathbf{z}$ into a distribution:

$$\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Properties used repeatedly:

- Adding a constant to all logits changes nothing. Only *differences* between logits matter.
- Multiplying logits by $1/\tau$ (temperature $\tau$) sharpens ($\tau<1$) or flattens ($\tau>1$) the distribution; $\tau \to 0$ gives the arg-max, $\tau \to \infty$ gives uniform (Module 10).
- A logit of $-\infty$ gives probability exactly 0. Masks use this (Module 05).
- Its derivative: $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$.

Numerically, $\log\sum_j e^{z_j}$ is computed after subtracting $\max_j z_j$ so that the
largest exponent is $e^0 = 1$; otherwise `exp` overflows for logits above about 88 in
single precision.

## 6. Information: surprise, entropy, cross-entropy, KL divergence

The **surprise** of an event with probability $p$ is $-\log p$: certain events carry 0,
rare events carry a lot. With $\log_2$ the unit is bits, with $\ln$ it is nats
(1 nat = 1.4427 bits).

| Quantity | Formula | Meaning |
|---|---|---|
| Entropy | $H(p) = -\sum_i p_i \log p_i$ | average surprise of a source; the best achievable average code length |
| Cross-entropy | $H(p, q) = -\sum_i p_i \log q_i$ | average surprise when reality follows $p$ but you predict with $q$ |
| KL divergence | $D_{\mathrm{KL}}(p\Vert q) = H(p,q) - H(p) \ge 0$ | the avoidable part: extra surprise caused by using $q$ instead of $p$ |
| Perplexity | $e^{H}$ (nats) or $2^{H}$ (bits) | "effective number of equally likely choices" |

Because $D_{\mathrm{KL}} \ge 0$, **no model can have cross-entropy below the entropy of the
data source**. For the synthetic Pocket language the course knows that number exactly,
which turns "is the loss good?" into a precise question (Module 02).

In training, $p$ is the one-hot distribution of the token that actually came next, so the
cross-entropy for one position reduces to $-\log q_{\text{target}}$: the surprise of the
model at the truth.

The gradient of softmax followed by cross-entropy with respect to the logits is
remarkably simple, and is derived in Module 01:

$$\frac{\partial L}{\partial \mathbf{z}} = \mathbf{q} - \mathbf{p}\quad(\text{prediction minus target})$$

## 7. Expectation, variance, and why initialisation scales by $1/\sqrt{n}$

If $y = \sum_{i=1}^{n} w_i x_i$ with independent zero-mean terms, then
$\mathrm{Var}(y) = n\,\mathrm{Var}(w)\,\mathrm{Var}(x)$. To keep the signal's variance the
same from layer to layer, choose $\mathrm{Var}(w) = 1/n$, i.e. standard deviation
$1/\sqrt{n}$ where $n$ is the number of inputs (the *fan-in*). A uniform distribution on
$[-s, s]$ has standard deviation $s/\sqrt{3}$, which is where nanochat's
$s = \sqrt{3}/\sqrt{n_\text{embd}}$ comes from (Module 07).

Sample means: averaging $N$ independent measurements with standard deviation $\sigma$
gives a standard error of $\sigma/\sqrt{N}$. An accuracy measured on $N$ examples has
standard error $\sqrt{a(1-a)/N}$; with $N = 200$ and $a = 0.5$ that is ±3.5 percentage
points. Keep this in mind whenever two results are compared (Module 10).

## 8. Rotations in the plane

Rotating the point $(x_1, x_2)$ by angle $\theta$:

$$\begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

Rotations preserve lengths, and rotating two vectors by $\alpha$ and $\beta$ changes
their dot product only through the difference $\alpha - \beta$. Rotary position
embeddings are built on exactly this (Module 04).

## 9. Singular value decomposition (only for Module 07)

Every matrix factors as $G = U S V^\top$ with orthonormal columns in $U$ and $V$ and
non-negative singular values on the diagonal of $S$. Geometrically: rotate, stretch along
axes by $S$, rotate. The Muon optimizer replaces a gradient-like matrix $G$ by
$UV^\top$, i.e. it keeps the directions and sets every stretch factor to 1.

## 10. Counting work: big-O and FLOPs

- A matrix product `(m, n) @ (n, p)` costs about $2mnp$ floating-point operations (one
  multiply and one add per term).
- A linear layer with $P$ weights applied to one token costs about $2P$ FLOPs forward and
  about $4P$ backward, hence nanochat's "6 FLOPs per matmul parameter per token" (Module 09).
- Attention over a sequence of length $T$ builds a $T \times T$ table per head: time and
  (naively) memory grow as $T^2$ (Modules 05 and 12).

## Self-check

You are ready for Module 01 if you can do these without looking anything up:

1. Give the shape of `x @ W` for `x: (32, 35)`, `W: (35, 16)`.
2. Compute $\mathrm{softmax}([0, 0, \ln 2])$.
3. A fair die is modelled by a model that puts probability 0.5 on "6" and 0.1 on each other face. Compute the cross-entropy in nats, and say what the entropy of the die is.
4. For $y = \tanh(wx)$, write $\partial y / \partial w$.

Answers: (1) `(32, 16)`. (2) $[0.25, 0.25, 0.5]$. (3) $\tfrac{1}{6}(5\ln 10 + \ln 2) = 2.034$ nats; entropy $\ln 6 = 1.792$ nats. (4) $x\,(1-\tanh^2(wx))$.
