# Prerequisites and self-check

## Who this course is written for

Someone who is comfortable programming, has met neural networks before (perhaps long
ago: perceptrons, backpropagation, digit recognition), reads mathematics without fear,
and wants an integrated understanding of modern language models rather than a tour of
buzzwords. No prior PyTorch or Transformer experience is assumed.

## What you need

| Area | Needed level | If you are rusty |
|---|---|---|
| Python | functions, classes, list comprehensions, running scripts, virtual environments | any Python tutorial; read `mlle/foundations/mlp.py` as a warm-up |
| Linear algebra | vectors, dot product, matrix multiplication, shapes | [mathematics-primer.md](mathematics-primer.md) §1–2 |
| Calculus | derivative, partial derivative, chain rule | primer §3; Module 01 re-derives backpropagation |
| Probability | distributions, conditional probability, expectation | primer §4–7 |
| Command line | `git`, environment variables, reading logs, `tmux`/`screen` for long runs | [getting-started.md](getting-started.md) |
| PyTorch | none required | Module 01 introduces tensors, autograd and `nn.Module` as they are needed |

Not needed: prior knowledge of Transformers, CUDA programming, distributed systems,
measure-theoretic probability.

## PyTorch in ten lines

Everything the first modules use:

```python
import torch
x = torch.randn(4, 3)                 # a (4, 3) tensor of random numbers
w = torch.randn(3, 2, requires_grad=True)
y = (x @ w).tanh()                    # matrix product, element-wise function -> (4, 2)
loss = y.pow(2).mean()                # a scalar
loss.backward()                       # fills w.grad with d(loss)/d(w), same shape as w
with torch.no_grad():                 # update without recording the operation
    w -= 0.1 * w.grad
w.grad = None                         # reset before the next backward()
print(y.shape, loss.item())           # shapes and Python numbers
```

`torch.nn.Module` is a container that registers parameters; `module(x)` calls its
`forward`. That is all the framework knowledge Part I relies on.

## Fifteen-minute self-check

Answer without running code. Answers are at the bottom; the longer baseline diagnostic,
with a scoring guide for choosing a learning path, is in
[instructor/assessments.md](../instructor/assessments.md).

1. What is the shape of `a @ b` for `a: (8, 16, 32)` and `b: (32, 4)`?
2. A network outputs logits `[2.0, 2.0, 2.0, 2.0]` for a 4-class problem. What is the cross-entropy loss, whatever the correct class is?
3. In gradient descent, what happens if the learning rate is far too large? Far too small?
4. Why do we evaluate on data that was not used for training?
5. You train a classifier for the digits 0–9 and the training loss starts at 2.30. Is that a coincidence?
6. True or false: with a `tanh` hidden layer, initialising all weights to zero is fine because the network can still learn.
7. `x` has shape `(B, T, C)`. What does `x.view(B, T, 4, C // 4).transpose(1, 2)` have as its shape?
8. What does `P(A | B)` mean, and how is `P(A, B)` obtained from it?

<details>
<summary>Answers</summary>

1. `(8, 16, 4)`: `@` multiplies the last two dimensions `(16, 32) @ (32, 4)` and keeps the leading batch dimension.
2. $\ln 4 = 1.386$. Equal logits give a uniform distribution; the loss is $-\ln(1/4)$.
3. Too large: the loss oscillates or diverges (steps overshoot the valley). Too small: the loss decreases, but uselessly slowly.
4. To estimate performance on new inputs. A model can memorise its training set; only held-out data reveals whether it learned something general.
5. No. $\ln 10 = 2.303$ is the loss of a uniform guess over 10 classes, which is what a freshly initialised network should produce.
6. False. With all-zero weights every hidden unit computes the same value and receives the same gradient, so the units never become different from each other (the symmetry is never broken). Here it is worse: the gradient into the first layer is exactly zero because it passes through the zero second-layer weights.
7. `(B, 4, T, C // 4)`: split the last dimension into 4 heads, then move the head dimension next to the batch dimension.
8. The probability of A given that B is known to have happened. $P(A, B) = P(A \mid B)\,P(B)$.

</details>

Six or more correct: start with Module 01. Fewer: read the
[mathematics primer](mathematics-primer.md) first; it takes about an hour.
