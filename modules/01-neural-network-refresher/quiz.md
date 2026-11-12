# Module 01 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/01-neural-network-refresher.md`](../../solutions/01-neural-network-refresher.md).

1. **(Concept)** In one sentence each: what is a logit, what does softmax do to a vector of logits, and what does cross-entropy measure?

2. **(Calculation)** A freshly initialised language model with a vocabulary of 4 096 tokens reports a first training loss of 8.32. A colleague's variant reports 11.9. What do you conclude about each?

3. **(Calculation)** For logits $[0, \ln 3]$ and correct class 0, compute the loss and the gradient with respect to both logits.

4. **(Shapes)** In the two-layer MLP with `X: (64, 35)`, `W1: (35, 32)`, `W2: (32, 10)`: give the shapes of `dW1`, `dh` and `dlogits`, and state the general rule.

5. **(Concept)** Why does the backward pass need tensors that were computed during the forward pass? Name the two that `ManualMLP.backward` reads, and one practical consequence for training large models.

6. **(Code reading)** What does this line from nanochat do, and why is `.view(-1, ...)` there?
   ```python
   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   ```

7. **(Concept)** "Self-supervised learning needs no labels." Is that literally true for next-token prediction? Where do the targets come from?

8. **(Experimental reasoning)** You change a backward pass and the finite-difference check now disagrees at the 1e-4 level in float32, with $\epsilon = 10^{-3}$. Before concluding that the change is wrong, what two things would you alter in the check, and why?

9. **(Concept)** Give three limitations of the fixed-window MLP language model and the mechanism that later addresses each.

10. **(Synthesis)** Name what is identical and what is different between `ManualMLP` and the `MLP` class inside a nanochat block.

## Mastery criteria

Move on to Module 02 when you can:

- [ ] Derive $\partial L/\partial \mathbf{z} = \mathbf{q} - \mathbf{p}$ on paper in under five minutes.
- [ ] Write the six backward-pass equations of the two-layer MLP with shapes, without looking.
- [ ] State the expected initial loss for any number of classes and explain a deviation.
- [ ] Explain to someone else why next-token prediction is classification, and what makes it "self-supervised".
- [ ] Answer at least 8 of the 10 questions above correctly.
