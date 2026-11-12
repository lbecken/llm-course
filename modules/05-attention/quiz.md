# Module 05 — Knowledge check

Closed book, about 35 minutes. Solutions: [`solutions/05-attention.md`](../../solutions/05-attention.md).

1. **(Concept)** State in two sentences the problem attention solves and how it differs from a recurrent network's solution.

2. **(Shapes)** For `x` of shape `(8, 100, 384)` and 6 heads, give the shapes of: `q` after splitting heads, the score tensor, the attention weights, the output before and after `c_proj`.

3. **(Calculation)** A query has scaled scores $(0, \ln 2, \ln 5)$ against three visible keys. Compute the attention weights. If the values are the scalars 10, 20, 30, what is the output?

4. **(Concept)** Why is the mask applied by setting scores to $-\infty$ *before* the softmax, rather than zeroing weights *after* it?

5. **(Concept)** What goes wrong at initialisation if scores are not scaled by $1/\sqrt{D}$ and $D$ is large? How does nanochat control the scale instead?

6. **(Calculation)** How many parameters does a multi-head attention layer with $C = 768$ and no biases have? How does this change if the number of heads goes from 6 to 12? If the context length doubles?

7. **(Cost)** A layer processes $T = 2048$ tokens. By what factor do (a) the projection cost and (b) the cost of the score tables change when $T$ becomes 8192?

8. **(Code reading)** In nanochat, after `q, k = norm(q), norm(k)` and the two multiplications by 1.2, what is the largest possible attention score before the softmax for `head_dim = 64`? Why can it not exceed that?

9. **(Testing)** Describe a test that would catch a bug in which the causal mask is off by one, letting each position see one token of the future.

10. **(Interpretation)** A head's attention weight from the verb position to the subject noun is 0.9. Give two reasons why this alone does not establish that the model uses that edge to decide the verb's number, and describe the experiment that would.

11. **(Training)** There is no loss on attention weights. Explain how a head nevertheless learns to attend to useful positions.

12. **(Synthesis)** "The formula for attention has $O(T^2)$ memory." Is that a property of the mathematics or of an implementation? Explain.

## Mastery criteria

- [ ] Compute a 3×3 causal attention example by hand, including the softmax.
- [ ] Write multi-head attention in PyTorch from memory, with correct shapes.
- [ ] Explain queries/keys/values, scaling, masking and the output projection, each in one sentence.
- [ ] State parameter count and time/memory complexity and what each depends on.
- [ ] Name three tests for an attention implementation and write the causality test.
- [ ] Explain, with the intervention experiment, why attention weights are not explanations.
- [ ] At least 10 of 12 quiz questions correct.
