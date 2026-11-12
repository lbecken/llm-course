# Module 04 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/04-embeddings.md`](../../solutions/04-embeddings.md).

1. **(Shapes)** `wte` has shape `(32768, 768)`. What is the shape of `wte(idx)` for `idx` of shape `(16, 2048)`? How many of the table's rows can receive a non-zero gradient from this batch, at most?

2. **(Concept)** Explain in one or two sentences why an embedding lookup costs no matmul FLOPs although it is mathematically a matrix product.

3. **(Concept)** What different questions do a token's input embedding and output embedding answer? What does weight tying assume?

4. **(Interpretation)** In a trained model, the cosine similarity between the raw `wte` rows of "cat" and "dog" is 0.05. A colleague concludes the model does not know that cats and dogs are similar. Give two reasons the conclusion does not follow.

5. **(Concept)** Why does a Transformer need position information at all? What exactly is true of attention without it?

6. **(Derivation)** Show that under RoPE (2-D case) the score between a query at position $m$ and a key at position $n$ depends only on $m - n$.

7. **(Code reading)** In nanochat's `apply_rotary_emb`, which channel is paired with channel 0 when the head dimension is 128? Does it matter that the rotation is by $-\theta$ rather than $+\theta$?

8. **(Comparison)** Give one advantage of RoPE over learned absolute positions and one situation in which you would expect no measurable difference.

9. **(Experimental reasoning)** In the position ablation at 800 steps, seed 0 gives agreement 0.99 for RoPE and 0.04 for learned positions. State what you can and cannot conclude, and what you would do next.

10. **(Synthesis)** Describe how the vector at one position of the residual stream changes in meaning from the embedding layer to the final layer, and name the experiment in this module that makes the change visible.

## Mastery criteria

- [ ] Explain embedding lookup, its gradient, and the two embedding matrices.
- [ ] Derive RoPE's relative-position property and map each symbol to `apply_rotary_emb`.
- [ ] State the difference between adding position to the stream and rotating queries/keys.
- [ ] Interpret an embedding-similarity result without over-claiming.
- [ ] At least 8 of 10 quiz questions correct.
