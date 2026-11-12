# Module 06 — Knowledge check

Closed book, about 35 minutes. Solutions: [`solutions/06-transformer-block.md`](../../solutions/06-transformer-block.md).

1. **(Equations)** Write the two equations of a pre-norm Transformer block. What would the post-norm version be, and what is the practical difference?

2. **(Concept)** Give the forward-pass and the backward-pass argument for residual connections, one sentence each.

3. **(Calculation)** Apply parameter-free RMSNorm to $\mathbf{x} = (1, -1, 1, -1, 2, 0, 0, 0)$. What is the RMS of the output? What information about $\mathbf{x}$ is lost?

4. **(Concept)** Attention is linear in the values. Where does a Transformer block get its non-linear computation on content, and why can't attention alone provide it?

5. **(Calculation)** A nanochat-style model has $C = 512$ and $L = 8$. How many parameters are in its blocks (ignore the tiny value-embedding gates)? What fraction of those are in MLPs?

6. **(Shapes)** In which step(s) of the forward pass does a tensor with two sequence-length dimensions appear? Why does that matter for long contexts?

7. **(Comparison)** Name six differences between GPT-2 and nanochat at the pinned revision. Mark which ones change the parameter count.

8. **(Code reading)** What do these lines from nanochat's `forward` do, and what is `x0`?
   ```python
   x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
   ```

9. **(Concept)** What are value embeddings, where do they enter the computation, and why are they cheap in FLOPs but expensive in parameters?

10. **(Testing)** The parity test randomises nanochat's matrices before comparing. Why? And why does it also test that re-enabling each addition *breaks* parity?

11. **(Experimental reasoning)** You remove the final normalization from a trained nanochat model and the loss becomes very large. Explain, using what the un-normalised residual stream is.

12. **(Synthesis)** Starting from token ids of shape `(B, T)`, list every operation up to the probabilities of the next token, with the output shape of each.

## Mastery criteria

- [ ] Draw a block and the complete model from memory, with shapes.
- [ ] Count the parameters of a given configuration by hand.
- [ ] Explain residual connections, pre-norm and the MLP's role without notes.
- [ ] List the GPT-2 → nanochat replacements and the five additions, and say what each computes.
- [ ] Explain what the parity test establishes.
- [ ] At least 10 of 12 quiz questions correct.
