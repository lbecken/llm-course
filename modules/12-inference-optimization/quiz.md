# Module 12 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/12-inference-optimization.md`](../../solutions/12-inference-optimization.md).

1. **(Concept)** Name the three resources an inference workload consumes, and for each one a situation in which it is the binding limit.

2. **(Concept)** FlashAttention is described as "exact". Exact with respect to what, and what does it not store that the naive implementation does?

3. **(Calculation)** Online softmax for one query: after the first chunk, $m=2$, $\ell=1.5$, $o=6$. The next chunk has scores $(4, 2)$ and scalar values $(10, 0)$. Give the new $(m, \ell, o)$ and the output so far.

4. **(Concept)** Why is a Python implementation of the tiled algorithm not faster than naive attention at moderate $T$, while a GPU-kernel implementation is?

5. **(Calculation)** A 6-layer model uses sliding windows of 256 in every layer. What is the largest distance over which information can, in principle, reach the output position? Why might the model still fail on dependencies shorter than that?

6. **(Concept)** What does grouped-query attention reduce, and what does it leave unchanged? What is the special case $H_{kv}=1$ called?

7. **(Calculation)** Weights: 4 GB. Cache read per decode step per sequence: 0.5 GB. Bandwidth: 200 GB/s. Bound on tokens/s at batch 1 and at batch 8?

8. **(Concept)** Define MBU and MFU. At a decode batch of 1, which is the informative one, and why?

9. **(Calculation)** Quantize $(1.0, 0.3, -0.26)$ symmetrically to 4 bits with one scale. Give scale, codes and reconstruction.

10. **(Concept)** Why do per-channel scales give lower error than a per-tensor scale at the same bit-width?

11. **(Method)** A colleague replaces full attention by a window of 128 in a trained model *at inference only* and reports unchanged validation loss on short documents. Name two problems with the conclusion "the window is free".

12. **(Code reading)** In nanochat, when is Flash Attention 3 used, and what happens to a sliding-window layer when it is not?

## Mastery criteria

- [ ] Carry out the online-softmax update by hand and explain why it is exact.
- [ ] State what sliding windows and GQA save, and design a test for what they might cost.
- [ ] Compute a bandwidth bound and interpret TTFT, TPOT, MBU and MFU.
- [ ] Quantize a tensor per tensor and per channel and predict the direction of the errors.
- [ ] Classify optimisations as exact or approximate and validate each kind correctly.
- [ ] At least 10 of 12 quiz questions correct.
