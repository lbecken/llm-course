# Module 09 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla), NeurIPS 2022 | paper | intermediate | The budget question and its answer: scale parameters and tokens equally. Section 3 describes three ways of locating the optimum; the appendix gives the exact FLOP formulas nanochat's docstring refers to. |
| Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) | paper | intermediate | The original power laws in parameters, data and compute, and the convention of excluding embedding parameters. Its conclusion about the optimal split was revised by Chinchilla: a useful example of a careful result being refined. |
| Dzmitry Bahdanau, [*The FLOPs Calculus of Language Model Training*](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) (2022) | blog post | introductory | The cleanest derivation of "6 FLOPs per parameter per token"; cited in nanochat's `estimate_flops`. |
| Bergsma et al., [*Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training*](https://arxiv.org/abs/2505.13738) (2025) | paper | advanced | Source of the batch-size exponent 0.383 used by nanochat, and of scaling rules for weight decay. |
| Wang & Aitchison, [*How to set AdamW's weight decay as you scale model and dataset size*](https://arxiv.org/abs/2405.13698) (2024) | paper | advanced | The "$T_\text{epoch}$" framework behind nanochat's weight-decay scaling formula. |
| nanochat, [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LOG.md) entries 2026-01-27 (scaling-law fits under three parameter-counting conventions) and 2026-02-05 (automatic batch size); [Discussion #420, miniseries v1](https://github.com/karpathy/nanochat/discussions/420) | engineering log | intermediate | Scaling laws measured on the code base you are using, with the raw tables. |
| Rich Sutton, [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019) | essay | introductory | Why methods that scale with compute have repeatedly beaten hand-designed ones: the context in which scaling laws matter. An argument, not a theorem. |

Historical versus current: Kaplan (2020) and Chinchilla (2022) are the foundational
results; their specific constants (exponents, 20 tokens per parameter) belong to their
architectures and data. Current practice treats the *method* as standard and re-measures
the constants, as nanochat does.
