# Module 12 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Dao, Fu, Ermon, Rudra, Ré, [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135), NeurIPS 2022 | paper | advanced | The source of D.2: tiling, the online softmax, recomputation in the backward pass, and the analysis in terms of memory traffic rather than FLOPs. Read sections 2 and 3. |
| Dao, [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691) (2023); Shah et al., [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608) (2024) | papers | advanced | The engineering refinements behind the kernels nanochat loads. Useful mainly to understand why a kernel is tied to a GPU generation. |
| Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245), EMNLP 2023 | paper | intermediate | Defines grouped-query attention and measures the quality/speed trade-off between multi-head and multi-query attention. |
| Beltagy, Peters, Cohan, [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150) (2020) | paper | intermediate | An early, clear treatment of sliding-window attention combined with a few global connections; the reasoning about receptive field growing with depth applies directly to nanochat's `SSSL` pattern. |
| Databricks, [*LLM Inference Performance Engineering: Best Practices*](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (2023) | article | intermediate | Introduces model bandwidth utilisation and explains TTFT/TPOT and the effect of batching with measurements. The conceptual basis of nanochat's `infer_bench.py`. |
| Jacob et al., [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://arxiv.org/abs/1712.05877), CVPR 2018 | paper | intermediate | The standard formulation of affine quantization and quantization-aware training. Predates LLMs; the arithmetic is unchanged. |
| Frantar, Ashkboos, Hoefler, Alistarh, [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323), ICLR 2023 | paper | advanced | How 3–4 bit weights are made usable for large language models: choose the rounding to minimise each layer's output error instead of rounding to nearest. The step beyond this module's lab. |
| Leviathan, Kalman, Matias, [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192), ICML 2023 | paper | intermediate | The acceptance rule that makes drafting with a small model exact in distribution. Background for the challenge exercise. |
| Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023 | paper | advanced | Continuous batching and paged cache management in a production serving system (vLLM). |

Historical versus current: FlashAttention and GQA are standard components of current
models. Quantization *methods* for very low bit-widths and serving systems change
quickly; the papers above are entry points, not the state of the art.
