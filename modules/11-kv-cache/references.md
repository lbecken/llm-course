# Module 11 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017), section 3.2.3 and the decoder description | paper | intermediate | The causal decoder whose structure makes incremental decoding possible. The paper does not name a "KV cache"; the technique follows directly from the masked self-attention it defines. |
| nanochat, [`nanochat/engine.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py) and [`tests/test_engine.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/tests/test_engine.py) | source code | intermediate | The primary source for section F: a complete cached, batched, streaming generator in about 300 lines, with tests. |
| Hugging Face Transformers, [*Best practices for generation with cache*](https://huggingface.co/docs/transformers/kv_cache) | documentation | introductory | How a widely used library exposes the same mechanism (dynamic, static and offloaded caches, cache re-use across calls). Useful for vocabulary when you move to other code bases. |
| Pope et al., [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102), MLSys 2023 | paper | advanced | Analyses prefill and decode separately for large models, including batch-size and latency trade-offs and the memory cost of the cache; motivates multi-query attention (one key/value head) as a cache-size measure. |
| Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (vLLM), SOSP 2023 | paper | advanced | What happens when many sequences of different lengths share one machine: cache memory managed in pages, prefix sharing, copy-on-write. The systems consequence of takeaway 4. |
| Databricks, [*LLM Inference Performance Engineering: Best Practices*](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (2023) | article | intermediate | Defines TTFT, TPOT and model bandwidth utilisation (MBU), the metrics nanochat's `infer_bench.py` reports; continued in Module 12. |

Historical versus current: caching past keys and values has been standard practice since
the first Transformer decoders and has not changed in substance. What changes quickly is
the *serving layer* built around it (paging, prefix caches, scheduling); treat the
systems papers as snapshots.
