# Module 11 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/11-kv-cache.md`](../../solutions/11-kv-cache.md).

1. **(Concept)** State the property of the model that makes a KV cache valid. Which component enforces it?

2. **(Concept)** Why are keys and values cached but not queries?

3. **(Calculation)** A model has 12 layers, 6 key/value heads of dimension 128, and caches in 16-bit floats. Give the cache size per token and for a batch of 8 sequences of 4 096 tokens.

4. **(Calculation)** Naive generation of 200 tokens after a 100-token prompt: how many positions are processed in total? And with a cache?

5. **(Concept)** Define prefill and decode. Which one determines time to first token? Which is limited mainly by memory bandwidth, and why?

6. **(Diagnosis)** After adding a KV cache, a RoPE model's first generated token is always right and later tokens are subtly wrong. Name the most likely bug.

7. **(Diagnosis)** A cached generator matches the naive one for prompts fed one token at a time, but not when a 10-token prompt is prefilled in one call. Name the most likely bug.

8. **(Code reading)** nanochat's `KVCache` has a field `prev_embedding`. Why does a cache of keys and values not suffice for nanochat's model?

9. **(Concept)** You need 16 samples for one prompt. Describe the efficient procedure and what is copied.

10. **(Critical thinking)** "KV caching made our model 7× faster." List two pieces of information without which this statement cannot be interpreted.

11. **(Concept)** Does a KV cache speed up training? Why or why not?

12. **(Design)** Greedy outputs of cached and naive generation are identical on your test prompt, yet under bf16 on a GPU a colleague sees a different token at position 310 of one long sample. Is the cache broken? How do you decide?

## Mastery criteria

- [ ] Derive the KV cache from causality and say exactly what is stored.
- [ ] Compute cache memory from a configuration.
- [ ] Implement prefill and decode with correct positions, and test equivalence against a full forward pass.
- [ ] Explain why the speed-up depends on sequence length and model size.
- [ ] Trace nanochat's `Engine.generate` from prompt to streamed tokens.
- [ ] At least 10 of 12 quiz questions correct.
