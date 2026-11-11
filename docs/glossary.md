# Glossary

Terms as they are used in this course. "→ NN" points to the module that develops the idea.

| Term | Meaning |
|---|---|
| **Activation checkpointing** | Trading compute for memory in training: discard some intermediate activations and recompute them during the backward pass. → 13 |
| **AdamW** | Optimizer with per-coordinate adaptive step sizes and decoupled weight decay. In nanochat it updates embeddings, the output head and scalars. → 07 |
| **Agent (in this course)** | A program that repeatedly calls a model, executes the action the model chose, feeds the observation back, and stops on an explicit condition. The loop is code, not a property of the model. → 18 |
| **Attention** | Operation in which each position computes a weighted average of *value* vectors from other positions, with weights given by the match between its *query* and their *keys*. → 05 |
| **Autoregressive** | Generating a sequence one element at a time, each conditioned on all previous ones. → 02 |
| **Backout** | nanochat refinement: subtract a learned fraction of the mid-depth residual stream before the final norm. → 06 |
| **Base model** | A model after pretraining only: it continues text; it has not learned the chat format. → 08, 14 |
| **bf16 / bfloat16** | 16-bit float with the exponent range of fp32 and 8 bits of mantissa. nanochat's default compute dtype on modern GPUs. → 13 |
| **Bits per byte (bpb)** | Cross-entropy normalised by the number of UTF-8 bytes of the target tokens, in bits. Comparable across tokenizers. → 10 |
| **Block size / sequence length / context length** | Maximum number of tokens the model processes at once (`sequence_len` in nanochat, `block_size` in MiniGPT). → 02 |
| **BOS** | Beginning-of-sequence token `<\|bos\|>`. Starts every document and every conversation in nanochat. → 03 |
| **BPE** | Byte-pair encoding: build a vocabulary by repeatedly merging the most frequent adjacent pair of tokens. → 03 |
| **Causal mask** | Constraint that position *t* may attend only to positions ≤ *t*. Makes parallel training consistent with left-to-right generation. → 05 |
| **Checkpoint** | Saved model weights (plus optimizer state and metadata) at some training step. |
| **ChatCORE** | nanochat's mean centered accuracy over ARC-Easy, ARC-Challenge, MMLU, GSM8K and HumanEval for chat models. → 10 |
| **Chinchilla** | Hoffmann et al. (2022): for a fixed compute budget, parameters and training tokens should grow together (about 20 tokens per parameter in their fits). → 09 |
| **CORE** | Benchmark from the DCLM paper: centered accuracy averaged over 22 in-context-learning tasks. nanochat's main quality metric for base models. → 10 |
| **Cross-entropy** | Average surprise of the model at the true next token, $-\log q(\text{target})$. The training loss. → 01, 02 |
| **Decode (phase)** | Generating tokens one at a time after the prompt has been processed. Memory-bandwidth-bound. → 11 |
| **DPO** | Direct Preference Optimization: fine-tuning from pairs (preferred, rejected) with a classification-style loss, without a separate reward model or sampling. → 15 |
| **Embedding** | A learned vector associated with a discrete item (a token), stored as a row of a matrix. → 04 |
| **Entropy (of a source)** | Average surprise under the true distribution; the lower bound on any model's cross-entropy. → 02 |
| **FA3 / Flash Attention** | Attention kernels that compute exact attention in tiles that fit in fast memory, never materialising the full T×T matrix. → 12 |
| **FLOPs** | Floating-point operations; nanochat estimates 6 per matmul parameter per training token plus an attention term. → 09 |
| **FP8** | 8-bit floating point (e4m3, e5m2). nanochat can run linear-layer matmuls in FP8 on supporting GPUs. → 13 |
| **GQA** | Grouped-query attention: several query heads share one key/value head, shrinking the KV cache. → 12 |
| **Gradient accumulation** | Summing gradients of several micro-batches before one optimizer step, to emulate a larger batch. → 13 |
| **GRPO** | Group Relative Policy Optimization: policy-gradient method that uses a group of samples per prompt as its own baseline. nanochat's RL script is a simplified relative. → 15 |
| **Hallucination** | Fluent output that is false or unsupported. A consequence of sampling from a model trained to be plausible, not to be correct. → 10 |
| **Head (attention)** | One independent attention computation with its own query/key/value projections. → 05 |
| **KV cache** | Stored keys and values of already-processed positions, so that generating a token needs one position's worth of computation. → 11 |
| **Logits** | Unnormalised scores, one per vocabulary entry; softmax turns them into probabilities. → 01 |
| **LoRA** | Low-rank adaptation: train a small low-rank update to frozen weight matrices. Course extension, not in nanochat. → 14 |
| **Loss mask** | Per-token flag selecting which positions contribute to the loss (assistant tokens in SFT). Implemented as target −1 = ignore. → 14 |
| **MBU** | Model bandwidth utilization: bytes that a decode step must move per second, relative to the device's peak memory bandwidth. → 12 |
| **MFU** | Model FLOPs utilization: achieved model FLOPs per second relative to the device's peak. → 13 |
| **MiniGPT** | This course's small, readable Transformer (`mlle/foundations/minigpt.py`). Not related to other projects of the same name. → 06 |
| **Muon** | Optimizer for weight matrices that orthogonalises the momentum update (all singular values ≈ 1). → 07 |
| **Nat** | Unit of information using natural logarithms; 1 nat = 1.4427 bits. |
| **Perplexity** | $e^{\text{cross-entropy in nats}}$: the effective number of equally likely next tokens. → 02 |
| **Pocket** | The course's synthetic language with a known entropy and a controllable long-range dependency (`mlle/data/pocket.py`). → 02 |
| **Policy gradient / REINFORCE** | Increase the log-probability of sampled outputs in proportion to their (baseline-subtracted) reward. → 15 |
| **Prefill (phase)** | Processing the whole prompt in one parallel forward pass, filling the KV cache. Compute-bound. → 11 |
| **Pre-norm** | Applying normalization to the *input* of each sub-layer, leaving the residual path untouched. → 06 |
| **QK-norm** | Normalising queries and keys before the dot product, so attention scores are scaled cosine similarities. → 06 |
| **Quantization** | Storing or computing with fewer bits per number (int8, int4, fp8). → 12, 13 |
| **RAG** | Retrieval-augmented generation: fetch relevant text and put it in the context. Not in nanochat. → 22 |
| **Residual stream** | The `(B, T, C)` tensor that every block reads from and adds to. → 06 |
| **RLHF** | Reinforcement learning from human feedback: train a reward model from human preferences, then optimise the policy against it. → 15 |
| **RMSNorm** | Normalization that rescales a vector to unit root-mean-square. nanochat's has no learnable parameters. → 06 |
| **RoPE** | Rotary position embedding: rotate query/key channel pairs by position-dependent angles so scores depend on relative position. → 04 |
| **Scaling parameters** | nanochat's term for the parameters used to size the training budget: block matrices plus output head. → 09 |
| **SDPA** | `torch.nn.functional.scaled_dot_product_attention`, PyTorch's fused attention; nanochat's fallback without FA3. → 05, 12 |
| **Self-consistency** | Sample several answers and return the most common one. → 17 |
| **SFT** | Supervised fine-tuning: continue training on curated conversations, usually with a loss mask. → 14 |
| **Smear** | nanochat refinement: mix a gated fraction of the previous token's embedding into the current one. → 06 |
| **Softmax** | Maps logits to a probability distribution: exponentiate, then normalise. → 01 |
| **Special token** | Vocabulary entry reserved for control (`<\|user_start\|>` etc.) that ordinary text can never produce. → 03 |
| **Teacher forcing** | Training each position on the true previous tokens rather than on the model's own earlier outputs. → 02 |
| **Temperature** | Divisor applied to logits before softmax when sampling; lower = more deterministic. → 10 |
| **Test-time compute** | Spending more model calls or tokens per query (sampling, search, verification) to improve results without changing weights. → 17 |
| **Token** | Integer id of a vocabulary entry; the unit the model reads and writes. → 03 |
| **Tool call** | Model output in an agreed format that surrounding code interprets as a request to run something. → 16 |
| **Top-k / top-p** | Restrict sampling to the k most likely tokens / the smallest set with cumulative probability p. → 10 |
| **TPOT / TTFT** | Time per output token / time to first token. → 11 |
| **Value embedding** | nanochat refinement (ResFormer-style): a token-indexed vector added to attention *values* in alternating layers through a learned gate. → 06 |
| **Weight tying** | Using one matrix as both token embedding and output head (GPT-2). nanochat does not. → 06 |
| **Window pattern** | nanochat's per-layer choice between full attention (`L`) and a quarter-context sliding window (`S`). → 12 |
