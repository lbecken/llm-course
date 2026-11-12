# Module 06 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Andrej Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2023), second half; and [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) | video + code | intermediate | Builds the GPT-2-style block (residuals, LayerNorm, MLP, stacking) step by step. nanoGPT's `model.py` is the cleanest reference for the classic architecture that MiniGPT's `gpt2_style` mirrors. |
| Radford, Wu, Child, Luan, Amodei, Sutskever, [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019), section 2.3 | paper | intermediate | The source of the pre-norm decoder-only layout and the scaled residual initialisation. Short on architectural detail; read with nanoGPT open. |
| He, Zhang, Ren, Sun, [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385), CVPR 2016 | paper | intermediate | Residual connections, introduced for image networks. Section 3 gives the "learn an edit, not a replacement" argument. |
| Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745), ICML 2020 | paper | advanced | Why pre-norm trains more stably than post-norm (gradient scale at initialisation). |
| Ba, Kiros, Hinton, [*Layer Normalization*](https://arxiv.org/abs/1607.06450) (2016); Zhang & Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467), NeurIPS 2019 | papers | intermediate | The two normalizations of D.2. The RMSNorm paper argues the re-centering of LayerNorm is dispensable. |
| Hendrycks & Gimpel, [*Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415) (2016) | paper | intermediate | The activation used by GPT-2. |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | web article | advanced | The residual-stream view of a Transformer: blocks as readers and writers of a shared linear space. |
| Keller Jordan et al., [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | code | advanced | The speed-running code base from which nanochat took QK-norm, ReLU², value embeddings, residual scalars and Muon. Its README lists each change with its measured effect; the best single source for "why does nanochat look like this". |
| nanochat, [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LOG.md) | engineering log | intermediate | Dated experiments, including many negative results, behind the pinned architecture. Primary source for the "reason / evidence" column of D.7. |

Historical versus current: the 2017 block (post-norm, ReLU, sinusoidal positions) and the
2019 GPT-2 block (pre-norm LayerNorm, GELU, learned positions, tied head) are the
ancestors. The pinned nanochat block represents one current small-model recipe; large
production models differ again in details (for example SwiGLU MLPs and grouped-query
attention are common), and those details keep changing. The residual stream, pre-norm
and the attention/MLP alternation have been stable since 2019.
