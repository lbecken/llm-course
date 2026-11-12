# Module 05 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017 | paper | intermediate | The Transformer paper. Section 3.2 defines scaled dot-product and multi-head attention; the footnote on $1/\sqrt{d_k}$ is the argument of D.3. Read sections 1–3 now, the rest after Module 06. |
| Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (2018) | web article | introductory | The best-known visual walk through queries, keys, values and multi-head attention. Describes the original encoder–decoder model; the decoder-only GPT is the right half of its pictures. |
| Andrej Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2023) | video, 2 h | introductory–intermediate | Develops self-attention from "average the previous tokens" to multi-head attention in code. Watch the first 75 minutes with this module. |
| Bahdanau, Cho, Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473), ICLR 2015 | paper | intermediate | Where attention entered deep learning: as a way for a recurrent decoder to look back at encoder states. Historical; shows the idea before it replaced recurrence. |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | web article | advanced | The QK/OV decomposition of D.5 and the analysis of how heads in different layers compose. The basis for reasoning about what small attention-only models compute. |
| Jain & Wallace, [*Attention is not Explanation*](https://arxiv.org/abs/1902.10186), NAACL 2019; Wiegreffe & Pinter, [*Attention is not not Explanation*](https://arxiv.org/abs/1908.04626), EMNLP 2019 | papers | intermediate | The two sides of the debate summarised in D.12: when attention weights fail as explanations, and what tests they would need to pass to count as one. |
| Henry, Dachapally, Pawar, Chen, [*Query-Key Normalization for Transformers*](https://arxiv.org/abs/2010.04245), Findings of EMNLP 2020 | paper | intermediate | The QK-norm idea used by nanochat: normalise queries and keys and control sharpness with an explicit scale. The paper uses a learned scale; nanochat's is the constant 1.2 × 1.2. |
| PyTorch documentation, [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | documentation | reference | The fused reference used in this module's equivalence tests and as nanochat's fallback; documents `is_causal`, `attn_mask` and `enable_gqa`. |

Historical versus current: attention as an add-on to recurrent networks (2015) is
history; self-attention as the sole sequence-mixing operation (2017) is the present. The
original paper's post-norm layout, sinusoidal positions and encoder–decoder structure
have been replaced in GPT-style models (Module 06), while scaled dot-product attention
itself is unchanged.
