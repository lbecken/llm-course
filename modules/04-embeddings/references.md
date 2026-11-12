# Module 04 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Su, Lu, Pan, Murtadha, Wen, Liu, [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) | paper | intermediate | The RoPE paper. Section 3.2 is the derivation in this lesson, in complex-number form. |
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017, section 3.5 | paper | intermediate | Sinusoidal positional encoding and the argument about relative offsets being linear functions. |
| Mikolov, Yih, Zweig, [*Linguistic Regularities in Continuous Space Word Representations*](https://aclanthology.org/N13-1090/), NAACL 2013 | paper | introductory | The classic observation that relations (plural, gender, tense) appear as consistent vector offsets. Historical: those were dedicated word-embedding models, not Transformer input tables. |
| Press & Wolf, [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859), EACL 2017 | paper | intermediate | The case for tying input and output embeddings, with analysis of how the two differ when untied. |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | web article | advanced | Introduces the "residual stream" view used throughout this course, and treats embedding/unembedding as the read-in and read-out of that stream. |
| nostalgebraist, [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (2020) | blog post | intermediate | Origin of the logit-lens technique of section D.4, with its caveats. |
| Zhou et al., [*Value Residual Learning*](https://arxiv.org/abs/2410.17897) (2024) | paper | advanced | The "ResFormer" idea behind nanochat's value embeddings: giving later layers a direct path to early value information. nanochat's variant (token-indexed tables with gates) differs from the paper's; treat the paper as motivation. |
| Haviv, Ram, Press, Izsak, Levy, [*Transformer Language Models without Positional Encodings Still Learn Positional Information*](https://arxiv.org/abs/2203.16634) (2022) | paper | advanced | Explains why causal models with no position input are not fully order-blind (relevant to the `pos=none` row of the ablation). |

Historical versus current: learned absolute positions (GPT-2, 2019) and sinusoidal
encodings (2017) have largely been replaced by rotary embeddings in decoder-only LLMs.
The idea that embeddings carry linear feature directions is well supported for many
features but is an empirical regularity, not a guarantee.
