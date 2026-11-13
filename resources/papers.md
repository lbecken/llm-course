# Papers

Primary papers by topic, with the module that uses each. Most are free on arXiv. Read
the abstract and the evaluation section first; the module tells you which sections carry
the idea. H = historical (ideas current, details aged).

## Foundations

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Rumelhart, Hinton, Williams, [*Learning representations by back-propagating errors*](https://www.nature.com/articles/323533a0) (1986) | 01 | H | Backpropagation as most readers of this course first met it. |
| Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html) (2003) | 01, 02, 04 | H · intermediate | Learned word vectors plus an MLP for next-word prediction: the bridge from classical networks to language models. |
| Shannon, [*Prediction and Entropy of Printed English*](https://archive.org/details/bstj30-1-50) (1951) | 02 | H | Language modeling as prediction, and entropy as its measure, decades before neural models. |
| Sennrich, Haddow, Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (2016) | 03 | intermediate | Byte-pair encoding for neural models. |
| Bahdanau, Cho, Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (2015) | 05 | H · intermediate | Attention introduced as a solution to the fixed-vector bottleneck. |
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) | 05, 06, 11 | H · intermediate | The Transformer. |
| Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) | 04, 05 | intermediate | Rotary position embeddings, as used by nanochat. |
| He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (2016) | 06 | H · intermediate | Residual connections. |
| Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019) | 06, 08 | H · introductory | The decoder-only architecture and scale that nanochat's "time to GPT-2" metric refers to. |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | 05, 06 | advanced | Residual stream and attention heads as composable circuits. |
| Kingma and Ba, [*Adam*](https://arxiv.org/abs/1412.6980) (2015); Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (2019) | 07 | intermediate | The adaptive optimizer and its corrected weight decay. |

## Training and evaluation

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) | 09 | intermediate | Loss as a power law in parameters, data and compute. |
| Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (2022) | 09 | intermediate | The compute-optimal ratio of data to parameters. |
| Li et al., [*DataComp-LM*](https://arxiv.org/abs/2406.11794) (2024) | 08, 10 | intermediate | Data curation as the variable; defines the CORE metric nanochat reports. |
| Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (2020) | 10 | intermediate | Nucleus sampling and the failure modes of other decoding methods. |
| Dror et al., [*The Hitchhiker's Guide to Testing Statistical Significance in NLP*](https://aclanthology.org/P18-1128/) (2018) | 10 | intermediate | Which test for which comparison. |
| Bouthillier et al., [*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098) (2021) | 10, 21 | advanced | Sources of variance and how many runs are needed. |

## Performance

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) | 12 | advanced | Exact attention without the quadratic intermediate. |
| Ainslie et al., [*GQA*](https://arxiv.org/abs/2305.13245) (2023) | 12 | intermediate | Grouped-query attention. |
| Pope et al., [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102) (2023) | 11, 12 | advanced | Prefill and decode analysed separately at scale. |
| Kwon et al., [*PagedAttention (vLLM)*](https://arxiv.org/abs/2309.06180) (2023) | 11, 12 | advanced | Serving many sequences: paging the cache. |
| Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) (2018) | 13 | intermediate | Low-precision compute, full-precision master weights. |
| Chowdhery et al., [*PaLM*](https://arxiv.org/abs/2204.02311) (2022), appendix B | 13 | intermediate | Model FLOPs utilisation. |

## From base model to assistant

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Ouyang et al., [*InstructGPT*](https://arxiv.org/abs/2203.02155) (2022) | 14, 15 | intermediate | Demonstrations, reward model, PPO: the standard pipeline. |
| Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685) (2022) | 14 | intermediate | Low-rank adapters. |
| Williams, [*Simple statistical gradient-following algorithms…*](https://link.springer.com/article/10.1007/BF00992696) (1992) | 15 | H · advanced | REINFORCE. |
| Schulman et al., [*Proximal Policy Optimization*](https://arxiv.org/abs/1707.06347) (2017) | 15 | intermediate | What nanochat's RL deliberately omits. |
| Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (2023) | 15 | intermediate | Preferences as a supervised loss. |
| Razin et al., [*Likelihood Displacement in DPO*](https://arxiv.org/abs/2410.08847) (2024) | 15 | advanced | The failure the lab reproduces. |

## Tools, reasoning, agents

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Schick et al., [*Toolformer*](https://arxiv.org/abs/2302.04761) (2023) | 16 | intermediate | Tool calls inside the generated sequence. |
| Greshake et al., [*Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173) (2023) | 16 | intermediate | The attack class through retrieved content and tool results. |
| Nye et al., [*Show Your Work: Scratchpads*](https://arxiv.org/abs/2112.00114) (2021) | 17 | intermediate | Intermediate computation as generated text. |
| Wei et al., [*Chain-of-Thought Prompting*](https://arxiv.org/abs/2201.11903) (2022); Wang et al., [*Self-Consistency*](https://arxiv.org/abs/2203.11171) (2023) | 17 | introductory | Prompted reasoning and voting over samples. |
| Turpin et al., [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388) (2023) | 17 | intermediate | Limits of reasoning text as explanation. |
| Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (2023) | 16, 18 | intermediate | The reason–act–observe loop. |
| Kapoor et al., [*AI Agents That Matter*](https://arxiv.org/abs/2407.01502) (2024) | 18, 19 | intermediate | Cost-controlled, baseline-aware evaluation of agents. |
| Cemri et al., [*Why Do Multi-Agent LLM Systems Fail?*](https://arxiv.org/abs/2503.13657) (2025) | 19 | intermediate | A failure taxonomy from traces. |
| Dwork et al., [*The reusable holdout*](https://www.science.org/doi/10.1126/science.aaa9375) (2015) | 20 | advanced | Validity under adaptive analysis. |

## Beyond the baseline

| Paper | Module | Level | Contribution |
|---|---|---|---|
| Lewis et al., [*Retrieval-Augmented Generation*](https://arxiv.org/abs/2005.11401) (2020) | 22 | intermediate | RAG named and evaluated. |
| Shazeer et al., [*Sparsely-Gated Mixture-of-Experts*](https://arxiv.org/abs/1701.06538) (2017) | 22 | intermediate | MoE layers. |
| Gu and Dao, [*Mamba*](https://arxiv.org/abs/2312.00752) (2023) | 22 | advanced | A state-space alternative to attention. |
| Dosovitskiy et al., [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) (2021) | 22 | intermediate | Images as token sequences. |
| Hinton, Vinyals, Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015) | 22 | introductory | Distillation. |
