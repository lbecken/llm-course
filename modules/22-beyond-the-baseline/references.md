# Module 22 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020 | paper | intermediate | The paper that named RAG: a retriever and a generator combined, with knowledge that can be replaced without retraining. Read sections 1–2 for the idea and 4.5 for index hot-swapping, the full-scale version of the lab's "changed table". |
| Karpukhin et al., [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906), EMNLP 2020 | paper | intermediate | How learned-embedding retrievers are trained and evaluated (top-k retrieval accuracy), i.e. how to measure the half of the system that the lab replaced by exact matching. |
| Shazeer et al., [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538), ICLR 2017 | paper | intermediate | The modern formulation of MoE layers: gating, sparsity, load balancing. Background for D.3. |
| nanochat, [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LOG.md), entries "Mixture of Experts (negative)" and "DyT for d12 pretraining (negative)" | experiment log | intermediate | Primary source for D.3: two ideas implemented, measured at equal wall-clock time, and rejected, with reasons. |
| Gu and Dao, [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) (2023) | paper | advanced | A prominent state-space alternative to attention. Read the introduction for the motivation and the experiments with D.1's question 3 in mind (equal what?). |
| Dosovitskiy et al., [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929), ICLR 2021 | paper | intermediate | Images as sequences of patch embeddings: the mechanism of D.5 in its original form. |
| Hinton, Vinyals, Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015) | paper | introductory | The original statement of distillation, including why soft targets carry more information than labels. Short and readable. |
| Sutton, [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019) | essay | introductory | One page arguing that general methods that scale with computation outlast clever domain-specific designs. A useful prior when applying D.1's question 4 ("at what scale?") to a new architecture. |
| Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), TACL 2024 | paper | intermediate | Evidence that a long context window is not uniformly used: relevant to both retrieval (where to place passages) and D.4. |

Historical versus current: this is the module most exposed to ageing. The entries above
were chosen as *origins* of ideas rather than as the state of the art, which you should
look up when you read this. The method of D.1 and the habits of D.8 are the durable part.
