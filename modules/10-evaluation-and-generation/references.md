# Module 10 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Holtzman, Buys, Du, Forbes, Choi, [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020 | paper | intermediate | Introduces nucleus (top-p) sampling and documents that maximisation-based decoding produces repetitive, atypical text while pure sampling is derailed by the unreliable tail: the phenomena measured in D.4. |
| Li et al., [*DataComp-LM*](https://arxiv.org/abs/2406.11794) (2024), section on evaluation and appendix | paper | intermediate | Definition of the CORE metric (22 tasks, centred accuracy) that nanochat reports. |
| Hendrycks et al., [*Measuring Massive Multitask Language Understanding*](https://arxiv.org/abs/2009.03300) (MMLU, 2021); Cobbe et al., [*Training Verifiers to Solve Math Word Problems*](https://arxiv.org/abs/2110.14168) (GSM8K, 2021); Chen et al., [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) (HumanEval, 2021); Clark et al., [*Think you have Solved Question Answering? Try ARC*](https://arxiv.org/abs/1803.05457) (2018) | papers | intermediate | The benchmarks behind ChatCORE. Read the dataset-construction sections: what is tested, how many items, how answers are checked. |
| Dror, Baumer, Shlomov, Reichart, [*The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*](https://aclanthology.org/P18-1128/), ACL 2018 | paper | intermediate | Which significance tests suit which evaluation setups, including paired bootstrap. The reference for D.6. |
| Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky, [*With Little Power Comes Great Responsibility*](https://arxiv.org/abs/2010.06595), EMNLP 2020 | paper | intermediate | Shows that many reported NLP improvements were tested on too few items to be detectable: sample size, quantified. |
| Bouthillier et al., [*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098), MLSys 2021 | paper | advanced | Measures how much seeds, data order and hyperparameter search each contribute to the variance of a result. |
| Ji et al., [*Survey of Hallucination in Natural Language Generation*](https://arxiv.org/abs/2202.03629) (2022) | survey | intermediate | Definitions and taxonomy of hallucination and an overview of mitigation approaches. The field moves quickly; use it for vocabulary rather than for the state of the art. |
| Hugging Face, [*Text generation strategies*](https://huggingface.co/docs/transformers/generation_strategies) | documentation | introductory | A practical catalogue of decoding methods and their parameters as implemented in a widely used library. |

Historical versus current: benchmark *names* age quickly (several above are considered
saturated or contaminated for frontier models, while remaining useful for small models
such as nanochat's). The statistical methods do not age.
