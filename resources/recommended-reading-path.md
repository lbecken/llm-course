# Recommended reading path

What to read or watch alongside each part. "Before" items make the modules easier;
"after" items deepen them. Nothing here is required: every module is self-contained.
Details and further items are in each module's `references.md`.

## Before starting

| If the diagnostic showed | Then |
|---|---|
| rusty calculus or linear algebra | [docs/mathematics-primer.md](../docs/mathematics-primer.md); 3Blue1Brown's [neural networks series](https://www.3blue1brown.com/topics/neural-networks), chapters 1–4 |
| no PyTorch | the official [*Learn the Basics*](https://pytorch.org/tutorials/beginner/basics/intro.html) tutorial (tensors, autograd, optimisation loop): about three hours |
| MLPs are a distant memory | Karpathy's [micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) (2.5 h) |

## Part I — Foundations (Modules 01–07)

| When | Resource | Why now |
|---|---|---|
| with 02 | Jurafsky and Martin, [*SLP3*](https://web.stanford.edu/~jurafsky/slp3/), chapter on n-gram language models | entropy, perplexity and smoothing in full |
| with 03 | Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (2 h) | BPE built on screen; compare with `mlle/foundations/bpe.py` |
| with 05 | Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) | pictures for what Module 05 derives |
| after 05 | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), sections 3.1–3.3 | the primary source, now readable |
| with 06 | Karpathy, [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) | the whole model in one sitting |
| after 06 | Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html), the first half | the residual-stream view |
| with 07 | Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101); Jordan, [*Muon*](https://kellerjordan.github.io/posts/muon/) | the two optimizers nanochat combines |

## Part II — Training language models (Modules 08–10)

| When | Resource | Why now |
|---|---|---|
| with 08 | Karpathy, [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU) (4 h), the data-loading and training-loop sections | a full pretraining run narrated; nanochat is its successor |
| with 09 | Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361); Hoffmann et al., [*Chinchilla*](https://arxiv.org/abs/2203.15556) | where the sizing rules come from |
| with 10 | Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) | why decoding settings matter |
| after 10 | Dror et al., [*The Hitchhiker's Guide to Testing Statistical Significance in NLP*](https://aclanthology.org/P18-1128/) | the statistics the rest of the course assumes |

## Part III — Performance engineering (Modules 11–13)

| When | Resource | Why now |
|---|---|---|
| with 11 | nanochat `nanochat/engine.py` and `tests/test_engine.py`, read end to end | a complete cached, batched, streaming generator |
| with 12 | Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135), sections 2–3 | tiling and the online softmax from the source |
| with 12 | Databricks, [*LLM Inference Performance Engineering: Best Practices*](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) | TTFT, TPOT and bandwidth utilisation with measurements |
| with 13 | Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) | short; the rule that master weights stay in fp32 |

## Part IV — From base model to assistant (Modules 14–15)

| When | Resource | Why now |
|---|---|---|
| with 14 | Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685), sections 4 and 7 | then compare with the lab's result on which matrices to adapt |
| with 15 | Sutton and Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), chapter 13 | policy gradients derived properly |
| with 15 | Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290), section 4 | the derivation the lesson sketches |
| after 15 | Ouyang et al., [*InstructGPT*](https://arxiv.org/abs/2203.02155) | the whole pipeline in one paper |

## Part V — Reasoning, tools and agents (Modules 16–20)

| When | Resource | Why now |
|---|---|---|
| with 16 | Greshake et al., [*Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173) | the attack class, systematically |
| with 17 | Nye et al., [*Show Your Work*](https://arxiv.org/abs/2112.00114); Turpin et al., [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388) | what scratchpads do, and what reasoning text does not prove |
| with 18 | Russell and Norvig, [*AIMA*](https://aima.cs.berkeley.edu/), chapters 2–3 | agents and search, the classical framing |
| with 19 | Kapoor et al., [*AI Agents That Matter*](https://arxiv.org/abs/2407.01502) | cost-controlled evaluation of agents |
| with 20 | Dwork et al., [*The reusable holdout*](https://www.science.org/doi/10.1126/science.aaa9375) | why adaptive search needs guarded evaluation |

## Part VI — Integration and frontier (Modules 21–22)

| When | Resource | Why now |
|---|---|---|
| with 21 | Sculley et al., [*Hidden Technical Debt in Machine Learning Systems*](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | the system around the model |
| with 22 | Lewis et al., [*Retrieval-Augmented Generation*](https://arxiv.org/abs/2005.11401); nanochat `dev/LOG.md` (the negative entries) | one positive primary source and one negative one |
| last | Sutton, [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) | read it again after the course |

## After the course

A second pass in a different voice consolidates: Raschka's
[*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch)
rebuilds Parts I, II and IV in a GPT-2-style code base; Stanford's
[CS336](https://cs336.stanford.edu/) covers the same ground as a university course with
heavier systems assignments. Then pick one item from Module 22 and reimplement its
smallest version.
