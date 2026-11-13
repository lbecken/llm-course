# Resource library

Every module has its own annotated `references.md`. This directory collects the
references by type and gives a reading order. Annotations say what each resource
contributes *to this course*; they are not reviews.

| File | Content |
|---|---|
| this file | the ten references to start with |
| [recommended-reading-path.md](recommended-reading-path.md) | what to read or watch alongside each part, and in which order |
| [papers.md](papers.md) | primary papers by topic, with the module that uses each |
| [books.md](books.md) | books and long-form texts |
| [videos.md](videos.md) | lectures and video courses |
| [websites.md](websites.md) | documentation, blogs and reference sites |

Conventions: **level** is relative to this course's learner (an experienced programmer
who knows classical neural networks). **Historical** marks a source whose ideas remain
current while its details (hyperparameters, benchmarks, model sizes) have aged.
Everything listed was chosen because the course relies on it or because it is the
clearest primary statement of an idea the course teaches. No link is an endorsement of
everything at its destination, and links to commercial books are for identification.

## The ten to start with

| # | Resource | Type | Why it is on this list |
|---:|---|---|---|
| 1 | Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) (source code; this course pins revision `92d63d4`) | code | The course's primary implementation environment: tokenizer, pretraining, fine-tuning, evaluation and inference in about 6 000 lines. Reading it with the modules is the course. |
| 2 | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) | paper · intermediate · historical | The architecture. Read sections 3.1–3.3 after Module 05; note what has changed since (pre-norm, rotary positions, decoder-only use), which Module 06 lists. |
| 3 | Andrej Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2023) | video, 2 h · introductory | A complete character-level GPT built on screen. The best single companion to Part I; the course's MiniGPT is the same construction, extended and tested. |
| 4 | Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd edition draft | book (free online) · introductory–intermediate | The standard text for language modeling, n-grams, perplexity, tokenization and evaluation. The reference for Modules 02, 03 and 10. |
| 5 | Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) | article · advanced | The residual stream as a shared channel, attention heads as read–write operations. The viewpoint of Modules 05 and 06, and the basis for treating attention pictures with caution. |
| 6 | Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (2022) | paper · intermediate | How to split a compute budget between parameters and data: the reasoning behind nanochat's data-to-parameter ratio and Module 09. |
| 7 | Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) | paper · advanced | The clearest demonstration that performance is about memory movement, not FLOPs. Module 12. |
| 8 | Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (2022) | paper · intermediate | The pipeline from base model to assistant: demonstrations, then preferences and reinforcement learning. Part IV. |
| 9 | Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629) (2022) | paper · intermediate | The thought–action–observation loop that Part V builds and bounds. |
| 10 | Sutton, [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019) | essay · introductory | One page on why general methods that scale with computation win. A useful prior for Module 22 and for reading everything else. |

If you read only three things outside this repository: numbers 1, 3 and 4.
