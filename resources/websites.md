# Websites, documentation and blogs

## Source code to read

| Site | Use with | What it contributes |
|---|---|---|
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | every module | The course's primary environment. Always read at the pinned revision ([docs/upstream-version.md](../docs/upstream-version.md)); `dev/LOG.md` and the commit history are part of the reading. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Modules 06–08 | nanochat's simpler predecessor: a GPT-2-style model and training loop in two files. Useful for seeing the classic architecture that Module 06 contrasts with nanochat. |

## Documentation

| Site | Use with | What it contributes |
|---|---|---|
| [PyTorch documentation](https://pytorch.org/docs/stable/index.html) and [tutorials](https://pytorch.org/tutorials/) | all labs | The reference for every tensor operation; the tutorials on autograd, AMP and `torch.compile` are cited in Modules 01 and 13. |
| [Hugging Face documentation](https://huggingface.co/docs) | Modules 03, 10, 11, 14 | How a widely used library exposes tokenizers, generation strategies, caches, chat templates and adapters. Read it for vocabulary and to see conventions that differ from nanochat's. |
| [NVIDIA DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/) | [docs/dgx-spark-setup.md](../docs/dgx-spark-setup.md) | The authoritative source for the machine's software stack. The course's setup guide labels what it took from documentation, from community reports, and what remains to be validated. |

## Explanatory articles

| Site | Use with | What it contributes |
|---|---|---|
| Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) | Module 05 | The best-known visual explanation of attention and the original architecture (encoder–decoder: note the difference from the decoder-only model of this course). |
| Rush et al., [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/) | Modules 05–06 | The original paper interleaved with a working implementation. |
| [Transformer Circuits Thread](https://transformer-circuits.pub/) | Modules 05–06 | Research on what attention heads and MLPs compute, with interventions as the standard of evidence. |
| Weng, [*Lil'Log*](https://lilianweng.github.io/) | Parts IV–V | Long, carefully referenced survey posts (agents, prompt engineering, hallucination, RL). Good maps of a topic's literature; check dates. |
| Jordan, [*Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/) | Module 07 | The optimizer nanochat uses for its matrices, explained by its author. |
| Willison, [*Simon Willison's Weblog*: prompt injection series](https://simonwillison.net/series/prompt-injection/) | Module 16 | A running, practical record of prompt-injection attacks and (non-)defences. |
| OWASP, [*Top 10 for Large Language Model Applications*](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Modules 16, 18 | A maintained checklist of application-level risks. |
| Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) | Modules 18–19 | Workflow and agent patterns, with advice to prefer the simplest. |

## How to use the web for this subject

- Prefer pages that show code or measurements over pages that summarise.
- Check the date. Anything about model capabilities or benchmark standings older than a year is history.
- When two sources disagree about how a current model works, the source code of an open implementation settles it. That is why this course is built on one.
