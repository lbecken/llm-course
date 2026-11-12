# Modern LLM Engineering

### From Transformer Foundations to Reasoning and Agentic Systems

A self-study course that takes you from "I trained a multilayer perceptron on handwritten
digits once" to a working, tested understanding of a modern GPT-style language model: how
it computes, how it is trained and made fast, how it becomes an assistant, and how tools,
reasoning strategies and agent loops are built around it.

The implementation studied throughout is Andrej Karpathy's
[**nanochat**](https://github.com/karpathy/nanochat), pinned at commit
[`92d63d4`](docs/upstream-version.md) (2026-07-03). The practical work is designed for a
single **NVIDIA DGX Spark**; everything in Parts I and V, and the smallest version of
every other experiment, also runs on a CPU.

```text
Intuition → Mathematics → Small numerical example → Minimal implementation
          → nanochat implementation → Experiment → Interpretation → Knowledge check
```

## Start here

| If you want to… | Go to |
|---|---|
| see the whole table of contents and workload | [SYLLABUS.md](SYLLABUS.md) |
| choose a route (35 h, 80 h or 140 h) | [LEARNING_PATHS.md](LEARNING_PATHS.md) |
| install and run the first lab in five minutes | [docs/getting-started.md](docs/getting-started.md) |
| check whether you have the prerequisites | [docs/prerequisites.md](docs/prerequisites.md) |
| set up a DGX Spark | [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md) |
| understand the whole system on two pages | [docs/architecture-overview.md](docs/architecture-overview.md) |
| know what is finished and what has been verified | [PROGRESS.md](PROGRESS.md) |
| begin | [Module 01](modules/01-neural-network-refresher/lesson.md) |

```bash
git clone https://github.com/lbecken/llm-course.git && cd llm-course
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test                                   # CPU test-suite, about 15 seconds
python labs/foundations/01_digits_mlp.py    # your 1990s digit classifier, backprop by hand
```

## The course in one table

| Part | Modules | Question it answers |
|---|---|---|
| **I. Rebuilding the Foundations** | 01 MLP refresher · 02 language modeling · 03 tokenization · 04 embeddings and position · 05 attention · 06 Transformer block · 07 optimization | *What exactly does a GPT compute, forwards and backwards?* |
| **II. Training a Modern LLM** | 08 data and pretraining · 09 scaling and compute · 10 evaluation and generation | *How is a model actually trained, sized and judged?* |
| **III. Performance Engineering** | 11 KV cache · 12 attention and inference optimization · 13 efficient training | *Why are real systems fast, and what is mathematics vs. kernel vs. system?* |
| **IV. From Base Model to Assistant** | 14 supervised fine-tuning · 15 preference optimization and RL | *How does a text predictor become a chat assistant?* |
| **V. Reasoning, Tools, Agents** | 16 tool use · 17 reasoning and test-time compute · 18 agent loop · 19 multi-agent · 20 bounded self-improvement | *What is in the model and what is orchestration around it?* |
| **VI. Integration and Frontier** | 21 the full system · 22 beyond the baseline | *How does it fit together, and how do I judge what comes next?* |

Three capstones: [train and explain a small GPT](projects/capstone-a-small-gpt/README.md),
[optimize an inference pipeline](projects/capstone-b-inference/README.md),
[build a tool-using agent](projects/capstone-c-agent/README.md).

## What makes this course different

- **Real code, verified.** The course's 250-line Transformer (`mlle/foundations/minigpt.py`)
  reproduces the logits of the real nanochat model to 4·10⁻⁶ once nanochat's five
  refinements are switched off, and a test keeps it that way. Explanations of "what
  nanochat does" were checked against the pinned source, with links to exact lines.
- **A language with a known answer.** A synthetic grammar ("Pocket") with an exactly
  computable entropy lets you compare any model's loss with the theoretical optimum, and
  contains a long-range dependency that counting models provably fail and attention solves.
- **Experiments you can interpret.** Labs ask for a prediction first, change one variable,
  state a success criterion, and write structured results. Where a single seed would
  mislead, the reference results show three.
- **Honest labels.** Every number is marked as *measured* (with hardware, date and
  command), *reported* (with source) or *expected*. No GPU was available while the course
  was written; GPU-dependent statements are marked until you validate them with the
  provided diagnostics.
- **The model/orchestration boundary stays visible.** Tool execution, voting, verification,
  agent loops and multi-agent protocols are built as ordinary code around a model
  interface, and each is measured against a simpler baseline.

## Repository layout

```text
modules/      22 modules: lesson.md, lab.md, quiz.md, references.md
solutions/    worked solutions per module
mlle/         course Python package (foundations, inference, bench, finetune, agents)
labs/         runnable lab scripts, one group per part
scripts/      setup, diagnostics, benchmarks, nanochat experiment wrappers
projects/     capstones with acceptance criteria, fixtures and tests
tests/        CPU test-suite (GitHub Actions runs it on every push)
docs/         setup, hardware and cost model, math primer, glossary, upstream audit
results/      reference measurements, templates, and (git-ignored) your own runs
resources/    annotated reading list
instructor/   teaching guide, assessments, rubrics
```

## Status

See [PROGRESS.md](PROGRESS.md) for the implementation manifest, the validation status of
every part, and the list of GPU-dependent claims awaiting confirmation on a DGX Spark.

## Attribution and license

nanochat is © Andrej Karpathy, MIT License; it is fetched from upstream at the pinned
commit and not redistributed here. This course is an independent project and is released
under the MIT License. Details: [NOTICE.md](NOTICE.md), [LICENSE](LICENSE).
Contributions and corrections: [CONTRIBUTING.md](CONTRIBUTING.md).
