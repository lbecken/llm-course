# Getting started

Three levels of setup. Start with the first; add the others when a module asks for them.

| Level | You can do | Needs | Time |
|---|---|---|---|
| **1. Course package** | All of Part I and Part V, most labs of Parts III–IV, the whole test-suite | Python ≥ 3.10, PyTorch (CPU is fine) | 2 minutes |
| **2. + pinned nanochat (CPU)** | nanochat walkthrough labs, parity test, the miniature end-to-end nanochat pipeline | `git`, [`uv`](https://docs.astral.sh/uv/), a C compiler | 5–10 minutes |
| **3. + GPU (DGX Spark)** | Real pretraining, SFT, RL, GPU benchmarks | Level 2 on the Spark | see [dgx-spark-setup.md](dgx-spark-setup.md) |

## Level 1: the course package

```bash
git clone https://github.com/lbecken/llm-course.git
cd llm-course
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # torch, numpy, pytest, matplotlib
make test                        # ~15 s; everything should pass, a few tests are skipped
python labs/foundations/01_digits_mlp.py
```

If `pip` picks a multi-gigabyte CUDA build of PyTorch on a machine without a GPU, install
the CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Level 2: add nanochat at the pinned commit

```bash
make env-cpu                                   # clones nanochat@92d63d4 into external/nanochat,
                                               # builds its venv from its lockfile, installs the course into it
source external/nanochat/.venv/bin/activate    # from now on use THIS environment for everything
make doctor                                    # what does this machine support?
make test-nanochat                             # MiniGPT reproduces nanochat's logits
```

Then run the complete nanochat pipeline on a miniature dataset (about two minutes on a
4-core CPU, no downloads of training data):

```bash
make nanochat-local-data      # synthetic corpus -> parquet shards -> nanochat tokenizer
make nanochat-local-train     # scripts.base_train, depth 4, 300 steps
make nanochat-local-eval      # bits per byte + samples through nanochat's Engine
```

What you should see, and what it means, is in
[results/reference/nanochat-local-cpu.md](../results/reference/nanochat-local-cpu.md).

Why one shared environment? nanochat pins exact dependency versions in its lockfile
(notably `torch==2.9.1`). Installing the small course package *into* nanochat's
environment guarantees that course labs and nanochat scripts see the same PyTorch.

## Level 3: the GPU

Follow [dgx-spark-setup.md](dgx-spark-setup.md), then
[hardware-and-costs.md](hardware-and-costs.md) to probe your throughput before any long run.

## How the repository is organised

```text
modules/NN-topic/        lesson.md  lab.md  quiz.md  references.md      <- read these in order
solutions/NN-topic.md    worked solutions for exercises and quizzes     <- after you tried
mlle/                    the course's Python package (small, readable, tested)
labs/<part>/*.py         runnable lab scripts; each begins with its purpose and success criterion
scripts/                 setup, diagnostics, benchmarks, nanochat experiment wrappers
projects/                capstone projects with acceptance criteria and tests
tests/                   CPU test-suite (also run by GitHub Actions)
results/reference/       measurements made by the course author (hardware, date, command)
results/templates/       forms for recording your own experiments
results/runs/            YOUR outputs (git-ignored)
docs/                    setup, hardware, math primer, glossary, upstream audit
resources/               annotated reading list
instructor/              teaching guide, assessments, rubrics
external/nanochat/       the pinned upstream checkout (git-ignored, never modified)
```

Lab scripts use `# %%` cell markers. They run top to bottom from the command line, and
editors such as VS Code or PyCharm (or Jupytext) can also execute them cell by cell like
a notebook. The course ships scripts instead of `.ipynb` files because scripts diff
cleanly, run in CI, and cannot hide stale outputs.

## How to work through a module

1. Read `lesson.md` up to the first "Predict" box and write your prediction down.
2. Run the lab script it names. Compare with your prediction.
3. Do the exercises in `lab.md`; record anything that takes longer than a few seconds in
   `results/runs/` using [the experiment template](../results/templates/experiment-record.md).
4. Take `quiz.md` closed-book. Check against `solutions/`.
5. Move on when you meet the module's mastery criteria (stated at the end of each quiz),
   not when you have run all the commands.

Where to start: [SYLLABUS.md](../SYLLABUS.md) for the full table of contents,
[LEARNING_PATHS.md](../LEARNING_PATHS.md) to choose a track,
[prerequisites.md](prerequisites.md) for a 15-minute self-check.
