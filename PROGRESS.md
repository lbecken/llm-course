# Progress manifest

Single source of truth for what exists, what has been verified, and what remains.
Update it in the same commit as the work it describes.

| | |
|---|---|
| Upstream pin | nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (2026-07-03), audited 2026-11-11 |
| Working branch | `claude/cool-cannon-10dksz` |
| Verification commands | `make test` · `make test-all` · `make test-nanochat` · `make check-docs` |
| Hardware available to the author | 4-vCPU x86-64 sandbox, no GPU. **No DGX Spark.** |
| State | **All phases complete** (2026-11-13). 131 tests pass; documentation checker reports no problems |

## Status legend

- **Done**: written in full, cross-checked, and (for code) covered by passing tests.
- **Draft**: usable but not yet cross-checked end to end.
- **Pending**: not written.
- Validation: **CPU-verified** (executed by the author), **source-verified** (checked by reading the pinned source), **awaiting GPU** (needs a DGX Spark run by the learner).

## Phase 1 — Research and architectural audit

| Deliverable | File | Status |
|---|---|---|
| Upstream revision, dependencies, source map, capability matrix | [docs/upstream-version.md](docs/upstream-version.md) | Done · source-verified, parity CPU-verified |
| Hardware compatibility and cost assessment | [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md), [docs/hardware-and-costs.md](docs/hardware-and-costs.md) | Done · source-verified + reported; GPU items awaiting GPU |
| End-to-end CPU validation of the pinned pipeline | [results/reference/nanochat-local-cpu.md](results/reference/nanochat-local-cpu.md) | Done · CPU-verified |

## Phase 2 — Curriculum and repository foundation

| Deliverable | File | Status |
|---|---|---|
| Table of contents with workload | [SYLLABUS.md](SYLLABUS.md) | Done |
| Learning paths | [LEARNING_PATHS.md](LEARNING_PATHS.md) | Done |
| Topic-to-file map | [COURSE_MAP.md](COURSE_MAP.md) | Done |
| Setup, prerequisites, math primer, glossary, troubleshooting, architecture overview | [docs/](docs/) | Done |
| Package, Makefile, CI, documentation checker | `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `scripts/diagnostics/check_links.py` | Done · CPU-verified |

## Phases 3–6 — Modules

Columns: L = lesson, B = lab sheet, Q = quiz, R = references, S = solutions, C = code,
lab scripts and tests. Every lab script was executed by the author on CPU and its output
is what the lesson reports.

| # | Module | L | B | Q | R | S | C | Validation |
|---:|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 01 | Neural network refresher | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified |
| 02 | Language modeling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified |
| 03 | Tokenization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified; nanochat tokenizer CPU-verified |
| 04 | Embeddings and position | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified (3 seeds for the position ablation) |
| 05 | Attention | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified |
| 06 | Transformer block | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified; nanochat logit parity CPU-verified |
| 07 | Optimization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified |
| 08 | Data and pretraining | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified incl. unmodified nanochat pipeline on local data; full-scale run awaiting GPU |
| 09 | Scaling and compute | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | sizing arithmetic source- and CPU-verified; durations awaiting GPU |
| 10 | Evaluation and generation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified; CORE awaiting GPU |
| 11 | KV cache | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified incl. nanochat Engine; `infer_bench` awaiting GPU |
| 12 | Inference optimization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified; GPU kernels and `infer_bench` awaiting GPU |
| 13 | Efficient training | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified; bf16/FP8 GPU timings awaiting GPU |
| 14 | Supervised fine-tuning | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified (course models); `chat_sft` awaiting GPU |
| 15 | RL and preferences | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified (course models); `chat_rl` awaiting GPU |
| 16 | Tool use | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified with trained small models; nanochat calculator and sandbox CPU-verified |
| 17 | Reasoning | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | part A CPU-verified with trained models; part B simulated solver (labelled) |
| 18 | Agentic loop | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified with a simulated policy (labelled); real-model exercise not validated |
| 19 | Multi-agent | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified with a simulated solver (labelled) |
| 20 | Self-improvement | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified on real small training runs; nanochat seed wrapper CPU-verified |
| 21 | Full system | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified (end-to-end run, reproducibility check) |
| 22 | Beyond the baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified (retrieval lab); other topics conceptual, source-cited |

✅ done

## Other deliverables

| Deliverable | File | Status |
|---|---|---|
| Course map | [COURSE_MAP.md](COURSE_MAP.md) | Done (generated from the repository contents; all links checked) |
| Assessments: baseline diagnostic, five milestones with rubrics, end-of-part assessments, ability checklist | [instructor/assessments.md](instructor/assessments.md) | Done |
| Grading rubrics | [instructor/grading-rubrics.md](instructor/grading-rubrics.md) | Done |
| Teaching guide with a 12-week schedule | [instructor/teaching-guide.md](instructor/teaching-guide.md) | Done |
| Capstones A, B, C with checkers, fixtures and tests | [projects/](projects/README.md), `tests/test_capstones.py` | Done · CPU-verified |
| Reference library | [resources/](resources/README.md) | Done (arXiv identifiers and less common URLs checked against their titles on 2026-11-13) |
| Benchmark suite | `scripts/benchmarks/run_all.py`, [results/reference/benchmarks-cpu.md](results/reference/benchmarks-cpu.md) | Done · CPU-verified, three runs |
| Measured reference results | [results/reference/](results/reference/) | Done: Part I–II, benchmarks, Part IV, Parts V–VI |

## Verification performed (2026-11-13)

| Check | Command | Result |
|---|---|---|
| Unit tests, course environment (Python 3.13, torch 2.15 CPU) | `python -m pytest -m "not gpu and not nanochat"` | 124 passed |
| All tests incl. nanochat parity, pinned environment (Python 3.10, torch 2.9.1 CPU) | `python -m pytest -m "not gpu"` | 131 passed |
| Byte-compilation under Python 3.10 | `python -m compileall mlle labs scripts tests projects` | no errors |
| Documentation: every relative link and anchor, every backticked repository path, and all 206 upstream links and line ranges against the pinned checkout | `make check-docs` | no problems |
| Capstone checkers on their fixtures | `tests/test_capstones.py` | pass |
| GitHub Actions workflow | `.github/workflows/ci.yml` | defined; its result on the hosted runners is reported in the pull request, not here |

Not performed: anything requiring a GPU; rendering of the 27 Mermaid diagrams. The
diagrams passed a static check (diagram type, balanced brackets and quotes, matching
`subgraph`/`loop`/`alt` and `end`, no HTML other than line breaks) but were not rendered
by the author: view them on GitHub and report any that fail.

## Claims awaiting validation on a DGX Spark

None of these has been executed by the author. Each has a procedure in
[docs/dgx-spark-setup.md](docs/dgx-spark-setup.md#8-validation-checklist-fill-this-in-on-your-machine).

1. `make env-spark` produces a working CUDA environment (which wheel index is needed).
2. `torch.compile` works after setting `TRITON_PTXAS_PATH`.
3. nanochat selects the SDPA attention path on the GB10 (expected from source).
4. bf16 training throughput and peak memory at depths 6 and 12 (`scripts/experiments/spark_probe.sh`).
5. Whether FP8 (`--fp8`) is usable and beneficial on the GB10.
6. All GPU timing columns in Part III benchmark tables, and `scripts/infer_bench.py`.
7. SFT and RL wall-clock times at the recommended sizes (`chat_sft`, `chat_rl`).
8. Every exercise marked "awaiting validation on GPU" in the lab sheets (Modules 08–17, 20, 21).
9. Agent-loop and experiment-loop exercises with a real language model (Module 18 Exercise 5, Module 20 challenge, Module 21 challenge).

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-11 | Pin nanochat at `92d63d4` (HEAD of master on the audit date) | Most recent revision; README and code agree; upstream tests pass on CPU |
| 2026-11-11 | Course lives at the repository root; nanochat in `external/` (git-ignored) | Keeps upstream external and unmodified |
| 2026-11-11 | One shared virtual environment (nanochat's, plus the course package) | Same PyTorch for nanochat and labs; lockfile reproducibility |
| 2026-11-11 | Lab scripts with `# %%` cells instead of notebooks | Run in CI, diff cleanly, no stale outputs |
| 2026-11-11 | Synthetic language "Pocket" as the main Part I–II corpus | Known entropy and a controllable long-range dependency; no licensing or download |
| 2026-11-11 | Recommend `--window-pattern L` on the Spark | nanochat's own guidance when Flash Attention 3 is unavailable |
| 2026-11-11 | Report multi-seed results where single seeds mislead | The position ablation changed its conclusion between 1 and 3 seeds |
| 2026-11-12 | One shared small base model (2 layers × 96) for Parts IV–V, deliberately imperfect at arithmetic | Later stages need room to show an effect; labs stay within minutes on CPU |
| 2026-11-12 | Report unfavourable results as measured (voting does not beat greedy decoding; RL improves only practised prompts; plain DPO lowers accuracy) | The course's own rule: no tuning until a result looks impressive |
| 2026-11-12 | Simulated solver and policy for Modules 17B, 18, 19, labelled everywhere | Orchestration questions need controllable error properties; no real model larger than the course's is available on CPU |
| 2026-11-12 | LoRA with separate trainable rows for new tokens; base tables never modified | Switching the adapter off must restore the base model exactly |
| 2026-11-13 | Top-1 retrieval with abstention in the Module 22 lab | Associative look-up among several passages did not train within a lab-sized budget on the small model |
| 2026-11-13 | `scripts/experiments/seeded_base_train.py` wrapper instead of any change to nanochat | nanochat has no seed argument; upstream stays unmodified |

## Corrections made during development (kept as teaching material)

| Where | First version | After measuring |
|---|---|---|
| Module 04 | "RoPE learns faster" from one seed | not supported over three seeds |
| Module 06 | removing residual connections makes gradients vanish | gradients do not vanish in the pre-norm model; training is slow or unstable depending on learning rate |
| Module 11 | a larger model would show a larger KV-cache speed-up | about the same on the reference CPU |
| Module 15 | DPO with near-miss negatives expected to improve accuracy | accuracy fell (likelihood displacement); documented with remedies |
| Module 17 | a running-sum task to show the benefit of scratchpads | sums are parallelisable; replaced by mixed operations modulo 5 |
| Module 18 | ReAct prompt with truncated history only | state could scroll out of the prompt; explicit current state added, regression test |
| Module 22 | closed-book accuracy 0.92 | data-split artefact (unshuffled documents); fixed, kept as an exercise |

## How to maintain this work

1. `git fetch origin && git checkout <branch>`; `make env-cpu && source external/nanochat/.venv/bin/activate && make test-all && make test-nanochat && make check-docs`.
2. To move the pin: follow [docs/upstream-version.md](docs/upstream-version.md), change `PINNED_SHA` in `mlle/nanochat_bridge.py`, re-run the parity tests and `make check-docs`; every failing upstream link marks a passage to re-read.
3. To add GPU results: run the procedure in [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md), fill in [results/templates/spark-validation.md](results/templates/spark-validation.md), and add a file under `results/reference/` with hardware, date and revision. Replace "awaiting validation" notes only with measured values.
4. New or changed content follows [CONTRIBUTING.md](CONTRIBUTING.md): sections A–M, every lab executed, every number labelled.

**Continuation instruction for an AI assistant:** "The course in `lbecken/llm-course` is
complete on CPU. Read `PROGRESS.md`, `CONTRIBUTING.md` and `docs/upstream-version.md`.
Do not rewrite completed modules. Useful next work, in order: (1) validate the items
under 'Claims awaiting validation on a DGX Spark' on real hardware and record measured
results; (2) re-audit against a newer nanochat revision; (3) run the real-model
exercises of Part V. Never present an estimate or a simulated value as a measurement,
and never modify the upstream nanochat repository."
