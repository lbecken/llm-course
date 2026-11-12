# Progress manifest

Single source of truth for what exists, what has been verified, and what remains.
Update it in the same commit as the work it describes.

| | |
|---|---|
| Upstream pin | nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (2026-07-03), audited 2026-11-11 |
| Working branch | `claude/cool-cannon-10dksz` |
| Verification commands | `make test` · `make test-all` · `make test-nanochat` · `make check-docs` |
| Hardware available to the author | 4-vCPU x86-64 sandbox, no GPU. **No DGX Spark.** |

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

Columns: L = lesson, B = lab, Q = quiz, R = references, S = solutions, C = code + tests.

| # | Module | L | B | Q | R | S | C | Validation |
|---:|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 01 | Neural network refresher | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-verified |
| 02 | Language modeling | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code CPU-verified |
| 03 | Tokenization | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code CPU-verified |
| 04 | Embeddings and position | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code CPU-verified |
| 05 | Attention | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code CPU-verified |
| 06 | Transformer block | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code + nanochat parity CPU-verified |
| 07 | Optimization | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | code CPU-verified |
| 08 | Data and pretraining | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ◐ | nanochat local run CPU-verified |
| 09 | Scaling and compute | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ◐ | estimator CPU-verified |
| 10 | Evaluation and generation | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 11 | KV cache | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ◐ | cache code CPU-verified |
| 12 | Inference optimization | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 13 | Efficient training | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 14 | Supervised fine-tuning | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 15 | RL and preferences | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 16 | Tool use | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 17 | Reasoning | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 18 | Agentic loop | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 19 | Multi-agent | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 20 | Self-improvement | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | |
| 21 | Full system | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | |
| 22 | Beyond the baseline | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | |

✅ done · ◐ partly · ⬜ pending

## Other deliverables

| Deliverable | Status |
|---|---|
| Capstones A, B, C (`projects/`) | Pending |
| Reference library (`resources/`) | Pending |
| Instructor material (`instructor/`) | Pending |
| Benchmark suite (`scripts/benchmarks/`, `mlle/bench/`) | Pending |

## Claims awaiting validation on a DGX Spark

None of these has been executed by the author. Each has a procedure in
[docs/dgx-spark-setup.md](docs/dgx-spark-setup.md#8-validation-checklist-fill-this-in-on-your-machine).

1. `make env-spark` produces a working CUDA environment (which wheel index is needed).
2. `torch.compile` works after setting `TRITON_PTXAS_PATH`.
3. nanochat selects the SDPA attention path on the GB10 (expected from source).
4. bf16 training throughput and peak memory at depths 6 and 12 (`scripts/experiments/spark_probe.sh`).
5. Whether FP8 (`--fp8`) is usable and beneficial on the GB10.
6. All GPU timing columns in Part III benchmark tables.
7. SFT and RL wall-clock times at the recommended sizes.

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

## How to resume this work

1. `git fetch origin claude/cool-cannon-10dksz && git checkout claude/cool-cannon-10dksz`
2. `make env-cpu && source external/nanochat/.venv/bin/activate && make test && make test-nanochat`
3. `make check-docs`: every "broken link" it prints is a file still to be written.
4. Take the first module in the table above that is not all ✅. Follow the template of
   `modules/01-neural-network-refresher/` (sections A–M; see [CONTRIBUTING.md](CONTRIBUTING.md)).
   Run every lab script and paste real output; put larger measurements in `results/reference/`.
5. Update the table and commit.

**Continuation instruction for an AI assistant:** "Continue the course in
`lbecken/llm-course` on branch `claude/cool-cannon-10dksz`. Read `PROGRESS.md`,
`SYLLABUS.md`, `CONTRIBUTING.md` and Module 01 as the quality reference. Do not rewrite
completed files. Implement the next pending module in full (lesson, lab, quiz,
references, solutions, code, tests), run its labs and tests, record measured results,
update `PROGRESS.md`, commit and push. Keep nanochat pinned at `92d63d4`."
