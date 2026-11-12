# Syllabus

**Modern LLM Engineering: From Transformer Foundations to Reasoning and Agentic Systems**

Twenty-two modules in six parts, five milestones, three capstone options. Study-time
figures are estimates for a learner matching the [prerequisites](docs/prerequisites.md),
and include reading, labs and the quiz; "machine time" is unattended waiting and depends
on your hardware (measure it, see [hardware and costs](docs/hardware-and-costs.md)).

| Part | Modules | Study time | Machine time (DGX Spark, to be measured) |
|---|---|---:|---|
| I — Rebuilding the Foundations | 01–07 | 40–45 h | minutes (CPU is enough) |
| II — Training a Modern LLM | 08–10 | 18–22 h | 1 h to overnight |
| III — Performance Engineering | 11–13 | 19–23 h | 1–3 h |
| IV — From Base Model to Assistant | 14–15 | 14–17 h | 1 h to overnight |
| V — Reasoning, Tools and Agentic Systems | 16–20 | 22–26 h | minutes (CPU is enough) |
| VI — Integration, Limits and Frontier | 21–22 | 8–10 h | — |
| Capstone (one of three) | — | 15–25 h | varies |
| **Total, full engineer track** | | **≈ 135–165 h** | |

Shorter routes (35 h and 80 h) are in [LEARNING_PATHS.md](LEARNING_PATHS.md).

Every module has the same four files and a solutions file:
`lesson.md` (objectives → intuition → mathematics → minimal implementation → nanochat
walkthrough → takeaways), `lab.md` (exercises, experimental procedure, results template),
`quiz.md` (knowledge check and mastery criteria), `references.md` (annotated further
study), and `solutions/NN-*.md`.

---

## Part I — Rebuilding the Foundations

*Goal: understand the complete forward and backward computation of a small GPT-style Transformer.*

| # | Module | You will be able to | Key lab | Tier | Hours |
|---:|---|---|---|---|---:|
| 01 | [From the 1990s perceptron to the language model](modules/01-neural-network-refresher/lesson.md) | derive and verify backprop by hand; explain next-token prediction as classification | hand-written MLP on digits, then on text | A | 4–5 |
| 02 | [Language modeling and autoregressive generation](modules/02-language-modeling/lesson.md) | factor sequence probability; relate loss, entropy and perplexity; explain teacher forcing and the train/generate asymmetry | n-gram models vs. the exact entropy of a synthetic language | A | 4–5 |
| 03 | [Tokenization](modules/03-tokenization/lesson.md) | implement BPE; explain special tokens and chat formatting; evaluate a tokenizer | train BPE by hand and with nanochat's `rustbpe` | A | 4–5 |
| 04 | [Embeddings and position](modules/04-embeddings/lesson.md) | explain embeddings as learned lookup tables; derive RoPE's relative-position property | position ablation with three seeds | A | 4–5 |
| 05 | [Attention from first principles](modules/05-attention/lesson.md) | compute attention by hand; implement it three ways; reason about masks, heads, cost and what weights do *not* explain | one head learns a dictionary; equivalence with PyTorch SDPA | A | 8–9 |
| 06 | [The complete Transformer block](modules/06-transformer-block/lesson.md) | trace every tensor from ids to probabilities; list all differences between GPT-2 and nanochat@`92d63d4` | reproduce nanochat's logits with MiniGPT | A | 8–9 |
| 07 | [Learning and optimization](modules/07-optimization/lesson.md) | follow gradients through attention; implement AdamW and Muon; reason about schedules, initialisation, stability | train MiniGPT to the entropy floor | A | 7–8 |

**Milestone 1 (Part I):** build, explain and train a minimal causal Transformer on a small
corpus, generate text, and map every component to nanochat.
[Checkpoint description and rubric](instructor/assessments.md#milestone-1).

## Part II — Training a Modern LLM

*Goal: understand the complete pretraining pipeline and how architecture, data and compute choices affect quality.*

| # | Module | You will be able to | Key lab | Tier | Hours |
|---:|---|---|---|---|---:|
| 08 | [Data preparation and pretraining](modules/08-data-and-pretraining/lesson.md) | explain data quality, splits, contamination, sequence packing; read a training log and diagnose failures | unmodified nanochat on a local dataset; best-fit packing statistics | A–C | 6–7 |
| 09 | [Model size, scaling laws and compute budgets](modules/09-scaling-and-compute/lesson.md) | count parameters and FLOPs; use scaling laws to plan a run; explain nanochat's depth dial | parameter/FLOP accounting; a small controlled scaling sweep | A–C | 6–7 |
| 10 | [Evaluation and generation](modules/10-evaluation-and-generation/lesson.md) | compute perplexity and bits per byte; explain CORE; implement temperature, top-k, top-p; design a trustworthy comparison | decoding strategies; seed variance and confidence intervals | A–B | 6–8 |

**Milestone 2 (Part II):** train and evaluate at least one small nanochat model, log
configuration and results, and investigate one variable with a controlled experiment.
[Details](instructor/assessments.md#milestone-2).

## Part III — Performance Engineering

*Goal: understand why modern LLM systems are fast, separating mathematical improvements, hardware-specific kernels and system-level techniques.*

| # | Module | You will be able to | Key lab | Tier | Hours |
|---:|---|---|---|---|---:|
| 11 | [Efficient autoregressive inference: the KV cache](modules/11-kv-cache/lesson.md) | derive the KV cache; size it; verify cached = uncached; explain prefill vs. decode | cache correctness test and speed/memory benchmark | A–B | 5–6 |
| 12 | [Attention and inference optimization](modules/12-inference-optimization/lesson.md) | explain FlashAttention's tiling; GQA; sliding windows; batching; quantization; bandwidth limits | tiled attention with identical output; int8 quantization; `infer_bench.py` | A–B | 7–8 |
| 13 | [Efficient training](modules/13-efficient-training/lesson.md) | choose precision; use gradient accumulation; account for training memory; interpret MFU; explain `torch.compile` and distributed training | precision and compile benchmarks | A–C | 7–9 |

**Milestone 3 (Part III):** a reproducible benchmark suite comparing baseline and optimized
approaches on latency, tokens/s, memory, correctness and quality, from measurements.
[Details](instructor/assessments.md#milestone-3).

## Part IV — From Base Model to Assistant

*Goal: understand how a next-token predictor becomes a conversational assistant.*

| # | Module | You will be able to | Key lab | Tier | Hours |
|---:|---|---|---|---|---:|
| 14 | [Instruction tuning and supervised fine-tuning](modules/14-supervised-finetuning/lesson.md) | render conversations to tokens with loss masks; run SFT; measure forgetting; implement LoRA | SFT of a tiny model with masked loss; LoRA vs. full fine-tuning | A–C | 7–8 |
| 15 | [Preference optimization and reinforcement learning](modules/15-rl-and-preferences/lesson.md) | derive the policy gradient; explain baselines, PPO, DPO, GRPO; read nanochat's RL loop; recognise reward hacking | REINFORCE with a verifiable reward; a DPO update | A–C | 7–9 |

**Milestone 4 (Part IV):** an appropriately scaled SFT run and a controlled experiment
showing reward-driven improvement, distinguishing a trained policy update from prompting.
[Details](instructor/assessments.md#milestone-4).

## Part V — Reasoning, Tools and Agentic Systems

*Goal: understand how language models take part in multi-step computation, and separate learned capabilities from application-level orchestration.*

| # | Module | You will be able to | Project | Tier | Hours |
|---:|---|---|---|---|---:|
| 16 | [Teaching LLMs to use tools](modules/16-tool-use/lesson.md) | explain tool-call data, parsing, execution, observation; sandboxing and prompt injection | a restricted tool-using assistant | A | 5–6 |
| 17 | [Reasoning and inference-time computation](modules/17-reasoning/lesson.md) | distinguish reasoning text, search, RL for reasoning, and weights; quantify self-consistency and verification | direct vs. decomposition vs. vote vs. verify on a benchmark | A | 5–6 |
| 18 | [The agentic loop](modules/18-agentic-loop/lesson.md) | build a ReAct-style loop with state, budgets, stop conditions and recovery | an instrumented, bounded agent | A | 5–6 |
| 19 | [Multi-agent collaboration](modules/19-multi-agent/lesson.md) | design planner/executor/reviewer protocols; measure against a single-agent baseline | single agent vs. team on a fixed suite | A | 4–5 |
| 20 | [Self-improvement and autonomous experimentation](modules/20-self-improvement/lesson.md) | separate improving an answer, a workflow and weights; build a bounded experiment loop; spot leakage | propose–test–evaluate–record loop | A | 5–6 |

**Milestone 5 (Part V):** the five projects above, each with tests and a measured comparison.
[Details](instructor/assessments.md#milestone-5).

## Part VI — Integration, Limitations and the Research Frontier

| # | Module | You will be able to | Tier | Hours |
|---:|---|---|---|---:|
| 21 | [Putting the full system together](modules/21-full-system/lesson.md) | describe the end-to-end system, its interfaces, bottlenecks, reproducibility and observability | A | 4–5 |
| 22 | [Beyond the baseline Transformer](modules/22-beyond-the-baseline/lesson.md) | evaluate new developments (RAG, MoE, alternative attention, SSMs, long context, multimodal, distillation) with evidence | A | 4–5 |

## Capstones (choose one; details in [`projects/`](projects/))

| | Capstone | Core deliverable |
|---|---|---|
| A | [Explain and train your own small GPT](projects/capstone-a-small-gpt/README.md) | a trained, inspected, evaluated and documented model with a written architectural explanation |
| B | [Optimize a real inference pipeline](projects/capstone-b-inference/README.md) | one optimization, verified for correctness, benchmarked against a baseline, with hardware analysis |
| C | [From LLM to tool-using agent](projects/capstone-c-agent/README.md) | a bounded tool-using agent with measurable evaluation; optional multi-agent extension compared with the single-agent baseline |

## Assessment

Mastery-based: a module is complete when its mastery criteria (end of each `quiz.md`) are
met, not when its commands have been run. The baseline diagnostic, end-of-part
assessments, milestone rubrics and the final checklist of abilities are in
[instructor/](instructor/).

## Design decisions (brief)

- **One synthetic language with a known entropy** ("Pocket") runs through Parts I–II, so that a loss value can be compared with the true optimum and a specific long-range dependency can be tested. Real text is used wherever nanochat is run at scale.
- **A small course package (`mlle`) next to nanochat, not instead of it.** Each mechanism is first built in its plainest form, then located in nanochat. A parity test ties the two together numerically.
- **Scripts, not notebooks**, with `# %%` cells: they run in CI and cannot carry stale outputs.
- **Part V runs against a model interface with three back-ends**: deterministic simulated models (for controlled experiments and tests), tiny trained models (for real learned behaviour on CPU), and adapters for nanochat or a locally served model (for realism). Systems claims are measured where they can be controlled.
- **Measured, reported and expected numbers are labelled differently** everywhere, and GPU-dependent statements are marked until validated on the target hardware.
