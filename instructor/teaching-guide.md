# Teaching guide

For someone leading a group through the course, and for a self-learner who wants a
schedule and a method.

## The method in one page

Every module follows the same sequence: intuition, mathematics, a small numerical
example, a minimal implementation, the nanochat implementation, an experiment, its
interpretation, a knowledge check. Three habits make the sequence work, and they are
what a teacher should protect when time is short.

1. **Predict before you run.** Every lesson has at least one "predict before you run"
   box, and every results table has a prediction column. Have learners write the
   prediction down. The gap between prediction and result is where learning happens; a
   result merely observed is forgotten in a week.
2. **One variable, with its noise.** From Module 04 onward the material repeatedly shows
   single-seed results being overturned by a second seed. Do not let a comparison stand
   without the question "how much does this vary if nothing changes?".
3. **Label every number**: measured (by whom, on what, when), reported (by which
   source), or estimated (from what). The course material does this throughout,
   including statements that something was *not* measured. Hold learners' write-ups to
   the same standard ([grading-rubrics.md](grading-rubrics.md), Rubric 3).

Two things the course deliberately does that may surprise a teacher:

- **It reports its own mistakes.** Several lessons describe a claim the author first
  made and then had to correct after measuring (the role of residual connections in
  Module 06, a position-encoding conclusion in Module 04, a context bug in Module 18, a
  data-split artefact in Module 22). These are among the most valuable passages. Use
  them: ask learners what would have happened had the check not been run.
- **Many results are negative or modest.** Voting does not beat greedy decoding;
  reinforcement learning improves only the prompts it practises; DPO damages accuracy
  unless anchored. Resist the temptation to tune until the result looks impressive.

## Before the first session

- Learners take the [baseline diagnostic](assessments.md#baseline-diagnostic) and choose a track in [LEARNING_PATHS.md](../LEARNING_PATHS.md).
- Everyone runs `make install`, `make test` and `make doctor` ([getting started](../docs/getting-started.md)). A CPU is sufficient for everything except the GPU exercises, which are marked.
- Decide the hardware plan. With no GPU, Parts I, III (labs), IV, V and VI run in full on CPU; Part II uses the miniature nanochat pipeline. With one DGX Spark for a group, schedule the long runs (Milestone 2, the optional SFT run) overnight and share the checkpoints.
- Pin the revision: the course is verified against one nanochat commit ([docs/upstream-version.md](../docs/upstream-version.md)). Do not update it mid-course.

## A 12-week schedule

About 10–12 hours per week: one 90-minute session plus individual work. Hours are the
syllabus estimates. "Session focus" is what to do together; everything else is
individual.

| Week | Modules | Session focus | Due at the end of the week |
|---:|---|---|---|
| 1 | 01 Neural-network refresher · 02 Language modeling | by-hand backpropagation on the board; entropy and perplexity with the n-gram lab | diagnostic; labs 01–02 |
| 2 | 03 Tokenization · 04 Embeddings | train a BPE by hand on ten words; the single-seed trap of Module 04 | labs 03–04 |
| 3 | 05 Attention | attention by hand for a 3-token example; interventions versus attention pictures | lab 05 |
| 4 | 06 Transformer block · 07 Optimization | read `gpt.py` together, line by line; the parity lab | **Milestone 1** |
| 5 | 08 Data and pretraining · 09 Scaling | size a run on the board from measured throughput; start Milestone 2 runs | sizing table; runs started |
| 6 | 10 Evaluation and generation | bootstrap intervals and paired comparison on learners' own results | **Milestone 2** |
| 7 | 11 KV cache · 12 Inference optimization | break the cache three ways; exact versus approximate optimisations | labs 11–12 |
| 8 | 13 Efficient training | precision formats; memory accounting; review of `make bench` outputs | **Milestone 3** |
| 9 | 14 Supervised fine-tuning · 15 RL and preferences | the five-row table and its controls; the flawed-reward run | **Milestone 4** |
| 10 | 16 Tool use · 17 Reasoning | threat-model a tool set together; equal-cost comparison of strategies | labs 16–17 |
| 11 | 18 Agentic loop · 19 Multi-agent · 20 Self-improvement | outcome distributions; "is the team a workflow?"; searching in noise | **Milestone 5** |
| 12 | 21 Full system · 22 Beyond the baseline · capstone | capstone design reviews; the seven questions applied to a paper the group chooses | capstone proposal (capstone due two to four weeks later) |

Compressed variants: an **8-week** version drops weeks 7–8 to labs only (no Milestone 3)
and merges weeks 10–11 into one, keeping Module 18 in full. A **4-week intensive** is
Track 1 of [LEARNING_PATHS.md](../LEARNING_PATHS.md) plus Modules 14 and 16.

## Session patterns that work

- **Prediction poll.** Before running a lab's main command, collect predictions (a number each). Run it. Discuss the spread of predictions before the result.
- **Code reading aloud.** One learner explains ten lines of nanochat while another checks each claim against the course's minimal implementation. The parity test settles disagreements.
- **Find the flaw.** The end-of-part assessments contain flawed claims. Add your own from recent papers or product announcements.
- **Trace review.** In Part V, have each learner bring one failing trace and present what the outcome label and the steps reveal.
- **Results clinic.** Learners bring a results table; the group applies Rubric 3 row by row.

## Common difficulties, by part

| Part | What learners find hard | What helps |
|---|---|---|
| I | tensor shapes in attention; why the mask makes training parallel | the by-hand 3-token example; printing shapes at every line of `labs/foundations/06_transformer_trace.py` |
| I | believing that an attention picture explains behaviour | the intervention exercise (`blocked_edges`) |
| II | separating parameters, tokens, steps, batch size and their scaling rules | `scripts/experiments/estimate_run.py` and a table filled in by hand first |
| II | treating upstream 8-GPU timings as predictions | insist on a measured probe before any estimate |
| III | "faster" without a correctness check | the benchmark suite's correctness column; breaking the cache on purpose |
| IV | attributing everything good to fine-tuning | the two control rows of lab 14a |
| IV | expecting RL to be dramatic | compare runs A and C of lab 15a and discuss information per example |
| V | attributing intelligence to the loop or to roles | "which line calls the model?"; the single-agent-same-workflow exercise |
| V | taking simulated numbers for facts about models | the labels in the labs; Exercise 5 of Module 18 with a real model |
| VI | wanting a survey of the frontier | the seven questions, applied by the learner to one item of their choice |

## Assessment in a taught course

- Formative: quizzes (self-marked with the solutions), prediction columns, lab write-ups.
- Summative: the five milestones and one capstone, graded with the rubrics. A suggested weighting: milestones 10 % each, capstone 40 %, end-of-part assessments 10 %.
- Oral check for milestones (10 minutes): the learner explains one result and answers "what would change this number?". This is the most reliable way to distinguish understanding from execution.

## What a teacher must not do

- Do not supply GPU results the course does not contain as if they were reference values. If you run the GPU exercises, publish your numbers with hardware, date and revision, and label them as yours.
- Do not let learners modify the upstream nanochat repository or open pull requests against it as course work. Experiments belong in forks or in this repository.
- Do not relax the limits of Part V (budgets, permissions, no execution of model output) to make a demo more impressive.

## Maintaining the course

When nanochat changes: re-run the audit described in
[docs/upstream-version.md](../docs/upstream-version.md), update the pinned SHA in
`mlle/nanochat_bridge.py`, and run `make check-docs` with the new checkout. The checker
verifies every upstream link and line range against the source; each failure marks a
passage to re-read. [PROGRESS.md](../PROGRESS.md) records what has been verified and on
what hardware, and [CONTRIBUTING.md](../CONTRIBUTING.md) the rules for changes.
