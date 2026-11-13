# Module 22 — Lab: stale weights, fresh context, and reading a negative result

Script: [`labs/integration/22_retrieval.py`](../../labs/integration/22_retrieval.py) ·
Solutions: [`solutions/22-beyond-the-baseline.md`](../../solutions/22-beyond-the-baseline.md)

The lab trains three very small models from scratch (about 1.5 minutes on a 4-core CPU).

## G. Exercises

### Exercise 1 (by hand): the seven questions

Apply the seven questions of lesson D.1 to each of the following, in three or four lines
each, using only what the lesson and earlier modules establish:

1. Retrieval-augmented generation for an assistant that answers from a company's internal documents.
2. Grouped-query attention (Module 12).
3. The mixture-of-experts experiment recorded in nanochat's `dev/LOG.md`.

Then: a paper claims "our architecture matches the baseline with 30 % fewer FLOPs".
Name three comparisons you would ask for before concluding that it is faster.

**Success criterion:** each answer names the limitation, the unit of comparison and one new failure mode.

### Exercise 2 (experimental): weights versus context

```bash
python labs/integration/22_retrieval.py
```

1. Answer the lesson's "predict before you run" question. Record the table.
2. Remove the line `r.shuffle(closed_docs)` and re-run. The closed-book model now reaches only 0.92 on the table it was trained on. Which keys does it get wrong? (Print them.) Explain, and say why this bug would be hard to notice in a larger system.
3. In the row "open-book / fact about another key", `right` is 0.02, not 0.00. Why is it not zero?
4. The "checking" model abstains on irrelevant passages but repeats a wrong value for the right key. Could any training make it detect the wrong value from the passage alone? What additional information would the *system* need?
5. Update the world again (a third table) and re-evaluate without retraining anything. Which rows change? What does that demonstrate about where the system's knowledge now lives?

**Success criterion:** recorded table and written answers to items 3–5.

### Exercise 3 (coding): a retriever that can fail

Replace the exact-match retriever by an imperfect one and measure the system:

1. `retrieve(key)` returns the right fact with probability $p$ and a random other fact otherwise. Measure end-to-end accuracy and abstention for $p \in \{1.0, 0.9, 0.7, 0.5\}$ for both open-book models. State the relationship between retrieval accuracy and answer accuracy for each.
2. For the checking model, define a system that *retries* retrieval when the model abstains (up to 3 times). Measure accuracy and the number of retrieval calls. Which module's pattern is this?
3. Insert one poisoned fact into the table: its "value" is a legal value, but it was planted by an attacker for a key the user will ask about. Which of your measurements could detect this? Which control from Module 16 applies?

**Success criterion:** a table for item 1 and working code for item 2.

### Exercise 4 (reading): a negative result as evidence

Read the entry "2026-02-19: Mixture of Experts (negative)" in `external/nanochat/dev/LOG.md`.

1. List every quantity it reports and the unit of comparison for each (per step, per FLOP, per second).
2. Which engineering interactions with the rest of the pipeline does it mention? Map each to a module of this course.
3. Write the two-sentence summary you would give a colleague who asks "so, is MoE bad?".
4. The entry ends with a statement of what would have to exist for the result to change. What is it, and what does that tell you about when to revisit a negative result?
5. Find one *positive* claim elsewhere (a paper, a blog post, a model card) and list which of the seven questions it leaves unanswered.

**Success criterion:** written answers; item 3 must state the scope of the result.

### Exercise 5 (design): your own extension

Choose one idea from this module (or from your reading) and design the smallest
experiment that could tell you whether it helps *on your hardware*:

1. Hypothesis, with the metric and the size of effect you would consider meaningful.
2. Baseline and the unit of equality (steps, tokens, FLOPs, wall-clock time).
3. Model size and budget, sized with `scripts/experiments/estimate_run.py` and a measured throughput, or on MiniGPT for a CPU-scale version.
4. Number of seeds and the acceptance rule (Module 20).
5. What you would record if the result is negative.

Do not run it as part of this exercise; the design is the deliverable. It can become your
capstone.

**Success criterion:** a one-page experiment design that another person could execute.

### Challenge (optional): distillation in miniature

Train a 4-layer MiniGPT "teacher" on Pocket. Train two 1-layer "students" for the same
number of steps: one on the text, one to match the teacher's next-token distribution
(KL divergence to the teacher's softmax) on the same text. Compare validation loss and
agreement accuracy over three seeds. Does the student trained on the teacher's
distribution learn faster? Relate the result to the amount of information per training
example in a distribution versus a single token.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Measure what retrieval fixes and what it introduces, using models small enough to train in seconds; practise structured reading of evidence |
| **Hypotheses** | (H1) After the table changes, the closed-book model's accuracy on the new table is near 0 and it mostly returns old values. (H2) The open-book model is as accurate on the changed table as on the old one, without retraining. (H3) Given an irrelevant passage, the plain open-book model repeats its value in more than 90 % of cases; the checking model abstains in more than 90 %. (H4) Given a wrong value for the right key, both repeat it. |
| **Hardware / software** | CPU; course package |
| **Tier** | A: about 1.5 minutes |
| **Metrics** | fraction right; fraction abstaining; fraction repeating the value shown in the context (for the closed-book model: the stale value) |
| **Independent variables** | where the knowledge is (weights or context); table version; retriever behaviour (correct, irrelevant, wrong value); training with or without irrelevant passages |
| **Controls** | identical architecture, steps and seed for the three models; the same 208 questions per row |
| **Interpretation** | 26 facts and top-1 exact-match retrieval: a caricature of real retrieval, chosen so that each failure mode appears in isolation. One seed. The magnitudes are properties of the toy; the pattern (stale / follows context / cannot verify) is the finding. |
| **Common failure modes** | Reporting only end-to-end accuracy. Evaluating retrieval only where it succeeds. Concluding that a model which abstains on irrelevant passages can also detect false ones. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.2.

| Model | Condition | right | abstains | repeats context | predicted |
|---|---|---:|---:|---:|---|
| closed-book | as trained | | | | |
| closed-book | changed table | | | | |
| open-book | correct retrieval, changed table | | | | |
| open-book | fact about another key | | | | |
| open-book, checking | fact about another key | | | | |
| open-book, checking | wrong value | | | | |

**Questions to answer in writing**

1. A product team says "we added retrieval, so the model no longer hallucinates". Which two rows of your table do you show them?
2. In one paragraph: what is the same about a lying calculator (Module 16), a poisoned document and a misleading passage?
3. Of everything in this course, which three ideas do you expect to be still true in ten years, and which three specifics do you expect to be obsolete?
