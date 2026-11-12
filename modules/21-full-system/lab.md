# Module 21 — Lab: the whole chain in one run

Script: [`labs/integration/21_end_to_end.py`](../../labs/integration/21_end_to_end.py) ·
Solutions: [`solutions/21-full-system.md`](../../solutions/21-full-system.md)

The lab reuses the cached base model of Part IV and takes under a minute once it exists
(about two minutes on the first run). Its output is a JSON manifest in
`results/runs/part6/`.

## G. Exercises

### Exercise 1 (by hand): interfaces

1. For each pair, name the artefact that crosses the boundary and one property of it that the consumer silently relies on: (a) tokenizer training → pretraining; (b) pretraining → fine-tuning; (c) fine-tuning → the application's prompt renderer; (d) engine → tool; (e) agent loop → experiment loop.
2. A team retrains its tokenizer on more data (same vocabulary size) and keeps using the old model checkpoint. What happens, and will any component raise an error?
3. A reference run takes 1.5 hours on 8 GPUs with one accumulation step per optimizer step. You have one GPU of the same type. Give the number of accumulation steps and an estimate of wall-clock time for the pretraining stage. Which stages of the pipeline are *not* slowed by the same factor?

**Success criterion:** written answers; item 3 with the arithmetic shown.

### Exercise 2 (experimental): run it, twice

```bash
python labs/integration/21_end_to_end.py --out results/runs/part6/run_a.json
python labs/integration/21_end_to_end.py --out results/runs/part6/run_b.json
```

1. Answer the lesson's "predict before you run" question. Then diff the two manifests (ten lines of Python, or `diff`). Which fields differ?
2. Run with `--seed 1`. Which metrics change, and by how much? Classify each metric of the manifest as "reproducible exactly", "replicable within noise" or "timing".
3. Which stage dominates the run? Which would dominate if the base model were not cached? Which dominates in nanochat's reference run, and why is the ranking different?
4. Run with `--sft-steps 100`. Which downstream metrics change? Trace each change to the stage that caused it.

**Success criterion:** a table classifying every metric, and a written answer to item 4.

### Exercise 3 (coding): break an interface on purpose

In a scratch copy of the lab, introduce each fault, run, and record which stage's metrics
reveal it and which do not.

1. **Template mismatch:** at serving time, render the question without `<|user_end|>`.
2. **Tokenizer mismatch:** build a second tokenizer with the alphabet in a different order and use it for serving only.
3. **Tool mismatch:** serve with a calculator that formats results with a decimal point (`5096.0`).
4. **Evaluation leak:** evaluate the assistant on the questions it was fine-tuned on instead of held-out ones.

For each: does anything crash? Which single check would have caught it before the full
pipeline ran?

**Success criterion:** a four-row table: fault, symptom, crash (yes/no), cheapest detecting check.

### Exercise 4 (design): observability

1. Extend the manifest with: the course commit (`git rev-parse HEAD`), the pinned nanochat SHA (`mlle.nanochat_bridge.PINNED_SHA`), and, for the serving stage, three complete traces (rendered prompt tokens, generated tokens with their sampled/forced mask, stop reason).
2. Add a `--compare OLD.json` option that prints, per stage, which metrics changed by more than a tolerance you pass, and exits non-zero if any did. Explain how this turns the manifest into a regression test.
3. Write the one-page "run card" you would want from a colleague before trusting their result: which ten fields, in which order?

**Success criterion:** working code for items 1–2 and the run card.

### Exercise 5 (nanochat): read your own pipeline

With any nanochat run of your own (the miniature CPU run of Module 08 suffices):

```bash
ls $NANOCHAT_BASE_DIR $NANOCHAT_BASE_DIR/base_checkpoints/*
python -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(m, indent=2)[:1500])" $NANOCHAT_BASE_DIR/base_checkpoints/<tag>/meta_<step>.json
```

1. List the artefacts on disk and assign each to a stage of lesson D.1.
2. From the metadata file alone, reconstruct the command line that produced the checkpoint. What can you *not* recover from it?
3. Write the pipeline of `runs/speedrun.sh` for your hardware as a table: stage, command with your flags, measured or estimated duration (state which), artefact produced. Use `scripts/experiments/estimate_run.py` with a measured throughput for the estimates.
4. Which of this module's observability items does nanochat provide out of the box, and which would you add first for a week-long run?

**Success criterion:** the table of item 3 with every duration labelled measured or estimated. *The author's GPU durations are not available; none are given.*

### Challenge (optional): the nanochat model in the course's loops

Write an adapter class with a `complete(prompt, temperature, max_tokens, seed)` method
that wraps nanochat's `Engine` and a loaded checkpoint, so that the tool assistant of
Module 16, the strategies of Module 17 and the agent loop of Module 18 can run on your
own model. Report what works with a small model and what fails, by outcome label.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Run every stage in order with a manifest; establish what is reproducible; find the bottleneck; observe how faults at interfaces surface |
| **Hypotheses** | (H1) Two runs with the same seed agree on every non-timing metric. (H2) A different seed changes trained-model metrics within the noise measured in earlier modules and leaves structural metrics (counts, hashes for the same data seed) unchanged. (H3) With the base model cached, fine-tuning dominates the run time. (H4) Interface faults produce degraded metrics without a crash. |
| **Hardware / software** | CPU; course package |
| **Tier** | A: under 1 minute (2 minutes on first use) |
| **Metrics** | per stage: seconds, status, and the stage's own metrics (loss, accuracy, tool-call rate, tokens/s, outcomes) |
| **Independent variables** | seed; fine-tuning steps; injected interface faults |
| **Controls** | one code revision; the same cached base model; held-out evaluation questions |
| **Interpretation** | Timings vary by 10–30 % between runs on a shared CPU. The miniature's stage proportions are not those of a full-scale system. |
| **Common failure modes** | Comparing manifests from different code revisions without noting it. Treating a run that did not crash as a run that worked. Recording aggregate metrics without any trace. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.2.

| Stage | seconds | key metric | identical across same-seed runs? |
|---|---:|---|---|
| pretrain (cached) | | | |
| eval-base | | | |
| fine-tune | | | |
| serve-eval | | | |
| agent | | | |

| Injected fault | symptom | crash? | cheapest check |
|---|---|---|---|
| template mismatch | | | |
| tokenizer mismatch | | | |
| tool format mismatch | | | |
| evaluation leak | | | |

**Questions to answer in writing**

1. Which two artefacts must be versioned together with a model checkpoint for it to be usable a year later, and why?
2. Your pipeline's final accuracy dropped by three points after "a small refactoring". Describe your first three steps.
3. What would you measure to decide whether to spend the next week on pretraining speed, on inference speed or on evaluation?
