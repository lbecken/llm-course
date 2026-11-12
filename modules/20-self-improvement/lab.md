# Module 20 — Lab: a loop that runs experiments, and the limits that keep it honest

Script: [`labs/agents/20_experiment_loop.py`](../../labs/agents/20_experiment_loop.py) ·
Code under study: [`mlle/agents/experiment.py`](../../mlle/agents/experiment.py) ·
Solutions: [`solutions/20-self-improvement.md`](../../solutions/20-self-improvement.md)

The lab performs about 70 real training runs of a very small model (about 90 seconds on a
4-core CPU) and writes its ledger to `results/runs/part5/20_ledger.jsonl`.

## G. Exercises

### Exercise 1 (by hand): selection arithmetic

1. Twenty configurations are, in truth, equally good; their measured validation scores are independent draws from the same distribution. If the first is the incumbent and each later one is accepted when it beats the best so far, what is the expected number of acceptances? (The $i$-th draw is a new record with probability $1/i$.)
2. The measured scores have standard deviation 0.006. Roughly how much better than the true mean is the best of 20 draws? Of 1 000? (The expected maximum of $n$ standard normal draws is about 1.87 for $n = 20$ and 3.24 for $n = 1000$.)
3. A loop accepts a change if it beats the incumbent by more than a threshold of 0.02, with the same standard deviation of 0.006 for a single run. Both the incumbent's score and the candidate's are noisy. What is the standard deviation of their difference, and how many standard deviations is the threshold?
4. A team reports: "our agent ran 500 experiments and improved validation loss by 0.4 %". Name the three numbers you need in order to judge the claim.

**Success criterion:** numerical answers for 1–3; item 1 gives about 2.6.

### Exercise 2 (experimental): read a ledger

```bash
python labs/agents/20_experiment_loop.py
```

1. Answer the lesson's "predict before you run" question. Record the noise, the ledger of part 2 and the table of part 3.
2. From the ledger alone, state which of the four parameters mattered most at this training budget, and give the evidence (which trials).
3. Warm-up was rejected in trial 1, accepted in trial 3, and shown to be worthless at the final configuration in trials 11 and 14. Explain each of the three observations. What does this tell you about one-change-at-a-time search?
4. Compare the final validation score with the test score, and with the mean over fresh seeds given in lesson D.4. Which of the three numbers would you report, and why?
5. Open `results/runs/part5/20_ledger.jsonl`. Which fields would you need in order to reproduce trial 6 exactly? Is anything missing?

**Success criterion:** written answers with trial numbers as evidence.

### Exercise 3 (experimental): noise and selection

1. In part 3, compare your number of acceptances at threshold 0 with the expectation from Exercise 1.1. Why can it be lower when the starting configuration was itself produced by a search?
2. Edit the null search to start from a configuration that was *not* selected (use `reseed=7` as the starting point) and run 40 trials at threshold 0. Report acceptances, validation gain and test gain.
3. Add a **replication rule**: a candidate that beats the incumbent must be re-run with a second seed and beat it again before being accepted. Re-run parts 2 and 3. What does it cost in runs, and what does it do to false acceptances?
4. The noise threshold is `max − min` over a few seeds. Propose a better-founded acceptance test using Module 10's tools (paired comparison, bootstrap), and say what it would cost per trial.

**Success criterion:** numbers for items 2 and 3 and a concrete proposal for item 4.

### Exercise 4 (coding): proposers and guard rails

1. Write a proposer that never repeats a proposal already rejected at the current incumbent. How many of the 14 trials of part 2 does it save?
2. Write a coordinate-sweep proposer (try every value of one parameter, keep the best, move to the next parameter). Compare the result and the number of runs with the random proposer.
3. Add a guard rail: reject any proposal whose estimated cost (for example `n_embd² × n_layer`) exceeds a budget. Show a proposal that is in the search space but refused for cost, and how the ledger records it.
4. Part 4's scripted proposer asked to evaluate on the training set. Suppose the search space had (mistakenly) contained a parameter `eval_on: ["val", "train"]`. What would the loop have done, and what would the test score have revealed? State the design rule this illustrates.

**Success criterion:** working code for items 1–3 with ledgers; a written answer to item 4.

### Exercise 5 (nanochat): an experiment loop on the real pipeline

On any machine with the nanochat environment and the miniature data set of Module 08
(CPU is sufficient; each trial takes about half a minute without compilation):

1. Write an `objective(config, seed)` that launches `scripts.base_train` on the miniature data with `--depth`, `--matrix-lr` and `--warmup-steps` taken from `config`, a fixed `--num-iterations`, and `TORCHDYNAMO_DISABLE=1`, then reads the final validation bpb from the log with `scripts/experiments/parse_train_log.py`. Return minus the bpb as `val`; use a second evaluation (`scripts.base_eval --eval bpb` on the train split is *not* acceptable: why?) or leave `test` to a final manual run.
2. Measure the noise over three seeds. `base_train` has no seed argument and `compute_init` fixes the global seed at 42; use [`scripts/experiments/seeded_base_train.py`](../../scripts/experiments/seeded_base_train.py), which shifts the seed without modifying nanochat. Check first that offset 0 reproduces a plain run exactly.
3. Run six trials with `run_experiment_loop` and keep the ledger.
4. Read commit `6ed7d1d` (`git show --stat 6ed7d1d` in `external/nanochat`). Pick two of its listed changes and say what evidence you would want before accepting each on your own hardware.

On a DGX Spark, first size a trial with `make spark-probe` and
`scripts/experiments/estimate_run.py`; a loop of $n$ trials costs $n$ times that.

**Success criterion:** a ledger from real nanochat runs, or (without the environment) the objective function written out and a reasoned answer to item 4. *The author has not run this loop on GPU hardware; no reference values.*

### Challenge (optional): a model as proposer

Connect `llm_proposer` to a real model through `OpenAICompatLLM` (Module 18, Exercise 5)
and compare, over three repetitions each, with the random proposer and with your
coordinate sweep: final test score and number of runs. State the claim you can make from
three repetitions, and the claim you cannot.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Run a bounded, audited improvement loop on real training runs; demonstrate selection effects in adaptive search and the function of each safeguard |
| **Hypotheses** | (H1) Starting from a poor recipe, the loop reduces the test loss by more than a factor of two within 14 trials. (H2) More than half of all trials are rejected. (H3) A search over a parameter with no effect accepts changes at threshold 0 and none at the noise threshold; its validation gain does not appear on the test set. (H4) The validation score recorded for the selected configuration is better than its mean over fresh seeds. (H5) Proposals outside the search space are rejected without a run and appear in the ledger. |
| **Hardware / software** | CPU; course package. Exercise 5: nanochat environment |
| **Tier** | A: about 90 seconds. Exercise 5: tier A on CPU with the miniature data, tier B–C on GPU |
| **Metrics** | validation and test score (minus loss, nats per character); seed-to-seed spread; number of accepted trials; validation gain versus test gain |
| **Independent variables** | configuration parameters; acceptance threshold; proposer |
| **Controls** | fixed training steps, data split and evaluation batches; the test split is read only in the final evaluation; one change per trial |
| **Interpretation** | Runs are 150 steps of a tiny model: conclusions about *which hyperparameters are good* do not transfer. Conclusions about the loop (selection, thresholds, ledgers, guard rails) do. |
| **Common failure modes** | Reporting the incumbent's validation score as the result. Using the test set to choose. Logging only accepted trials. Letting the proposer set its own budget or evaluation. Estimating noise once and never again. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.3–D.5.

| | value | your prediction |
|---|---:|---|
| noise at the starting configuration (5 seeds) | | |
| trials accepted in part 2 (of 14) | | |
| final validation score | | |
| final test score (once) | | |
| null search, threshold 0: accepted / validation gain / test gain | | |
| null search, noise threshold: accepted | | |

**Questions to answer in writing**

1. List everything in `run_experiment_loop` that the proposer cannot influence. For each item, describe the failure that would follow if it could.
2. Which level of "self-improvement" (answer, workflow, weights, process) is your lab run, and which is nanochat's autoresearch? What would have to be added to reach the next level, and what new measurement problem would that create?
3. Why is a complete ledger of rejected trials a safety feature and not only good record-keeping?
