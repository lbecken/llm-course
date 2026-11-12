# Solutions — Module 20

## Lab exercises

### Exercise 1: selection arithmetic

1. Draws 2 to 20 are each a new record with probability $1/i$: $\sum_{i=2}^{20} 1/i = 2.60$ expected acceptances, although nothing is better than anything else.
2. $1.87 \times 0.006 = 0.011$ for 20 draws; $3.24 \times 0.006 = 0.019$ for 1 000. The "improvement" found by picking the best grows with the number of things tried, slowly but without limit.
3. Both scores are noisy, so the difference has standard deviation $0.006\sqrt2 = 0.0085$; a threshold of 0.02 is about 2.4 standard deviations. A single false acceptance has probability under 1 % per trial; over hundreds of trials some will still occur.
4. The noise (spread of the metric across seeds for one configuration); the result on a held-out test set that the 500 experiments never touched (or a replication with fresh seeds); and the number of experiments against which the selection was made, together with the acceptance rule. Also useful: the full ledger.

### Exercise 2: read a ledger

Reference ledger: lesson D.3.

2. Learning rate, by far: trial 2 (3e-4 → 1e-2) moved the score from about −2.71 to −0.63, a gain of 2.1 nats; trial 10 shows 3e-3 is 0.07 worse than 1e-2. Then depth (trial 4: +0.096; trials 8–9 at the final configuration: removing a layer costs 0.053), width (trial 6: +0.077; trial 7: the reverse), and warm-up last.
3. Trial 1: at learning rate 3e-4 the model has barely started to learn in 150 steps; spending 20 of them on warm-up makes that slightly worse. Trial 3: at 1e-2 warm-up avoids early instability and gains 0.023, just above the threshold. Trials 11 and 14: with two layers and width 64 the same change is worth 0.0001. The value of a setting depends on the others. One-change-at-a-time search follows one path through a space with interactions; it can keep a setting that has become irrelevant and can miss combinations that only help together. It is a cheap heuristic, and the ledger is what makes its path inspectable.
4. Validation −0.4293 (selected, optimistic), mean over 20 fresh seeds on validation −0.4374, test −0.4205 (single use, a different text sample, so not directly comparable in level with the validation numbers). Report the **test score** as the result and, where budget allows, the mean and spread over fresh seeds; never the incumbent's validation score alone.
5. The ledger stores index, parameter, old and new value, whether guard rails passed, validation score, incumbent score, accepted, seconds, a note and the configuration *after* the trial. To reproduce trial 6 one needs the candidate configuration (`config_after`, since it was accepted; for a rejected trial: `config_after` with `parameter` set to `new`), plus things the ledger does **not** contain: the seed, the number of training steps, the data and its split, the code version and the software environment. A production ledger should record all of them (Module 21).

### Exercise 3: noise and selection

1. Reference: 2 acceptances in 20 trials, against an expectation of 2.6 for an unselected incumbent. The starting point was the *winner* of part 2's search: its recorded score (−0.4293) is better than 17 of 20 re-runs (mean −0.4374), so fewer random draws beat it. In an extended run of 60 trials from the same incumbent the author observed no acceptance at all.
2. Measured from `reseed=7` (validation −0.4427, a typical draw), 40 trials, threshold 0: **5 accepted, validation gain +0.0160, test gain +0.0012**. Sixteen thousandths of apparent improvement from a parameter that does nothing; essentially none of it on the test set.
3. Measured on the same null search: with replication on a second seed, 1 acceptance instead of 5, for 46 runs instead of 40. A noise candidate that wins once wins the replication about half the time, so replication halves false acceptances at each step rather than eliminating them; requiring two replications, or a threshold as well, tightens further. Cost: two extra runs per *apparent* win, which is cheap because apparent wins are rare.
4. For example: evaluate incumbent and candidate on the same validation batches and compute the paired difference per batch (or per sentence), then a bootstrap interval for the mean difference across $k$ seeds each; accept if the interval excludes zero. Cost: $k$ runs per candidate instead of one (the incumbent's runs are reused). This replaces a fixed threshold by a test whose error rate is known, and uses pairing to remove the variation due to the evaluation data (Module 10 D.6).

### Exercise 4: proposers and guard rails

1. Keep a set of `(parameter, value)` pairs rejected since the last acceptance and resample until the proposal is not in it. In the reference ledger trials 9, 12, 13 (`n_layer 2 → 1` again) and 14 (`warmup 20 → 0` again) are repeats: four of fourteen runs saved.
2. A sweep over 5 + 3 + 2 + 2 values needs 12 runs minus the 4 starting values = 8 runs for one pass and reaches the same final configuration here (lr 1e-2, n_embd 64, n_layer 2; warm-up either). It is more efficient than random proposals in this small space and shares the weakness of item 2.3: the order of parameters matters.
3. Compute the cost in the guard-rail step; record a trial with `passed_tests=False` and a note such as "exceeds cost budget (8192 > 4096)". The proposal is in the space, costs no training run, and is visible in the ledger: refused proposals are part of the audit trail too.
4. The loop would have accepted `eval_on = train` as soon as it was proposed: evaluating on the training set gives a better score, and the loop optimises the score. Every later trial would then be judged on training loss, favouring whatever overfits. The single-use test score would have exposed it as a gap between "validation" and test. Design rule: the evaluation is not a parameter. Anything that defines *how success is measured* must be outside the search space, not merely unlikely to be chosen.

### Exercise 5: nanochat

1. Sketch:

   ```python
   def objective(config, seed):
       tag = f"trial-d{config['depth']}-lr{config['matrix_lr']}-w{config['warmup_steps']}-s{seed}"
       cmd = [sys.executable, SEEDED, str(seed), f"--depth={config['depth']}", f"--matrix-lr={config['matrix_lr']}",
              f"--warmup-steps={config['warmup_steps']}", "--num-iterations=200", "--eval-every=200", f"--model-tag={tag}", *FIXED_FLAGS]
       out = subprocess.run(cmd, cwd=NANOCHAT_DIR, env={**os.environ, "TORCHDYNAMO_DISABLE": "1"}, capture_output=True, text=True, timeout=1800)
       bpb = float(re.search(r"Minimum validation bpb: ([0-9.]+)", out.stdout).group(1))
       return {"val": -bpb, "test": float("nan")}
   ```

   with a timeout per trial (a budget the proposer cannot change) and the fixed flags of `scripts/experiments/nanochat_local.sh`. Evaluating on the train split is not acceptable as a test because the model was fitted to it; a proper test needs text the runs never saw, e.g. a second held-out parquet shard generated with a different seed.
2. Reference (miniature data, depth 4, 60 iterations, CPU, no compilation): offset 0 gives validation bpb 0.969349 twice (exactly reproducible), offset 1 gives 0.942159: a seed-to-seed difference of 0.027 bpb at this very short training length. Any trial-to-trial difference of that size is noise.
3. No reference ledger (depends on your budget per trial).
4. For example: "RoPE base 10 000 → 100 000" and "logit soft-cap 20 → 15". Evidence to ask for on your own hardware: the change in validation bpb at your depth with the *other* settings as you run them (interactions, Exercise 2.3), over at least three seeds, compared with your measured seed noise; and whether the gain persists at the training length you can afford, since the upstream tuning was done at depth 12 for a specific compute budget on different hardware.

### Challenge: a model as proposer

With three repetitions per proposer you can claim: "in these runs, proposer X reached
test score A, B, C using N runs", and whether the ranges overlap. You cannot claim that
one proposer is better unless the ranges are clearly separated; with a search space of
60 configurations, all reasonable proposers find the same optimum and differ only in the
number of runs. The model's possible advantage (using the history, prior knowledge about
learning rates) shows in runs-to-optimum, which needs more repetitions to estimate.

### Written questions

1. The search space (it could propose anything); the guard-rail check; the objective function and which split it reads; the noise threshold and acceptance rule (it could accept its own proposals); `max_trials` and `max_seconds` (no termination); the ledger (no audit); the final test evaluation and its seed (the result would be selected). Each corresponds to a row of the table in lesson D.5.
2. The lab is level 2 (a configuration). nanochat's autoresearch is mostly level 2 with elements of level 4 (changes to model code, reviewed by a human). Reaching level 3 would mean feeding the loop's results back into the *model that proposes*; the new measurement problem is evaluating the proposer itself, across generations, with an evaluation it has not been optimised against.
3. Because it shows what was *tried*, including attempts to leave the search space, to enlarge budgets or to change the evaluation. A record of successes alone cannot reveal that a system is probing its limits, or how many attempts lie behind a reported gain.

## Quiz

1. An answer (a check on that answer); a workflow or configuration (a metric on validation data); the weights (an evaluation the training never saw); the process or code (experimental results under review). Each is an optimisation loop that needs a measure it cannot influence.
2. Propose one change; test it against guard rails; run and evaluate on validation data; decide against the noise threshold; record in the ledger. After the last trial: evaluate the final configuration once on the held-out test set, and hand ledger and result to a human.
3. The same configuration scores differently with different seeds. Without knowing that spread, a lucky run is indistinguishable from an improvement, and a loop that keeps the best of many runs accumulates luck.
4. About **3** ($\sum_{i=2}^{31} 1/i \approx 3.0$).
5. The validation gain is selection: the loop kept the reseeds that happened to score best on the validation batches. Nothing about the model changed, so on independent data the chosen run is no better than any other; the test change is noise around zero and happened to be negative.
6. The configuration a search selects was selected partly because its measured score was favourable, so that score overestimates its true performance (here: better than 17 of 20 re-runs of the same configuration). Unaffected: the score on a test set used once, or a re-run with fresh seeds.
7. The evaluation code and data; the test set; budgets; the search space and permissions; the loop's own code and ledger. Example: if the evaluation is reachable, the loop will adopt changes that make evaluation easier (evaluate on training data, shorten the test) because they raise the score.
8. Because it improves the number, and the proposer's task is to improve the number. No intent is required. Constraints therefore cannot rely on the proposer's goodwill or on instructions; they must be structural: the evaluation is simply not among the things a proposal can change.
9. Every proposal passes through fixed validation, a fixed evaluation and a fixed acceptance rule before it has any effect, and all of it is logged. A wrong or manipulated proposer wastes trials; it cannot change the outcome's validity.
10. Nothing: differences in the fourth decimal are far below any plausible seed noise, and they come after many experiments against the same metric. The claim is carried by the time-to-target column, by a second independent metric (CORE), by the check at a larger scale than the search was run at, and by publication of the exact changes so that others can replicate.
11. Whether the outputs used for training are filtered by an independent, exact check (verified answers, passing tests, game outcomes) and whether progress is measured by an evaluation outside the loop. With an exact filter the loop can exceed its starting point; with none, errors are learned and compound.
12. Any two: each experiment costs compute and time the loop does not create; evaluations saturate and are overfitted by adaptive search; exact filters for self-generated data exist only in narrow domains; improvements found on cheap proxies must transfer; the proposer's own capability must be measured across generations without the measure being optimised away.
