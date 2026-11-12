# Module 09 — Lab: count, sweep, plan

Scripts: [`labs/pretraining/09_param_flops.py`](../../labs/pretraining/09_param_flops.py) ·
[`labs/pretraining/09_scaling_sweep.py`](../../labs/pretraining/09_scaling_sweep.py) ·
[`scripts/experiments/estimate_run.py`](../../scripts/experiments/estimate_run.py) ·
[`scripts/experiments/spark_probe.sh`](../../scripts/experiments/spark_probe.sh) ·
Solutions: [`solutions/09-scaling-and-compute.md`](../../solutions/09-scaling-and-compute.md)

## G. Exercises

### Exercise 1 (mathematical): size a model by hand

For nanochat with `--depth=16` (defaults otherwise: aspect ratio 64, head dimension 128,
vocabulary 32 768, sequence length 2 048, `--target-param-data-ratio=12`):

1. Compute the width and the number of heads.
2. Compute block-matrix parameters ($12\,L\,C^2$, ignore the gate weights), `lm_head` parameters, and the scaling-parameter count.
3. Compute the token budget.
4. Estimate FLOPs per token using only the $6 N$ term, and the total training FLOPs.
5. Check against `python scripts/experiments/estimate_run.py --depths 16`. By how much does the attention term change item 4?
6. If depth is increased to 32, by what factor do parameters, tokens and total compute grow?

**Success criterion:** items 1–3 match the script exactly (up to the ignored gate weights); item 4 within 10 %.

### Exercise 2 (measurement): does the FLOP rule predict time?

```bash
python labs/pretraining/09_param_flops.py
```

1. Record "achieved FLOPs/s" for each model. Is it constant? Which model deviates most and why?
2. Add a configuration with `block_size=1024` (edit the list) and predict the attention share before running.
3. With `--nanochat` (needs `make env-cpu`): what fraction of a depth-4 nanochat model's parameters are lookup tables? Of a depth-26 model? Why does the fraction fall?

**Success criterion:** an explanation of the deviation in terms of fixed overhead versus per-FLOP cost.

### Exercise 3 (experimental): the frontier

```bash
python labs/pretraining/09_scaling_sweep.py            # ~8 minutes; --quick for 2 budgets
```

1. Predict the best width for each budget before running.
2. From `results/runs/scaling_sweep.jsonl`, compute the tokens-per-matmul-parameter ratio of the best model at each budget. Is it roughly constant?
3. Re-run the two largest budgets with `--seed 1` and `--seed 2`. Is the ranking of widths 16 and 32 stable?
4. For width 8, plot or tabulate loss against budget. What is its apparent floor, and what does that floor represent?

**Success criterion:** a table over three seeds for the two largest budgets, and a statement of which conclusions survive.

### Exercise 4 (planning, GPU): from probe to plan

On your GPU machine:

```bash
DEPTH=6  scripts/experiments/spark_probe.sh
DEPTH=12 scripts/experiments/spark_probe.sh
python scripts/experiments/estimate_run.py --depths 6 8 12 16 --window-pattern L --tok-per-sec <your measured value for the nearest depth>
```

1. Record tokens/s and peak memory for both depths. Is tokens/s × FLOPs-per-token (achieved FLOPs/s) the same at both depths?
2. Using each depth's own measured throughput, compute hours for its full token budget.
3. Decide which depth you will train for Milestone 2 and justify it in two sentences.

**Success criterion:** two probe JSON files and a written plan. *Awaiting validation: no reference numbers exist for the DGX Spark at the pinned revision.*

### Exercise 5 (Milestone 2, part 2): a controlled nanochat experiment

Investigate **one** variable with nanochat itself. Suggested designs (pick one; each
compares runs that differ in exactly one flag):

| Question | Runs | Control |
|---|---|---|
| Does the tokens:params ratio matter at fixed compute? | depth 6 at ratio 12 vs. depth 8 at the ratio that gives the same total FLOPs (`--target-flops`) | equal `--target-flops` |
| What does the sliding window cost in quality on your hardware's attention path? | `--window-pattern L` vs `SSSL` at depth 6 | equal `--num-iterations` |
| How much does batch size matter? | `--total-batch-size` 131072 vs 262144 vs 524288 at depth 6 | equal total tokens (adjust `--num-iterations`) |

On a CPU-only machine, run the same design on the local dataset with
`scripts/experiments/nanochat_local.sh` (`EXTRA="--window-pattern SSSL"` etc.).
Fill in an [experiment record](../../results/templates/experiment-record.md), with the
prediction written first.

**Success criterion:** a completed record with prediction, exact commands, one changed variable, validation bpb for each run, and a conclusion that states its own limits.

### Challenge (optional): fit an exponent

Extend the sweep to five budgets and widths {8, 16, 32, 64, 128}. For each budget fit a
parabola to loss versus $\log_2(\text{width})$ and take its minimum as the optimal width;
convert to matmul parameters; fit $N_\text{opt}\propto C^{a}$ by linear regression in
log-log space. Compare $a$ with Chinchilla's 0.5 and nanochat's 0.54, and list three
reasons your number may differ.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Verify the parameter/FLOP accounting; observe an interior optimum of model size at fixed compute and its shift with budget; plan a real run from a measurement |
| **Hypotheses** | (H1) Measured time per token is proportional to estimated FLOPs for all but the smallest models. (H2) At every budget some intermediate width is best. (H3) The best width is non-decreasing in the budget. |
| **Hardware / software** | Ex. 1–3: CPU. Ex. 4–5: GPU with nanochat environment (CPU variant of Ex. 5 possible) |
| **Tier** | Ex. 2: A (seconds). Ex. 3: A–B (8 min; ~25 min with three seeds). Ex. 4: B. Ex. 5: B–C |
| **Metrics** | parameters by group; FLOPs/token; achieved FLOPs/s; validation loss; tokens/s; peak memory; validation bpb |
| **Independent variables** | width and depth (Ex. 2); (budget, width) (Ex. 3); one training flag (Ex. 5) |
| **Controls** | Ex. 3: data, depth, batch, learning rate, seed; compute equalised by construction. Ex. 5: everything except the chosen flag; compare at equal FLOPs or equal tokens, stated explicitly |
| **Interpretation** | Differences smaller than the seed spread (±0.003 nats/char for MiniGPT on Pocket) are ties. An undertrained model shows few steps and high loss; a capacity-limited one shows many steps and a loss that barely moves with budget. |
| **Common failure modes** | Comparing models at equal *steps* and calling it equal compute. Reading `params_total` when the rule needs matmul parameters. Sweeps on a machine that is also doing something else (throughput noise). |

## I. Results and interpretation

Reference (author, CPU, seed 0): see lesson D.2 and D.6.

| budget | width 8 | width 16 | width 32 | width 64 | best (yours) |
|---:|---:|---:|---:|---:|---:|
| 1.0e10 | | | | | |
| 4.0e10 | | | | | |
| 1.6e11 | | | | | |
| 6.4e11 | | | | | |

| GPU probe | depth 6 | depth 12 |
|---|---:|---:|
| tokens/s (median) | | |
| peak memory (GiB) | | |
| achieved FLOPs/s | | |
| hours for full budget | | |

**Questions to answer in writing**

1. Which of H1–H3 held? With how many seeds?
2. The sweep keeps depth fixed and varies width; nanochat varies both together. Would you expect the same qualitative result? What would you need to check?
3. State one conclusion from your Milestone 2 experiment that you are confident transfers to a larger model, and one that you are not.
