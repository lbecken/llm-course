# Reference result: Part V and VI labs (tools, reasoning, agents, experiments, integration) on CPU

| | |
|---|---|
| Date | 2026-11-12 and 2026-11-13 |
| Hardware | 4-vCPU x86-64 sandbox, 15 GiB RAM, no GPU |
| Software | Python 3.13, torch 2.15.0+cpu |
| Status | **Measured** by the course author, one seed per cell unless stated. Rows marked *simulated* come from a solver with assumed error rates (`mlle.agents.llm.NoisySolver`, `SimulatedReActLLM`): they are consequences of those assumptions, not properties of any language model |

Trained models are MiniGPTs with 2 layers × 96 (about 232 k parameters) unless stated.

## Lab 16 — `python labs/agents/16_tool_assistant.py` (about 3 min)

Three assistants fine-tuned for 1 000 steps on 3 000 questions (additions up to 3 digits,
products up to 2 digits). 200 held-out questions per cell, greedy decoding.

| Model | Condition | additions | multiplications | 2-digit × 2-digit | fabricates an output | answer = tool output |
|---|---|---:|---:|---:|---:|---:|
| direct | — | 0.795 | 0.650 | 0.025 | 0.00 | n/a |
| tool-trained | tool on | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 |
| tool-trained | tool off | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| tool-trained, outputs unmasked | tool on | 0.970 | 0.990 | 1.000 | 0.00 | 0.99 |
| tool-trained, outputs unmasked | tool off | 0.610 | 0.495 | 0.035 | 1.00 | 0.00 |
| tool-trained | lying tool (+1) | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 |

Beyond the training range: 4-digit + 4-digit: direct 0.000, tool 0.000, request copied
exactly 0.000, answer equals tool output 0.990. 3-digit × 3-digit: 0.000, 0.000, 0.010,
0.365. Direct model on multiplications by operand size: both one digit 101/104, one
two-digit operand 27/64, both two digits 2/32. Sampling at temperature 1.0 (400
questions): tool-trained 0.9925 (greedy 1.000), direct 0.67 (greedy 0.7225).

## Lab 17 — `python labs/agents/17_reasoning_benchmark.py` (about 3.5 min)

Chains evaluated left to right modulo 5 (chance 0.20); 2 500 training steps; 200 chains per cell.

| Model | 1 op | 2 | 3 | 4 | 5 | 6 (unseen) | tokens at 5 ops |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct, trained on 1–5 operations | 0.250 | 0.190 | 0.165 | 0.215 | 0.185 | 0.200 | 1 |
| scratchpad, trained on 1–5 operations | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 0.115 | 29 |
| direct, trained on 1–2 operations | 1.000 | 0.665 | 0.185 | 0.230 | 0.205 | 0.195 | 1 |

Scratchpad model at 1 200 steps: 0.320, 0.235, 0.185, 0.205, 0.155; at 1 600 steps: 0.305,
0.965, 0.245, 0.185, 0.175. "Running values" format (`3*4+2-1=2 4 3`): 2 500 steps: 0.215,
0.255, 0.190, 0.175, 0.210; 5 000 steps: 0.315, 0.335, 0.240, 0.590, 0.230.

Voting on the direct model (4 operations, temperature 1.0): majority right 0.160 / 0.175 /
0.185 for k = 1 / 5 / 15; right answer among samples 0.160 / 0.610 / 0.965.
Voting on the under-trained chat model (300 held-out additions): greedy 0.523; majority
0.470 / 0.527 / 0.513 / 0.513 for k = 1 / 5 / 15 / 25; among samples 0.470 / 0.630 / 0.753 /
0.810. Agreement of 25 samples: ≥ 80 %: 104 questions, 0.981 right; 50–80 %: 47, 0.787;
< 50 %: 149, 0.101.

*Simulated* (400 problems per row; accuracy / calls):

| Strategy | 4 operations | 8 operations |
|---|---|---|
| direct | 0.525 / 1.0 | 0.270 / 1.0 |
| decompose | 0.795 / 4.0 | 0.672 / 8.0 |
| vote k=5 | 0.787 / 5.0 | 0.398 / 5.0 |
| vote k=15 | 0.993 / 15.0 | 0.608 / 15.0 |
| verify and retry, model checks | 0.865 / 3.9 | 0.593 / 5.0 |
| verify and retry, exact check | 0.968 / 1.8 | 0.725 / 2.7 |
| tool only | 1.000 / 0 | 1.000 / 0 |

Systematic share 0 / 0.5 / 1 (6 operations): direct 0.407 / 0.400 / 0.380; vote k=15
0.868 / 0.630 / 0.380; verify with model 0.738 / 0.568 / 0.380; verify with exact check
0.845 / 0.618 / 0.380.

## Lab 18 — `python labs/agents/18_agent_loop.py` (seconds; *simulated* policy)

200 number-game tasks, budget 20 steps and 3 invalid actions.

| Policy | success | mean steps | outcomes |
|---|---:|---:|---|
| search | 1.000 | 5.1 | success 200 |
| heuristic in code | 0.970 | 10.3 | success 194, step budget 6 |
| simulated model, 0 % mistakes | 0.970 | 10.3 | success 194, step budget 6 |
| simulated model, 20 % mistakes | 0.610 | 11.4 | success 122, too many invalid 53, step budget 25 |
| simulated model, 50 % mistakes | 0.150 | 8.1 | too many invalid 170, success 30 |

Success by optimal plan length (3–8 moves): 20 % mistakes: 0.80, 0.64, 0.69, 0.58, 0.32,
0.29; 50 %: 0.36, 0.19, 0.15, 0.06, 0.09, 0.00. Settings at 50 % mistakes: 1 invalid: 0.010
(2.7 steps); 10 invalid: 0.455 (16.0); 8 steps: 0.085 (6.7); 60 steps with loose guards:
0.580 (27.4), including 46 episodes ending as `failed`.

## Lab 19 — `python labs/agents/19_multi_agent.py` (seconds; *simulated*)

| System | 4 operations: accuracy / calls | 8 operations |
|---|---|---|
| single call | 0.525 / 1.0 | 0.270 / 1.0 |
| team, same-model reviewer | 0.940 / 8.5 | 0.915 / 15.3 |
| team, tool reviewer | 0.990 / 6.4 | 0.983 / 13.2 |
| cost-matched vote | 0.892 / 8.0 | 0.608 / 15.0 |
| single agent, same workflow | 0.940 / 8.5 | 0.915 / 15.3 |

Reviewer (6 operations): none 0.735 / 7.0; same model 0.938 / 11.7; careless 0.825 / 13.6;
tool 0.983 / 9.4. Half-systematic solver (accuracy / calls / wrong answers approved): same
model 0.545 / 2.9 / 0.370; independent model 0.552 / 4.0 / 0.062; tool 0.595 / 2.1 / 0.000.
Plan validation: 0.943 / 10.7 without, 0.990 / 9.5 with. With `p_step = p_direct = 0.15`:
team 0.787 (4 ops), 0.550 (8 ops) against cost-matched vote 0.963, 0.720.

## Lab 20 — `python labs/agents/20_experiment_loop.py` (about 90 s, 71 real training runs)

Noise at the starting configuration (5 seeds): 0.0187. Accepted in part 2: trials 2, 3, 4,
6 of 14. Final configuration lr 0.01, n_embd 64, n_layer 2, warmup 20: validation −0.4293,
test (once) −0.4205; starting configuration test −2.6390. Null search over `reseed`, 20
trials: threshold 0: 2 accepted, validation +0.0053, test −0.0058; noise threshold: 0
accepted. Selected configuration re-run with 20 other seeds: −0.4476 … −0.4240, mean
−0.4374. Null search from an unselected start (40 trials, threshold 0): 5 accepted,
validation +0.0160, test +0.0012; with replication on a second seed: 1 accepted.
nanochat seed wrapper (miniature data, depth 4, 60 iterations): offset 0: 0.969349 bpb
(twice), offset 1: 0.942159.

## Lab 21 — `python labs/integration/21_end_to_end.py` (about 50 s with the base model cached)

| Stage | seconds | metrics |
|---|---:|---|
| pretrain (cached) | 0.2 | 439 288 training tokens; 231 744 parameters |
| eval-base | 1.0 | Pocket loss 0.4034; raw completion accuracy 0.755 (200 questions) |
| fine-tune (600 steps) | 40.4 | loss 4.764 → 0.458; 39.2 % of positions supervised |
| serve-eval | 7.7 | tool on 1.000, tool off 0.000; 1 149 decode tokens/s |
| agent | < 0.1 | 5 of 5 episodes succeed |

A second run with the same seed: every metric identical except timings (1 272 tokens/s).

## Lab 22 — `python labs/integration/22_retrieval.py` (about 1.5 min)

208 questions per row; right / abstains / repeats the context.

| Model | Condition | right | abstains | repeats |
|---|---|---:|---:|---:|
| closed-book | table as trained | 1.00 | 0.00 | — |
| closed-book | changed table | 0.00 | 0.00 | 1.00 (old value) |
| open-book | correct retrieval, old / changed table | 1.00 / 1.00 | 0.00 | 1.00 |
| open-book | fact about another key | 0.02 | 0.00 | 1.00 |
| open-book | wrong value for the key | 0.00 | 0.00 | 1.00 |
| open-book, checking | correct retrieval, old / changed table | 1.00 / 1.00 | 0.00 | 1.00 |
| open-book, checking | fact about another key | 0.00 | 1.00 | 0.00 |
| open-book, checking | wrong value for the key | 0.00 | 0.00 | 1.00 |

An earlier version of this lab did not shuffle the closed-book documents; the last two
keys then fell entirely into the validation split and closed-book accuracy was 0.92.

## Not measured

Nothing in Parts V–VI was run on a GPU or with a real language model behind the agent
loops. Exercises that call for either are marked in the lab sheets and have no reference
values.
