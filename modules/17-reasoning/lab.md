# Module 17 — Lab: buying computation with tokens and with calls

Script: [`labs/agents/17_reasoning_benchmark.py`](../../labs/agents/17_reasoning_benchmark.py) ·
Code under study: [`mlle/agents/reasoning.py`](../../mlle/agents/reasoning.py), [`mlle/agents/llm.py`](../../mlle/agents/llm.py), `mod_chain` in [`mlle/data/tasks.py`](../../mlle/data/tasks.py) ·
Solutions: [`solutions/17-reasoning.md`](../../solutions/17-reasoning.md)

Part A trains three small models (about 3.5 minutes on a 4-core CPU) and uses the cached
chat model of Part IV. Part B is a simulation and runs in seconds (`--part b`).

## G. Exercises

### Exercise 1 (by hand): steps, tokens, votes

1. Evaluate `2*3-4+1` left to right modulo 5 and write the scratchpad target the lab would train on.
2. How many tokens does the scratchpad model generate for a chain of $n$ operations (single-digit operands)? Check your formula against the lab's "tokens generated at 5 ops".
3. Each sample is right with probability $p = 0.6$, independently. If every wrong sample gives a *different* wrong answer, what is the probability that the majority of 3 samples is right? Of 5?
4. Now $p = 0.4$ and there are only two possible answers, so all wrong samples agree. What is the probability that the majority of 3 is right? What general condition for voting to help does this illustrate?
5. A checker approves a correct answer with probability 0.9 and a wrong answer with probability 0.1. The solver is right 50 % of the time, independently per attempt. For verify-and-retry with at most 2 attempts (the second answer is returned unchecked if the first is rejected), compute the final accuracy.

**Success criterion:** numerical answers; item 2 gives 29 for $n = 5$.

### Exercise 2 (experimental): the same network, two target formats

```bash
python labs/agents/17_reasoning_benchmark.py --part a
```

1. Answer the lesson's "predict before you run" question. Record table A1.
2. Explain row 1 with the help of row 3. What would you conclude about the direct format if you had only run row 1?
3. Re-run with `--steps 1200` and `--steps 1600`. Describe how the scratchpad model's accuracy develops. Is learning gradual?
4. The scratchpad model fails at six operations. Is this a limit of the *method* or of the *training data*? Design the experiment that decides, and state what result would support each answer.
5. For a 5-operation chain the scratchpad costs 29 generated tokens. Suggest a cheaper scratchpad format for this task and predict whether it is easier or harder to learn. (The challenge exercise tests one.)

**Success criterion:** tables for three step counts and a written experiment design for item 4.

### Exercise 3 (experimental): votes and confidence

Use sections A2 and A3 of the output.

1. In A2 the right answer is among 15 samples 96 % of the time, yet the vote is right 19 % of the time. Explain both numbers.
2. In A3, why is greedy decoding as good as a 25-sample vote? Under which circumstances would you expect the vote to win?
3. Design an abstention policy from the agreement table: for the thresholds "answer if at least 80 % agree" and "answer if at least 50 % agree", compute coverage (fraction of questions answered) and accuracy on the answered questions.
4. The high-agreement group still contains wrong answers. What kind of error are they, and which of the strategies of this module could catch them?
5. Combine modules: write 20 lines that answer with the model when agreement is at least 80 % and call the calculator (Module 16's `safe_calculate`) otherwise. What accuracy and what fraction of tool calls do you get? When is such a cascade preferable to always calling the tool?

**Success criterion:** numbers for item 3 and working code for item 5.

### Exercise 4 (experimental): strategies at equal cost

```bash
python labs/agents/17_reasoning_benchmark.py --part b
```

1. Record tables B1 and B2. For each of 4 and 8 operations, which strategy is best under a budget of 5 calls per problem? Of 15?
2. In `NoisySolver`, which assumption makes decomposition better than direct answering? Is that assumption plausible for real models? Cite evidence from part A.
3. Raise the voting budget to `k=45` at 8 operations and the retry budget to `max_attempts=12` with the exact checker. Compare accuracy per call.
4. Set `p_check_err=0.3`. What happens to "verify and retry, model checks", and why can a poor checker be worse than no checker?
5. With `systematic=0.5`, compare `blind_spot=0.9` and `blind_spot=0.0` for the model checker. Accuracy is identical; what differs, and why does that difference matter to a system built on top?

**Success criterion:** a table of your variations with one sentence of explanation each.

### Exercise 5 (nanochat): samples from the engine

```bash
export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local     # the miniature model of Module 08, CPU
cd external/nanochat
python - <<'PY'
from collections import Counter
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
_, _, _, _, device = compute_init(autodetect_device_type())
model, tok, _ = load_model("base", device, phase="eval", model_tag="pocket-d4")
engine = Engine(model, tok)
prompt = tok.encode("the cats that chase the dog", prepend=tok.get_bos_token_id())
seqs, masks = engine.generate_batch(prompt, num_samples=32, max_tokens=1, temperature=1.0, seed=1)
print(Counter(tok.decode(s[len(prompt):]) for s in seqs).most_common())
PY
```

1. Record the distribution of the next token over 32 samples. Does any single word win a majority? Now vote on the *property* "is the verb plural" (the subject is `cats`; `dog` is a distractor). What does the comparison tell you about what must be done to answers before voting?
2. How many prefill passes and how many decode steps did the 32 samples cost? (Module 11 D.6.)
3. Read `run_gsm8k_eval` in `scripts/chat_rl.py`. Write, without running it, the ten additional lines that would turn its pass@k records into majority-vote accuracy. What must `extract_answer` return for a sample without a `####` marker, and how should the vote treat it?
4. *(GPU, with your SFT model.)* Implement item 3 and compare pass@1, vote@8 and pass@8 on 100 GSM8K test problems.

**Success criterion:** items 1–3 on CPU. Item 4: *awaiting validation on GPU, no reference values.*

### Challenge (optional): a cheaper scratchpad

Train the 2-layer model with the target format "running values only":
`3*4+2-1=2 4 3` (5 generated tokens instead of 11 for three operations). Compare
accuracy by chain length with the rewriting scratchpad at equal training steps. Explain
the result in terms of what each format requires the model to keep track of, and relate
it to the statement "the best scratchpad is the one whose every token is easy to
compute from what is already written".

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Show that intermediate tokens extend the computation a fixed-depth model can perform; measure voting and agreement on real models; compare inference-time strategies at equal cost under controlled error properties |
| **Hypotheses** | (H1) On mixed chains modulo 5 with three or more operations, the direct 2-layer model stays within 0.1 of chance; the scratchpad model exceeds 0.95 on trained lengths. (H2) Trained on 1–2 operations only, the direct format exceeds 0.95 on one operation. (H3) The scratchpad model falls below 0.5 at an unseen length. (H4) Majority voting does not raise the direct model above chance. (H5) On the chat model, voting does not beat greedy decoding by more than 0.03, while accuracy differs by more than 0.5 between high- and low-agreement questions. (H6) In simulation, voting gains vanish as errors become systematic. |
| **Hardware / software** | CPU; course package. Exercise 5: nanochat environment and the miniature checkpoint |
| **Tier** | A: part A ≈ 3.5 min, part B ≈ 5 s |
| **Metrics** | exact-match accuracy by chain length; generated tokens per answer; majority-vote accuracy and "right answer among k samples"; accuracy by agreement bucket; model calls per problem |
| **Independent variables** | target format; training lengths; number of samples; strategy; assumed error rates and their systematic share |
| **Controls** | identical architecture, data, seed and steps for the trained models; identical problems for all strategies; fixed sampling seeds |
| **Interpretation** | Part A: one seed, 200–300 problems per cell (standard error up to 0.035). Part B is a simulation: conclusions are of the form "if errors behave like this, then". |
| **Common failure modes** | Comparing a k-sample vote with a single sample instead of greedy decoding. Omitting the cost column. Reading simulated numbers as properties of real models. Concluding from row 1 of A1 that "the model cannot learn the task". |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.1, D.4, D.5.

| Model | 1 op | 2 | 3 | 4 | 5 | 6 | tokens at 5 ops | predicted (4 ops) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direct, 1–5 ops | | | | | | | | |
| scratchpad, 1–5 ops | | | | | | | | |
| direct, 1–2 ops | | | | | | | | |

| Chat model | greedy | vote@5 | vote@25 | any@25 |
|---|---:|---:|---:|---:|
| accuracy | | | | |

| Agreement | questions | accuracy |
|---|---:|---:|
| ≥ 80 % | | |
| 50–80 % | | |
| < 50 % | | |

**Questions to answer in writing**

1. In one paragraph, without the word "think": why does writing intermediate results help a Transformer?
2. A paper reports that "self-consistency with 40 samples improves accuracy from 61 % to 68 %". List the three things you check before being impressed.
3. When is the reasoning text of a model trustworthy as an explanation, and when is it not? Use the scratchpad model as one of your examples.
