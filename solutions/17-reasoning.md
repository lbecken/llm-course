# Solutions — Module 17

## Lab exercises

### Exercise 1: steps, tokens, votes

1. $2\cdot3 = 6 \equiv 1$; $1-4 = -3 \equiv 2$; $2+1 = 3$. Target: `2*3-4+1=1-4+1=2+1=3`.
2. After the $j$-th reduction the expression has $n-j$ operations, i.e. $2(n-j)+1$ characters. Sum over $j=1..n$: $n^2$. Plus $n-1$ separators: $n^2 + n - 1$. For $n=5$: **29** (the lab's measured 29.0).
3. With all wrong answers distinct, the right answer is the most common as soon as it appears at least twice. $k=3$: $p^3 + 3p^2(1-p) = 0.216 + 0.432 = $ **0.648** (one right and two different wrong answers is a three-way tie, counted as failure). $k=5$: $1 - (1-p)^5 - 5p(1-p)^4 = 1 - 0.0102 - 0.0768 = $ **0.913**.
4. Two possible answers, $p = 0.4$: the right answer wins only with at least 2 of 3: $0.064 + 0.288 = $ **0.352**, *lower* than a single sample. Voting helps when the right answer is the single most probable answer (the mode of the answer distribution), not when it is merely "often right". Scattered wrong answers (item 3) make that easy; concentrated wrong answers (item 4) make voting amplify the error.
5. First attempt right and accepted: $0.5\cdot0.9 = 0.45$. First attempt rejected: $0.5\cdot0.1 + 0.5\cdot0.9 = 0.5$; then the second attempt is right with probability 0.5: $0.25$. Total **0.70** (against 0.50 without verification).

### Exercise 2: two target formats

Reference table: lesson D.1.

2. Row 3 shows that the direct format is learnable for what two layers can represent (1.000 on one operation, 0.665 on two). So in row 1 the model did not fail for lack of capacity for one-step problems: four fifths of its training examples had targets it could not compute, and that noise prevented it from learning even the easy fifth in 2 500 steps. From row 1 alone one would wrongly conclude "this architecture cannot learn modular arithmetic".
3. Reference (scratchpad model, accuracy at 1–5 operations): 1 200 steps: 0.320, 0.235, 0.185, 0.205, 0.155 (chance). 1 600 steps: 0.305, **0.965**, 0.245, 0.185, 0.175. 2 500 steps: 1.000, 1.000, 1.000, 1.000, 0.995. Learning is not gradual: nothing for over a thousand steps, then individual lengths appear abruptly (two-operation chains first), then everything. A loss curve that looks flat does not mean nothing is being learned; and stopping at 1 600 steps would have produced a misleading table.
4. Train a second scratchpad model on 1–8 operations (same steps per example or more) and evaluate at 6: if it reaches high accuracy at 6, the failure was a limit of the *data* (that length was simply unseen). To test the *method*, evaluate that model at 9 or 10: if it again fails just beyond its training range, the method learns length-specific procedures rather than the general algorithm in this model. Expected: both, i.e. each trained length is learned, and each unseen length fails. A block-size check is needed too: at 8 operations the scratchpad needs 71 generated characters and no longer fits the 64-token context.
5. "Running values only": `3*4+2-1=2 4 3`: $2n-1$ tokens instead of $n^2+n-1$. Prediction: harder to learn, because the model must keep track of *which* operand comes next (a pointer into the question) instead of always operating on the first operation of the most recent step. See the challenge.

### Exercise 3: votes and confidence

1. There are five possible answers and the model's samples are spread nearly uniformly, so 15 samples almost always include the right one (0.965). But the right one is not more frequent than the others: the plurality winner is right about as often as a random guess (0.185). Presence among samples is not knowledge.
2. For a short answer, greedy decoding returns approximately the mode of the answer distribution, which is what a large vote estimates. The vote wins when the answer follows a long sampled derivation: there, greedy commits to one derivation while different sampled derivations can converge on the same right answer and outvote scattered wrong ones.
3. At least 80 % agreement: coverage $104/300 = 35\,\%$, accuracy **0.981**. At least 50 %: coverage $151/300 = 50\,\%$, accuracy $(102 + 37)/151 = $ **0.921**. Answering everything: 0.513.
4. Systematic errors: the model consistently produces the same wrong sum. Re-sampling the same model cannot catch them. An independent check can: an exact tool, a different model, or a reformulation of the question (e.g. verifying by subtraction).
5. Sketch: compute agreement from 25 samples; if at least 0.8, return the majority answer; else return `safe_calculate(question)`. Result with the reference numbers: $102 + 196 = 298$ of 300 right (0.993) with the tool called on 65 % of questions. A cascade is preferable to always calling the tool when the tool is expensive, slow, rate-limited or has side effects. Here the calculator is free and exact, so always calling it is better (1.000 with no sampling cost); the exercise is about the pattern.

### Exercise 4: strategies at equal cost

Reference tables: lesson D.5.

1. Budget 5 calls: at 4 operations "verify and retry, exact check" (0.968 at 1.8 calls), or without an exact checker "verify, model checks" (0.865 at 3.9); at 8 operations "verify, exact check" (0.725 at 2.7); decomposition (0.672) needs 8 calls and is over budget. Budget 15: at 4 operations vote k=15 (0.993); at 8 operations "verify, exact check" still leads (0.725) among those listed, with decomposition (0.672 at 8 calls) the best that needs no checker.
2. `p_step` (0.05) is smaller than `p_direct` (0.15 per operation): a single operation is assumed easier than an operation performed implicitly inside a long chain. Part A supports the direction strongly for a small model: single reductions were learned perfectly while multi-step direct answers were at chance.
3. Reference at 8 operations: vote k=15: 0.608; k=45: 0.898 (45 calls). Verify with exact checker, at most 4 attempts: 0.725 at 2.7 calls; at most 12 attempts: 0.968 at 3.6 calls. With an exact checker, extra attempts are spent only on problems that need them; voting spends the full budget on every problem.
4. Reference at 8 operations: accuracy falls from 0.593 (10 % wrong checks) to 0.380 (30 %), at 4.5 calls. A poor checker rejects right answers (forcing re-samples that are more likely wrong than right when $p$ is small) and accepts wrong ones. It converts budget into noise; below some quality it is worse than taking the first answer of a stronger strategy.
5. Reference: accuracy 0.568 in both; calls 3.09 with the blind spot, 4.79 without. With the blind spot the checker *approves* the solver's systematic wrong answer, so the loop stops early and reports success. Without it the checker keeps rejecting, the loop exhausts its attempts, and although the final answer is the same wrong one, the system *knows* that it never obtained an approved answer: it can abstain or escalate. The value of an independent checker is often the failure signal, not a better answer.

### Exercise 5: nanochat

1. Reference (miniature `pocket-d4`, 32 samples of one token, seed 1): `like` 5, `see` 5, `help` 5, `wait` 4, `run` 4, `sleep` 3, `sing` 3, `chase` 2, and one stray token. No word has a majority; the vote over words is nearly a tie. But 31 of 32 samples are plural verb forms, which is the grammatical property (the subject is `cats`). Answers must be mapped to the equivalence class you care about before voting: surface variation that does not matter must not split the vote. (For GSM8K that normalisation is "the number after `####`".)
2. One prefill pass over the prompt (batch 1), one replication of the cache to 32 rows, one decode step per generated token at batch 32. Here: 1 prefill and 1 decode step.
3. Sketch:

   ```python
   from collections import Counter
   from tasks.gsm8k import extract_answer
   answers = [extract_answer(tokenizer.decode(seq[prefix_length:])) for seq in generated_token_sequences]
   valid = [a for a in answers if a is not None]
   voted = Counter(valid).most_common(1)[0][0] if valid else None
   reference = extract_answer(conversation["messages"][-1]["content"][-1]["text"])
   is_correct = voted is not None and voted == reference
   ```

   `extract_answer` returns `None` when no `####` marker is found. Such samples should not vote (they are not answers); if all samples are `None` the question counts as wrong.
4. No reference values (requires a trained SFT model on a GPU).

### Challenge: a cheaper scratchpad

Measured (same model, data and seed as lab A1; target `3*4+2-1=2 4 3`):

| Format | steps | 1 op | 2 | 3 | 4 | 5 | tokens at 5 ops |
|---|---:|---:|---:|---:|---:|---:|---:|
| rewriting (lab A1) | 2 500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 29 |
| running values | 2 500 | 0.215 | 0.255 | 0.190 | 0.175 | 0.210 | 9 |
| running values | 5 000 | 0.315 | 0.335 | 0.240 | 0.590 | 0.230 | 9 |

A third of the tokens, and not learned in twice the training time (one length is
beginning to appear at 5 000 steps). In the rewriting format the next step is always
"apply the first operation of the previous line, copy the rest": the same local rule
every time, with everything needed adjacent. In the running-values format the model must
locate the $i$-th operation of the *question* for its $i$-th output, which requires
counting its own outputs: a harder computation per token. Cheaper tokens, harder tokens.
A good scratchpad makes every token easy to compute from what is already written; the
optimum depends on the model, and a larger model would tolerate the terser format.

### Written questions

1. A forward pass performs a fixed number of sequential processing stages. A result that depends on a longer chain of dependent operations cannot be produced in one pass. When the model writes an intermediate result into its context, the next pass reads that result as input and starts where the previous one stopped, so the chain is split across passes, each short enough to compute.
2. (a) The baseline: 61 % with greedy decoding or with one sample at temperature > 0? (b) The cost: forty times the tokens; what would a larger model or a tool achieve for that budget? (c) The uncertainty and the nature of errors: how many problems, is +7 points outside the noise, and does the gain persist on problems where errors are systematic?
3. Trustworthy when the task cannot be done without the text: the scratchpad model's steps are the computation (remove them and it is at chance; corrupt a step and the answer changes accordingly). Not trustworthy as explanation when the model could answer directly and the text is an accompaniment, or when it is generated after the answer. In between, test it: perturb the reasoning and see whether the answer follows.

## Quiz

1. One pass has a fixed number of layers, hence a bounded number of sequential computation steps. A task needing more dependent steps than that cannot be completed in a pass. Generated tokens are fed back as input, so intermediate results written out by one pass are available to the next: the computation is spread over several passes.
2. (a) Reasoning as generated intermediate text: weights unchanged at inference (but the behaviour was trained). (b) Strategies around the model (voting, verify-and-retry, decomposition): unchanged. (c) Search over partial solutions: unchanged. (d) Training for reasoning (SFT on worked steps, RL on final answers, distillation): weights change.
3. The architecture can learn single steps (third model), so the first model's failure is not incapacity for easy problems: being trained mostly on targets it cannot compute prevented learning altogether. The second model shows the same network solves all lengths when each target token is locally computable. Together: the limit of direct answering is serial depth, and worked steps help both training and inference.
4. $n^2 + n - 1 = 16 + 4 - 1 = $ **19**.
5. Right is the most common if it appears at least twice: $p^3 + 3p^2(1-p) = 0.343 + 0.441 = $ **0.784**.
6. When errors are systematic: the model gives the same wrong answer on every sample, so the samples agree. Verify-and-retry with the same model as checker (shared blind spot) and re-sampling with any checker fail too, since re-sampling reproduces the answer. Escaping needs independent information: a tool, a different model, a reformulation, or training.
7. Not useless: it does not raise accuracy here, but the *agreement* among samples is a confidence estimate (98 % accuracy at high agreement, 10 % at low), which enables abstention, escalation to a tool, or selective review. Greedy decoding gives one answer and no such signal.
8. Model as checker: makes both kinds of error and tends to share the solver's blind spots; the benefit is bounded by its error rates, and a poor checker can make results worse. Exact checker: accepts only right answers, so accuracy is bounded only by whether a right answer is ever sampled within the attempt budget; it exists only for tasks with cheap verification.
9. Any three: equal cost (tokens or calls) or the cost reported; the right baseline (greedy, not one hot sample); the same problems, with uncertainty on the difference; evaluation within and beyond the trained lengths/formats; a check whether a tool makes the comparison moot.
10. An evaluator that judges partial solutions better than chance (and a decomposition into steps). Exists: games with a value function, formal proofs with a proof checker, code with tests. Usually missing: open-ended writing, questions of fact without a reference, most multi-step real-world planning.
11. The whole sampled sequence, reasoning included, receives the gradient weighted by the final reward, so whatever text tends to precede right answers becomes more likely; the reasoning is shaped by outcomes rather than by imitating human derivations. Risk carried over: the reward is all that is optimised. If it can be satisfied without sound reasoning (a leaky checker, guessable answers), the model learns that, and its reasoning text need not be a faithful account of how it gets answers.
12. (a) The explanation was generated after the answer, conditioned on it: it is a rationalisation, a plausible derivation of a given conclusion, not a record. (b) Even reasoning generated before an answer has been shown not always to determine it (answers influenced by unmentioned cues; answers unchanged when the reasoning is corrupted). An explanation is evidence only to the extent that intervening on it changes the answer.
