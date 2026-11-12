# Solutions — Module 18

## Lab exercises

### Exercise 1: compounding, budgets, traces

1. $0.95^5 = 0.774$; $0.95^{20} = 0.358$; $0.95^{50} = 0.077$. For 90 % at 50 steps: $p = 0.9^{1/50} = 0.9979$, i.e. at most 0.2 % error per step.
2. Three tries: $1 - 0.05^3 = 0.999875$ per step; $0.999875^{50} = 0.994$. Optimistic because it assumes (a) a perfect verifier for every step, (b) independent retries, whereas the same model in the same context tends to repeat its mistake (Module 17: systematic errors), and (c) that a wrong step can be undone or was never executed.
3. Statuses: 1 ok, 2 ok, 3 ok, 4 **invalid** (first invalid action, below the limit of 2, so the episode continues and the state stays 10), 5 ok, 6 ok. State 8 is visited for the second time at step 6, which does not exceed the loop window of 2. The loop runs out of its 6 steps: outcome **`step_budget`**.
4. (a) The model says DONE when the goal is not reached: it misjudges the state, or was instructed by injected text to stop. (b) The model says DONE having reached something that *looks* like the goal (a plausible answer, a passing subset of tests), i.e. it satisfies its own criterion rather than yours. Also: DONE as the path of least resistance after repeated errors.

### Exercise 2: outcomes and policies

Reference tables: lesson D.3 and D.4.

2. Reference trace (8 → 114; optimal plan `add3, add3, double, double, double, add3, sub1`, 7 moves): the heuristic doubles to 64, then adds 3 seventeen times (67, 70, … 115) and runs out of steps one above the target. It is myopic: "double while the double does not overshoot" ignores that adding *before* doubling is multiplied later. Search does not have this problem because it evaluates whole sequences against a model of the environment instead of applying a local rule.
3. `too_many_invalid` means the model's replies were not usable (malformed, or not legal actions), so first fix the interface: a stricter output format or constrained decoding, clearer listing of legal actions, and error observations that say what was wrong. Raising `max_invalid` (the loop) is the cheap mitigation; the model is the last thing to change. Had `step_budget` dominated, the policy's *choices* would be the problem.
4. $(0.8)^n$: 0.51, 0.41, 0.33, 0.26, 0.21, 0.17 for $n = 3..8$; measured 0.80, 0.64, 0.69, 0.58, 0.32, 0.29. The stand-in's mistakes are of three kinds in equal shares: malformed reply, illegal action, arbitrary legal action. The first two never reach the environment (rejected, counted against `max_invalid`); only the third, about 6.7 % of steps, changes the state. With $\epsilon_\text{effective} \approx 0.067$ and the heuristic's 8–12 actual steps, $(0.933)^{8..12} \approx 0.57$–$0.43$, minus the episodes that hit three invalid replies: consistent with the measurements.

### Exercise 3: budgets as a trade

1. | Setting | success | mean steps |
   |---|---:|---:|
   | 1 invalid | 0.010 | 2.7 |
   | 8 steps | 0.085 | 6.7 |
   | default | 0.150 | 8.1 |
   | 10 invalid | 0.455 | 16.0 |
   | loose | 0.580 | 27.4 |

   No setting is dominated: each higher success costs more steps. This is a frontier, and the choice needs costs.
2. Expected cost = mean steps + failure cost × (1 − success). Failure cost 50: 52.2, 52.5, 50.6, **43.3**, 48.4 → "10 invalid" is best. Failure cost 5: **7.7**, 11.3, 12.4, 18.7, 29.5 → "1 invalid" is best: when failing is cheap, give up immediately. The best budget is a property of the economics, not of the agent. (This calculation prices all failures equally; item 3 shows why that is too kind to the loose setting.)
3. The value left the environment's allowed range (beyond ±500): the agent's arbitrary moves, given 60 steps, eventually doubled it out of bounds. Under tight budgets the invalid-action limit ended those episodes earlier, before enough arbitrary moves accumulated. General lesson: budgets bound not only waste but *exposure*. Each extra step granted to an unreliable policy is another chance for an action with consequences; a loop that stops early fails cheaply, a loop that persists can fail expensively.
4. Sketch: pass a counter or the `llm` into the loop; before calling the policy, `if llm.calls >= max_calls: episode.outcome = "call_budget"; break`. Test with `SimulatedReActLLM` and `max_calls=3`: the episode has exactly three model calls and outcome `call_budget`.

### Exercise 4: context and recovery

1. Construct it with `history_limit=2` and a policy that returns illegal actions three times in a row (see `test_react_prompt_keeps_the_current_state_when_history_is_all_errors` for the corrected behaviour). Without the explicit state, the prompt contains only "invalid action" lines. The stand-in crashed because its code looked for an observation that was not there. A real model does not crash: it *guesses* a state, typically continuing from whatever the last visible lines suggest, and acts on the guess with full fluency. That is worse than a crash, because nothing signals the problem.
2. For the number game a list of visited values lets a policy avoid revisiting them, which removes circles but not the heuristic's myopia (its failures are long straight runs, not circles). For the key-and-door world the summary must carry *facts about the world and the agent*, not just places: where the key is, whether it is carried, whether the door is unlocked. A summary that keeps "visited: hall, library, cellar" and drops "carrying the key" makes the agent fetch the key again. What to keep is task-specific, which is the difficulty of memory.
3. On the stand-in the rule cannot help: its mistakes are drawn at random and do not depend on the prompt, so the measured effect is zero by construction (or noise). That is the honest statement: this experiment cannot show whether clearer error messages help a real model, only that the loop delivers them. Evidence needs a real model (Exercise 5).

### Exercise 5: a real model

No reference values (external model server; not validated by the author). Typical
findings to compare with: small instruction-tuned models produce mostly *format*
deviations (extra text after the action, quoting, capitalisation) rather than genuinely
illegal actions; normalising the reply (strip, lower-case, take the first line after
`Action:`) recovers many of them, which is a legitimate use of code as long as it never
maps an unknown action to a guessed legal one. Three failure points between reply and
action: no `Action:` line (parse failure), an action string not in the legal set, and a
legal action chosen for a state the model misread.

### Challenge: plan, execute, re-plan

Outline. Planning policy: one model call returns a list of actions; the loop executes
them, comparing each observed value with the value the plan implies; on deviation or
exhaustion it asks for a new plan from the current state. With the stand-in, a plan is
right only if *every* step of it is, so at mistake rate $\epsilon$ a $k$-step plan
survives with probability about $(1-\epsilon)^k$; re-planning recovers at the cost of
another call. Planning ahead is cheaper when steps are predictable and the model is
reliable (one call instead of $k$); it is more fragile when the environment surprises
or errors are frequent, where reactive control wastes less work per mistake.

### Written questions

1. The success test (`env.step`'s result and the `if success` branch), the budget checks, the validation `if action not in env.actions()`, and the logging. These are the lines that make the loop's guarantees hold regardless of the model; a model able to edit them could declare success, lift its own limits, or run unvalidated actions.
2. For example: the outcome distribution of the other 20 % (and whether any failures had side effects); cost per task and per successful task (steps, calls, tokens, time); success as a function of task length or difficulty; variance across repeated runs; how the success test is implemented and whether the benchmark tasks were used during development.
3. Search or rules: the state can be modelled exactly and the options enumerated (scheduling, routing, configuration checks, anything with a simulator). A model: the state is unstructured text or the criteria are fuzzy (triaging reports, drafting, deciding which document is relevant). The distinction is whether a correct, complete description of states and actions exists in code.

## Quiz

1. Observe the state; **choose an action (model)**; validate it; execute it; record the result; decide whether to stop.
2. The model can be wrong or manipulated about whether the goal is reached, and a system that takes its word reports false successes. Example: an agent "fixing" a failing test declares success after deleting the test; a code-level check (the original tests pass) would have caught it.
3. Any five with a remedy: `success`; `failed` (the environment ended the episode: wrong actions with consequences, tighten permissions/budgets, improve the policy); `step_budget` (policy too slow or going nowhere: better policy or planning); `time_budget` (too slow per step); `loop_detected` (policy cycles: memory, better state summary); `too_many_invalid` (interface problem: output format, prompt, constrained decoding).
4. $0.9^{10} = 0.349$; $0.9^{30} = 0.042$.
5. Two thirds of the stand-in's mistakes are malformed or illegal replies, which the loop rejects without changing the environment and retries; only about 7 % of steps make a wrong legal move. The loop's validation converted most errors from fatal to recoverable, so the effective per-step failure rate is far below 20 %.
6. When the environment can be simulated or modelled exactly, legal actions can be enumerated, the search space is tractable (possibly with heuristics), and the goal test is code. Then search is cheaper, optimal or near-optimal, and deterministic; a model adds cost and error.
7. Each turn the model writes a short thought, then one action; the loop executes the action and appends the observation; repeat. The thought is a scratchpad (Module 17): intermediate text that gives the model a step of computation before it commits to an action.
8. The information needed for the decision has left the context: the history was truncated or summarised and the current state (or a key fact, or the goal) is no longer visible, so the model acts on a stale or guessed state. Remedy: always include goal, legal actions and current state explicitly, and manage history by truncation or summaries that preserve task-critical facts.
9. Not simply better. Success is higher, at 3.4 times the steps, and 46 episodes now end with the environment in a failed state: the agent acted long enough to cause damage that tighter budgets prevented. Whether that trade is acceptable depends on what a `failed` episode costs in the real setting; where actions have side effects it usually is not.
10. For example: changing its own permissions or budgets; modifying the loop's code or the success test; executing arbitrary shell commands or code outside a sandbox; irreversible or externally visible actions without human approval; access to credentials or to its own logs with write permission.
11. Outcome distribution; cost per task and per success (steps, calls, tokens, time); success by task length; variance over repeated runs; efficiency against an optimal reference; qualitative trace review. Any four.
12. The loop, not the model, guarantees invariants: the stored conversation must be well-formed whatever the model produced. About context growth it does nothing: the token list only grows, nothing truncates or summarises it, and each turn prefills the whole conversation again; the model's context length is the only limit.
