# Module 18 — Lab: one loop, three policies, every way of stopping

Script: [`labs/agents/18_agent_loop.py`](../../labs/agents/18_agent_loop.py) ·
Code under study: [`mlle/agents/loop.py`](../../mlle/agents/loop.py), [`mlle/agents/envs.py`](../../mlle/agents/envs.py), [`mlle/agents/policies.py`](../../mlle/agents/policies.py) ·
Solutions: [`solutions/18-agentic-loop.md`](../../solutions/18-agentic-loop.md)

The lab runs in a few seconds and needs no trained model. Its "model" is a simulated
stand-in with a controllable mistake rate; Exercise 5 replaces it with a real one.

## G. Exercises

### Exercise 1 (by hand): compounding, budgets, traces

1. A policy makes a state-changing mistake in 5 % of its steps, independently, and any such mistake is fatal. What is the success probability of tasks with 5, 20 and 50 steps? Which per-step reliability is needed for 90 % success on 50-step tasks?
2. Now each step can be verified, and a failed step is retried up to twice (each try independently right with probability 0.95). What is the per-step success probability with retries, and the 50-step success probability? What assumption makes this calculation optimistic?
3. In the trace below, name the status of each step and the outcome label the loop assigns (budget: 6 steps, 2 invalid actions, loop window 2).

   ```text
   0 (start)   -> value=4
   1 double    -> value=8
   2 sub1      -> value=7
   3 add3      -> value=10
   4 fly       -> ?
   5 sub1      -> value=9
   6 sub1      -> value=8
   ```
4. An agent's success test is "the model replies DONE". Give two different ways this produces false successes.

**Success criterion:** numerical answers for items 1–2; item 3 matches `run_agent` on a scripted policy.

### Exercise 2 (experimental): outcomes and policies

```bash
python labs/agents/18_agent_loop.py
```

1. Answer the lesson's "predict before you run" question. Record tables 2 and 3.
2. The heuristic policy fails on six tasks with outcome `step_budget`. Pick one (print its trace with `format_episode`) and explain what the heuristic does wrong. Why does search not have this problem?
3. In table 2, the 20 %-mistake model fails more often through `too_many_invalid` than through `step_budget`. What does that tell you to change first: the model, the prompt, or the loop?
4. Compare measured success by plan length with $(1-\epsilon)^n$ for $\epsilon = 0.2$. Explain the gap using the three kinds of mistake the stand-in makes.

**Success criterion:** recorded tables; one explained failing trace; a quantitative answer to item 4.

### Exercise 3 (experimental): budgets as a trade

Use table 4 and your own variations (edit the list of settings).

1. Plot or tabulate success against mean steps for the five settings. Which settings are dominated (worse on both)?
2. Suppose a step costs 1 unit and a failed task costs 50. Which setting minimises expected cost per task? Recompute with a failure cost of 5.
3. In the last setting 46 episodes end as `failed`. What happened in the environment? Why did the tighter settings never produce this outcome? State the general lesson about budgets and safety.
4. Add a *cost* budget: stop when the number of model calls exceeds a limit, with outcome `call_budget`. (Ten lines in a copy of `run_agent`.) Test it.

**Success criterion:** a cost table for item 2 and working code with a test for item 4.

### Exercise 4 (coding): context and recovery

1. Reproduce the context bug described in lesson D.6: write a policy wrapper that shows the model *only* the last `k` history lines and no current state, and construct an episode in which the state is lost. What does a real model do in that situation, as opposed to the stand-in (which crashed)?
2. Implement a summarising memory: every 5 steps, replace the older history by one line `visited values: ...`. For the number game, does the summary help the heuristic avoid circles? What information would a summary need to carry for the key-and-door world?
3. Add a *recovery rule* in code: if the same invalid action is proposed twice in a row, append to the observation the list of legal actions and the sentence "choose one of these exactly". Measure the effect at a 50 % mistake rate. Why is the measured effect on this stand-in not evidence about real models?

**Success criterion:** code for items 2 and 3 with before/after numbers and an honest statement of what the numbers do and do not show.

### Exercise 5 (real model, optional): put a language model in the loop

Requires a model server with an OpenAI-compatible endpoint on your own machine (for
example llama.cpp or vLLM serving any small instruction-tuned model; on the DGX Spark
this fits easily). **This uses external software and is not validated by the author.**

```python
from mlle.agents.llm import OpenAICompatLLM
from mlle.agents.loop import react_policy, run_agent, format_episode
from mlle.agents.envs import KeyDoorWorld
llm = OpenAICompatLLM("http://localhost:8080/v1", model="local")
print(format_episode(run_agent(KeyDoorWorld(), react_policy(llm), max_steps=15)))
```

1. Run 20 episodes of each environment. Record the outcome distribution and mean steps.
2. Collect the invalid actions. Are they format errors, illegal actions, or something else? Change the prompt template once, and re-measure.
3. Estimate the per-step error rate of your model from the traces and compare its task success with table 3.

On a machine without a model server: read `react_policy` and `OpenAICompatLLM`, and write
down which three things can go wrong between the model's reply and a valid action.

**Success criterion:** an outcome table for a real model, or the written analysis.

### Challenge (optional): plan, execute, re-plan

Implement a policy for the number game in which the (simulated or real) model first
writes a complete plan as a list of actions, the loop executes it step by step, and on
the first deviation from the expected value the model is asked for a new plan from the
current state. Compare success and model calls with the reactive policy at 20 % and
50 % mistakes. When is planning ahead cheaper, and when is it more fragile?

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Make every control decision of an agent loop observable; measure how policy quality, task length and budgets determine outcomes |
| **Hypotheses** | (H1) Search solves all tasks in about half the steps of the heuristic. (H2) A flawless model behind the heuristic matches the heuristic exactly, at one call per step. (H3) Success decreases with plan length for imperfect policies, faster for higher mistake rates. (H4) Measured success exceeds $(1-\epsilon)^n$ because invalid actions are rejected and retried. (H5) Raising the invalid-action budget from 3 to 10 at least doubles success at 50 % mistakes, at a cost in steps. (H6) Very loose budgets produce a failure type (environment failure) absent under tight budgets. |
| **Hardware / software** | CPU; course package; no model |
| **Tier** | A: about 5 seconds |
| **Metrics** | success rate; outcome distribution; mean steps; model calls; success by optimal plan length |
| **Independent variables** | policy; mistake rate; budgets (steps, invalid actions, loop window) |
| **Controls** | the same 200 tasks for every policy and setting; seeded stand-in (seed = task index) |
| **Interpretation** | The stand-in's mistakes are independent across steps and of three fixed kinds; real models' errors are correlated and context-dependent. Treat the numbers as an illustration of the *loop's* behaviour. Cells with few tasks (8-move plans: 7 tasks) are noisy. |
| **Common failure modes** | Reporting success without the outcome distribution. Letting the model decide success. Measuring only short tasks. Tuning budgets on the evaluation tasks. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.3–D.5.

| Policy | success | mean steps | calls | outcomes | your prediction of success |
|---|---:|---:|---:|---|---|
| search | | | | | |
| heuristic | | | | | |
| model, 0 % | | | | | |
| model, 20 % | | | | | |
| model, 50 % | | | | | |

| Setting (50 % mistakes) | success | mean steps | outcomes |
|---|---:|---:|---|
| default | | | |
| 1 invalid | | | |
| 10 invalid | | | |
| 60 steps, loose | | | |

**Questions to answer in writing**

1. Which lines of `run_agent` would you be most reluctant to let a model write or modify at run time, and why?
2. A vendor reports "our agent completes 80 % of tasks". Which four further numbers do you ask for?
3. Describe a task from your own work for which the right "policy" is search or a rule, and one for which it is a model. What distinguishes them?
