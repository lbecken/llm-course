# Capstone C — From LLM to tool-using agent

**Goal.** Build a bounded agent that solves a fixed suite of tasks with tools, evaluate
it honestly against simpler baselines at equal cost, and show by construction and by
test that it cannot do more than it is permitted to. An optional extension compares a
multi-agent version with the single agent.

## What you build

1. **A policy** for the task suite in [`fixtures/tasks.json`](fixtures/tasks.json): number-game tasks (reach a target with `add3`, `double`, `sub1`) and key-and-door episodes, the environments of Module 18. Your policy may be a model behind a prompt (real, through `OpenAICompatLLM`, or the simulated stand-in), a planner, or a combination. It is a Python file exposing

   ```python
   def make_policy(env):            # called once per task
       ...
       return policy                # policy(goal, history, legal_actions) -> (thought, action)
   ```

2. **At least one tool of your own** registered in a `ToolRegistry` (Module 16) with a schema and a permission class, used by your policy or by a separate assistant, with tests for a valid call, an invalid call and a refused call.
3. **An evaluation**: the harness below, run on your policy and on at least two baselines (suggested: the search policy in [`fixtures/reference_policy.py`](fixtures/reference_policy.py) as an upper bound, the heuristic `greedy_number` as a lower one), reporting success, the outcome distribution, steps and model calls.
4. **A threat model** (one page): every source of text your agent reads, every action it can take, what an injected instruction could cause, and which mechanism in code prevents each harmful case.

## The harness

```bash
python projects/capstone-c-agent/evaluate_agent.py projects/capstone-c-agent/fixtures/reference_policy.py
python projects/capstone-c-agent/evaluate_agent.py my_policy.py --out results/runs/capstone-c/eval.json
```

It runs every task with **fixed budgets that your policy cannot change** (steps, wall-clock
time per episode, invalid actions, loop detection), decides success **in the
environment's code**, and writes the outcome distribution, mean steps, the number of
model calls if your policy reports them, and every trace. It exits non-zero if the
acceptance thresholds are not met.

## Deliverables

| File | Content |
|---|---|
| your policy and tool code, with tests | in this repository or your fork |
| `eval.json` for your policy and for each baseline | written by the harness |
| `report.md` | the report (below) |

**The report** (5–8 pages): the design (which decisions are made by code and which by a
model, and why); the evaluation with baselines at equal cost; an analysis of failures by
outcome label, with at least three traces discussed; the threat model; and, if you use a
simulated model anywhere, a clear statement of what your numbers do and do not show.

**Optional extension.** A planner–executor–reviewer version of your agent, compared with
the single agent *at the same number of model calls* and with a single agent running the
same workflow (Module 19). State where independent information enters, if anywhere.

## Acceptance criteria

- [ ] `evaluate_agent.py` passes on your policy: success rate at least 0.80 on the suite, no episode ending in `failed` (the environment's failure state), every episode within budget.
- [ ] Your tool has a schema, a permission class and three tests (valid, invalid, refused).
- [ ] No part of the system executes model output as code or as a shell command; the action space is an explicit finite set or a schema-validated tool set.
- [ ] Results are compared with at least two baselines, with cost reported; simulated components are labelled.
- [ ] The threat model names a concrete attack and the line of code that stops it, and a test demonstrates it.
- [ ] Rubrics 2, 3 and 5 at least proficient on every row.

## Suggested plan

1. Run the harness on the two baselines. 2. Write your tool and its tests. 3. Write the
simplest policy that beats the heuristic. 4. Read every failing trace. 5. Write the
threat model and the attack test. 6. Only then consider the multi-agent extension.

## Starting points

`mlle/agents/loop.py`, `mlle/agents/policies.py`, `mlle/agents/tools.py`,
`mlle/agents/assistant.py`, `labs/agents/18_agent_loop.py`, `labs/agents/16_tool_assistant.py`
(part B, including the injection demonstration).
