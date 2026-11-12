"""
Lab 18 — The agent loop: a program with a model inside.   Tier A (a few seconds, CPU)

    python labs/agents/18_agent_loop.py

An agent is a LOOP: observe, let a policy choose an action, validate it, execute it,
record what happened, decide whether to stop. Only "choose an action" involves a model.
This lab runs the same loop (mlle.agents.loop.run_agent) with three kinds of policy:

    search     plan_then_execute: breadth-first search finds the optimal plan (no model)
    heuristic  a myopic rule written in code (no model)
    model      a SIMULATED model behind a ReAct prompt, with a controllable rate of mistakes
               (malformed replies, illegal actions, arbitrary choices). It is a stand-in, not
               a language model: the subject here is the loop, and a stand-in lets every
               stop condition be triggered on demand.

on two environments whose success test is code: a number game and a key-and-door world.
Reported: success rate, steps, model calls, and WHY each episode ended.
Predict before running: if the model makes a mistake in 20 % of its steps, what fraction
of 6-step tasks will it complete? Does the loop change that number?
"""

import random
from collections import Counter

from mlle.agents.envs import KeyDoorWorld, NumberGame
from mlle.agents.loop import episode_to_mermaid, format_episode, react_policy, run_agent
from mlle.agents.policies import SimulatedReActLLM, greedy_number, plan_then_execute
from mlle.utils import REPO_ROOT, JsonlLogger

log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part5" / "18_agent_loop.jsonl")

# %% 1. One episode, step by step
print("1. one episode in the key-and-door world (simulated model, no mistakes)")
episode = run_agent(KeyDoorWorld(), react_policy(SimulatedReActLLM(0.0)), max_steps=12)
print(format_episode(episode))

print("\n   the same world, the model wrong in half of its steps (seed 3): the loop validates, counts and stops")
episode = run_agent(KeyDoorWorld(), react_policy(SimulatedReActLLM(0.5, seed=3)), max_steps=12)
print(format_episode(episode))

# %% 2. Policies compared on many tasks
rng = random.Random(0)
tasks = []
while len(tasks) < 200:
    start, target = rng.randint(1, 20), rng.randint(21, 120)
    plan = NumberGame(start, target).shortest_solution()
    if plan is not None and 3 <= len(plan) <= 9:
        tasks.append((start, target, len(plan)))


def evaluate(make_policy, max_steps=20, **loop_kw):
    outcomes, steps, calls, by_len = Counter(), [], 0, {}
    for i, (start, target, optimal) in enumerate(tasks):
        env = NumberGame(start, target)
        policy, llm = make_policy(env, i)
        ep = run_agent(env, policy, max_steps=max_steps, **loop_kw)
        outcomes[ep.outcome] += 1
        steps.append(len(ep.steps) - 1)
        calls += llm.calls if llm is not None else 0
        ok, n = by_len.get(optimal, (0, 0))
        by_len[optimal] = (ok + ep.success, n + 1)
    return outcomes, sum(steps) / len(steps), calls / len(tasks), by_len


def model_policy(rate):
    def make(env, i):
        llm = SimulatedReActLLM(rate, seed=i)
        return react_policy(llm), llm
    return make


policies = [("search (optimal plan, no model)", lambda env, i: (plan_then_execute(env), None)),
            ("heuristic in code (no model)", lambda env, i: (greedy_number, None)),
            ("simulated model, 0 % mistakes", model_policy(0.0)),
            ("simulated model, 20 % mistakes", model_policy(0.2)),
            ("simulated model, 50 % mistakes", model_policy(0.5))]
print(f"\n2. number game, {len(tasks)} tasks solvable in 3-9 moves; budget 20 steps, 3 invalid actions, loop detection on")
print(f"{'policy':34s} {'success':>8} {'mean steps':>11} {'model calls':>12}   why episodes ended")
results = {}
for name, make in policies:
    outcomes, mean_steps, calls, by_len = evaluate(make)
    results[name] = by_len
    ended = ", ".join(f"{k} {v}" for k, v in outcomes.most_common())
    print(f"{name:34s} {outcomes['success'] / len(tasks):>8.3f} {mean_steps:>11.1f} {calls:>12.1f}   {ended}")
    log.log(part=2, policy=name, success=outcomes["success"] / len(tasks), mean_steps=mean_steps, calls=calls, outcomes=dict(outcomes))

# %% 3. Success against task length
print("\n3. success rate by length of the optimal plan")
lengths = sorted({n for _, _, n in tasks})
print(f"{'policy':34s} " + " ".join(f"{str(n) + ' moves':>8}" for n in lengths))
for name, by_len in results.items():
    print(f"{name:34s} " + " ".join(f"{by_len[n][0] / by_len[n][1]:>8.2f}" for n in lengths))
print(f"{'(tasks of that length)':34s} " + " ".join(f"{results[policies[0][0]][n][1]:>8d}" for n in lengths))

# %% 4. What the guards are worth
print("\n4. the 50 %-mistake model under different loop settings")
print(f"{'setting':46s} {'success':>8} {'mean steps':>11}   why episodes ended")
for label, kw in [("default: 20 steps, 3 invalid, loop window 3", {}),
                  ("no tolerance: 1 invalid action ends the episode", dict(max_invalid=1)),
                  ("tolerant: 10 invalid actions", dict(max_invalid=10)),
                  ("small budget: 8 steps", dict(max_steps=8)),
                  ("large budget: 60 steps, 20 invalid, loop window 50", dict(max_steps=60, max_invalid=20, loop_window=50))]:
    outcomes, mean_steps, calls, _ = evaluate(model_policy(0.5), **kw)
    print(f"{label:46s} {outcomes['success'] / len(tasks):>8.3f} {mean_steps:>11.1f}   " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    log.log(part=4, setting=label, success=outcomes["success"] / len(tasks), mean_steps=mean_steps, outcomes=dict(outcomes))

# %% 5. A trace as a diagram
print("\n5. Mermaid sequence diagram of one short episode (paste into any Markdown file):\n")
print(episode_to_mermaid(run_agent(NumberGame(5, 26), react_policy(SimulatedReActLLM(0.3, seed=1)), max_steps=10)))
