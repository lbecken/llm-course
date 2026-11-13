"""
Evaluation harness for Capstone C.

    python projects/capstone-c-agent/evaluate_agent.py path/to/my_policy.py [--out eval.json] [--tasks tasks.json]

Loads `make_policy(env)` from the given file and runs it on the fixed task suite with
FIXED budgets (the policy cannot change them), deciding success in the environment's
code. Prints the outcome distribution and writes every trace. Exit status 0 only if the
acceptance thresholds are met.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from mlle.agents.envs import KeyDoorWorld, NumberGame
from mlle.agents.loop import run_agent

HERE = Path(__file__).resolve().parent


def load_policy_factory(path: str | Path):
    spec = importlib.util.spec_from_file_location("submitted_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the learner's own file, run on the learner's own machine
    if not callable(getattr(module, "make_policy", None)):
        raise SystemExit(f"{path} must define make_policy(env)")
    return module.make_policy


def evaluate(make_policy, tasks: dict) -> dict:
    budgets = tasks["budgets"]
    envs = [("number", NumberGame(s, t)) for s, t in tasks["number_game"]] + [("key_door", KeyDoorWorld()) for _ in range(tasks["key_door_episodes"])]
    episodes, outcomes, steps = [], Counter(), 0
    for kind, env in envs:
        policy = make_policy(env)
        ep = run_agent(env, policy, **budgets)
        within = len(ep.steps) - 1 <= budgets["max_steps"]
        outcomes[ep.outcome] += 1
        steps += len(ep.steps) - 1
        episodes.append({"kind": kind, "goal": ep.goal, "outcome": ep.outcome, "steps": len(ep.steps) - 1, "within_budget": within,
                         "trace": [{"i": s.index, "thought": s.thought, "action": s.action, "observation": s.observation, "status": s.status} for s in ep.steps]})
    n = len(envs)
    th = tasks["thresholds"]
    result = {"episodes": n, "success_rate": outcomes["success"] / n, "outcomes": dict(outcomes), "mean_steps": steps / n,
              "all_within_budget": all(e["within_budget"] for e in episodes), "budgets": budgets, "thresholds": th, "traces": episodes}
    result["accepted"] = (result["success_rate"] >= th["min_success_rate"] and outcomes["failed"] <= th["max_failed_episodes"]
                          and result["all_within_budget"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_file")
    ap.add_argument("--tasks", default=str(HERE / "fixtures" / "tasks.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text())
    result = evaluate(load_policy_factory(args.policy_file), tasks)
    print(f"episodes {result['episodes']} | success rate {result['success_rate']:.3f} | mean steps {result['mean_steps']:.1f} | outcomes {result['outcomes']}")
    print(f"budgets (fixed by the harness): {result['budgets']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"traces written to {args.out}")
    print("ACCEPTED (mechanical criteria); now apply rubrics 2, 3 and 5" if result["accepted"] else
          f"NOT ACCEPTED: need success >= {result['thresholds']['min_success_rate']}, no 'failed' episodes, all within budget")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
