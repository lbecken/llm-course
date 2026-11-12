"""
Several model roles working on one task (Module 19).

Roles are just different prompts to (possibly the same) model, connected by code:

    planner    "PLAN: <problem>"   -> a list of steps
    executor   "STEP: <one step>"  -> an intermediate result
    reviewer   "CHECK: <problem>=<answer>" -> yes / no        (or an exact tool check)

They communicate through typed messages and a shared blackboard, both plain data
structures that are logged. The orchestrator (this code) decides who speaks when, how
often work is redone, and when to stop.

A team costs more model calls than a single agent, so the honest comparison is against a
single agent given the SAME number of calls (for example, self-consistency voting).
`compare_systems` produces that comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from mlle.agents.llm import LLM, evaluate_chain
from mlle.agents.reasoning import direct, self_consistency, tool_checker


@dataclass
class Message:
    sender: str
    recipient: str
    kind: str  # plan | step_result | answer | review
    content: str


@dataclass
class TeamRun:
    answer: str
    calls: int
    rounds: int
    messages: list[Message] = field(default_factory=list)
    blackboard: dict = field(default_factory=dict)


def run_team(problem: str, planner: LLM, executor: LLM, reviewer: LLM | None = None,
             tool_review: bool = False, max_rounds: int = 3) -> TeamRun:
    """Planner -> executor -> reviewer, repeated until the reviewer approves or rounds run out."""
    run = TeamRun(answer="", calls=0, rounds=0)
    board = run.blackboard
    for round_no in range(1, max_rounds + 1):
        run.rounds = round_no
        # 1. plan (re-planned each round with a different sampling seed, so a bad plan can be replaced)
        plan = planner.complete(f"PLAN: {problem}", temperature=1.0 if round_no > 1 else 0.0, seed=round_no)
        run.calls += 1
        run.messages.append(Message("planner", "executor", "plan", plan))
        board["plan"] = plan.split(";")
        # 2. execute step by step, carrying the running value on the blackboard
        value = None
        for step in board["plan"]:
            expr = step if value is None else f"{value}{step}"
            value = executor.complete(f"STEP: {expr}", temperature=1.0 if round_no > 1 else 0.0, seed=round_no)
            run.calls += 1
            run.messages.append(Message("executor", "blackboard", "step_result", f"{expr} = {value}"))
        board["answer"] = run.answer = value
        run.messages.append(Message("executor", "reviewer", "answer", value))
        # 3. review
        if tool_review:
            approved = tool_checker(problem, value)
        elif reviewer is not None:
            approved = reviewer.complete(f"CHECK: {problem}={value}") == "yes"
            run.calls += 1
        else:
            approved = True
        run.messages.append(Message("reviewer", "orchestrator", "review", "approve" if approved else "reject"))
        if approved:
            break
    return run


def compare_systems(problems: list[str], make_llm: Callable[[], LLM], max_rounds: int = 3) -> list[dict]:
    """Accuracy and cost of a single agent, a team, and cost-matched baselines, on the same problems."""

    def score(fn) -> dict:
        correct = calls = 0
        for p in problems:
            answer, used = fn(p)
            calls += used
            try:
                correct += int(answer) == evaluate_chain(p)
            except (ValueError, TypeError):
                pass
        return {"accuracy": correct / len(problems), "calls_per_problem": calls / len(problems)}

    rows = []
    llm = make_llm()
    rows.append({"system": "single agent, one call", **score(lambda p: direct(llm, p))})

    llm = make_llm()
    team = score(lambda p: (lambda r: (r.answer, r.calls))(run_team(p, llm, llm, llm, max_rounds=max_rounds)))
    rows.append({"system": "team: planner + executor + reviewer (same model)", **team})

    llm = make_llm()
    rows.append({"system": "team: planner + executor + TOOL reviewer",
                 **score(lambda p: (lambda r: (r.answer, r.calls))(run_team(p, llm, llm, None, tool_review=True, max_rounds=max_rounds)))})

    k = max(1, round(team["calls_per_problem"]))
    llm = make_llm()
    rows.append({"system": f"single agent, self-consistency with k={k} (cost-matched)", **score(lambda p: self_consistency(llm, p, k=k))})
    return rows


def plan_is_valid(problem: str, plan: str) -> bool:
    """Deterministic check that a plan reproduces the problem (cheap structural validation of a message)."""
    return re.sub(r"\s", "", "".join(plan.split(";"))) == re.sub(r"\s", "", problem)
