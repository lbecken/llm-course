"""
Inference-time strategies: spending more model calls on one question (Module 17).

All four strategies wrap the SAME model; none changes its weights.

    direct            one call, take the answer
    decompose         split the problem into single steps, one call per step, carry the result
    self_consistency  sample k answers, return the most common (majority vote)
    verify_and_retry  propose an answer, check it, re-sample if the check fails

Each returns (answer, calls_used). The task format is a left-to-right arithmetic chain
("17+5*3-4"), chosen because problems of any length can be generated and checked exactly.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

from mlle.agents.llm import LLM, evaluate_chain, parse_chain


def make_chain_problems(n: int, num_ops: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        expr = str(rng.randint(2, 30))
        for _ in range(num_ops):
            op = rng.choice("+-*")
            expr += op + str(rng.randint(2, 9) if op == "*" else rng.randint(2, 30))
        problems.append(expr)
    return problems


def split_chain(expr: str) -> list[tuple[str, str]]:
    """'17+5*3' -> [('+', '5'), ('*', '3')]: the operations that follow the first number."""
    return [(op, str(num)) for op, num in parse_chain(expr)[1]]


def direct(llm: LLM, expr: str, **_) -> tuple[str, int]:
    return llm.complete(f"SOLVE: {expr}"), 1


def decompose(llm: LLM, expr: str, **_) -> tuple[str, int]:
    """The ORCHESTRATOR does the planning (splitting into steps); the model does each step."""
    value = str(parse_chain(expr)[0])
    steps = split_chain(expr)
    for op, operand in steps:
        value = llm.complete(f"STEP: {value}{op}{operand}")
    return value, len(steps)


def self_consistency(llm: LLM, expr: str, k: int = 5, **_) -> tuple[str, int]:
    answers = [llm.complete(f"SOLVE: {expr}", temperature=1.0, seed=i) for i in range(k)]
    return Counter(answers).most_common(1)[0][0], k


def llm_checker(llm: LLM) -> Callable[[str, str], bool]:
    """Verification by asking the model itself (fallible, costs a call)."""
    return lambda expr, answer: llm.complete(f"CHECK: {expr}={answer}") == "yes"


def tool_checker(expr: str, answer: str) -> bool:
    """Verification by an exact tool. Possible here because arithmetic is cheap to check;
    for most real tasks no such oracle exists, and that is the hard part."""
    try:
        return evaluate_chain(expr) == int(answer)
    except ValueError:
        return False


def verify_and_retry(llm: LLM, expr: str, checker: Callable[[str, str], bool], max_attempts: int = 4, check_cost: int = 0, **_) -> tuple[str, int]:
    """Sample an answer; accept it if the checker approves; otherwise sample again."""
    calls, answer = 0, ""
    for attempt in range(max_attempts):
        answer = llm.complete(f"SOLVE: {expr}", temperature=1.0, seed=attempt)
        calls += 1 + check_cost
        if checker(expr, answer):
            break
    return answer, calls


def tool_only(llm: LLM, expr: str, **_) -> tuple[str, int]:
    """Reference point: hand the whole computation to an exact tool. Zero model calls, always right.
    When such a tool exists, the best "reasoning strategy" is not to ask the model to compute."""
    return str(evaluate_chain(expr)), 0


def run_benchmark(strategy: Callable, llm: LLM, problems: list[str], **kwargs) -> dict:
    correct = calls = 0
    for expr in problems:
        answer, used = strategy(llm, expr, **kwargs)
        calls += used
        try:
            correct += int(answer) == evaluate_chain(expr)
        except ValueError:
            pass
    return {"accuracy": correct / len(problems), "calls_per_problem": calls / len(problems)}
