"""
One interface, several "models" (Part V).

Everything in Part V talks to a language model through a single method:

    llm.complete(prompt: str, temperature=0.0, max_tokens=64, seed=None) -> str

so the orchestration code (tool loops, voting, agents, teams) is independent of what
answers. Back-ends:

    ScriptedLLM      replays fixed responses or calls a Python function. For tests and demos.
    NoisySolver      a SIMULATED arithmetic model with controllable error rates. For controlled
                     experiments about orchestration (Modules 17, 19). It is not a language model.
    OpenAICompatLLM  any server speaking the OpenAI chat-completions protocol (llama.cpp, vLLM,
                     Ollama, ...), e.g. a model you serve locally on the DGX Spark. External dependency.
    (mlle.agents.tool_engine.MiniGPTAssistant wraps a tiny trained model; a nanochat adapter
     is described in Module 16.)

Every back-end counts its calls, so that strategies can be compared at equal cost.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from typing import Callable, Protocol


class LLM(Protocol):
    calls: int

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, seed: int | None = None) -> str: ...


class ScriptedLLM:
    """Returns canned responses in order, or the result of `fn(prompt)`."""

    def __init__(self, responses: list[str] | None = None, fn: Callable[[str], str] | None = None):
        assert (responses is None) != (fn is None), "give either responses or fn"
        self.responses, self.fn = list(responses or []), fn
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, seed: int | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.fn is not None:
            return self.fn(prompt)
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


def parse_chain(expr: str) -> tuple[int, list[tuple[str, int]]]:
    """'17+5*3-4' -> (17, [('+', 5), ('*', 3), ('-', 4)]). The first number may be negative
    (a running value); operands are non-negative."""
    expr = expr.replace(" ", "")
    match = re.fullmatch(r"(-?\d+)((?:[+\-*]\d+)*)", expr)
    if match is None:
        raise ValueError(f"not an arithmetic chain: {expr!r}")
    return int(match.group(1)), [(op, int(num)) for op, num in re.findall(r"([+\-*])(\d+)", match.group(2))]


def evaluate_chain(expr: str) -> int:
    """Left-to-right evaluation of a chain like '17+5*3-4' (no precedence): the task format of Modules 17-19."""
    value, steps = parse_chain(expr)
    for op, num in steps:
        value = {"+": value + num, "-": value - num, "*": value * num}[op]
    return value


class NoisySolver:
    """A simulated model that answers arithmetic prompts, wrongly some of the time.

    Prompt protocol (plain text):
        "SOLVE: 17+5*3-4"      whole chain in one go ("direct answering")
        "STEP: 17+5"           one operation
        "CHECK: 17+5=22"       answers "yes" or "no"
        "PLAN: 17+5*3-4"       returns the steps "17+5;*3;-4"

    Error model, per call and independent unless stated:
        p_step        probability that a single operation is answered wrongly
        p_direct      probability of an error PER OPERATION when a whole chain is solved at once
                      (larger than p_step: multi-step work without intermediate results is harder)
        systematic    fraction of direct errors that are SYSTEMATIC: a deterministic function of
                      the problem, so repeating the question reproduces the same wrong answer.
                      The rest are random and differ between samples.
        p_check_err   probability that a CHECK is answered wrongly (false accept or false reject)
        p_plan        probability that a PLAN garbles one of its steps
        blind_spot    probability that a CHECK approves the solver's OWN systematic wrong answer
                      (a reviewer that shares the solver's weights shares its blind spots)

    With temperature 0 the solver is deterministic (seeded by the prompt); with temperature > 0
    random errors are re-drawn using `seed`. These knobs exist to study when voting,
    decomposition and verification help. They are assumptions, stated openly, not measurements
    of any real model."""

    def __init__(self, p_step: float = 0.05, p_direct: float = 0.15, systematic: float = 0.0, p_check_err: float = 0.1,
                 p_plan: float = 0.03, blind_spot: float = 0.9, seed: int = 0):
        self.p_step, self.p_direct, self.systematic, self.p_check_err, self.seed = p_step, p_direct, systematic, p_check_err, seed
        self.p_plan, self.blind_spot = p_plan, blind_spot
        self.calls = 0

    def _rng(self, prompt: str, temperature: float, seed: int | None) -> random.Random:
        key = f"{self.seed}|{prompt}" if temperature == 0.0 or seed is None else f"{self.seed}|{prompt}|{seed}"
        return random.Random(key)

    def _systematic_answer(self, expr: str) -> str | None:
        """The problem-specific wrong answer this solver always gives, if this problem triggers one."""
        if self.systematic <= 0:
            return None
        fixed = random.Random(f"{self.seed}|systematic|{expr}")
        n_ops = len(parse_chain(expr)[1])
        if fixed.random() < 1 - (1 - self.p_direct * self.systematic) ** n_ops:
            return str(evaluate_chain(expr) + fixed.choice([-10, -3, 3, 10]))
        return None

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, seed: int | None = None) -> str:
        self.calls += 1
        kind, _, body = prompt.partition(":")
        body = body.strip()
        rng = self._rng(prompt, temperature, seed)
        if kind == "CHECK":
            expr, _, claimed = body.partition("=")
            if self._systematic_answer(expr) == claimed and rng.random() < self.blind_spot:
                return "yes"  # the reviewer makes the same mistake the solver made
            truth = evaluate_chain(expr) == int(claimed)
            if rng.random() < self.p_check_err:
                truth = not truth
            return "yes" if truth else "no"
        if kind == "PLAN":
            first, ops = parse_chain(body)
            steps = [f"{first}{ops[0][0]}{ops[0][1]}"] + [f"{op}{num}" for op, num in ops[1:]]
            if rng.random() < self.p_plan:
                i = rng.randrange(len(steps))
                steps[i] = re.sub(r"\d+$", lambda m: str(int(m.group(0)) + 1), steps[i])  # one operand copied wrongly
            return ";".join(steps)
        truth = evaluate_chain(body)
        n_ops = len(parse_chain(body)[1])
        if kind == "STEP":
            return str(truth + rng.choice([-2, -1, 1, 2, 10])) if rng.random() < self.p_step else str(truth)
        if kind == "SOLVE":
            wrong = self._systematic_answer(body)
            if wrong is not None:
                return wrong  # the same wrong answer every time this problem is asked
            if rng.random() < 1 - (1 - self.p_direct * (1 - self.systematic)) ** n_ops:
                return str(truth + rng.choice([-10, -5, -2, -1, 1, 2, 5, 10]))
            return str(truth)
        return "?"


class OpenAICompatLLM:
    """Minimal client for an OpenAI-compatible /v1/chat/completions endpoint (standard library only).

        llm = OpenAICompatLLM("http://localhost:8000/v1", model="my-local-model")

    Use it to run the Part V labs against a real model served on your own machine. Not used
    in tests. No retries, no streaming: keep it readable."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", system: str | None = None, timeout: float = 120.0):
        self.base_url, self.model, self.api_key, self.system, self.timeout = base_url.rstrip("/"), model, api_key, system, timeout
        self.calls = 0

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, seed: int | None = None) -> str:
        self.calls += 1
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"]
