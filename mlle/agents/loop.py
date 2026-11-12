"""
The agent loop, with every control decision explicit (Module 18).

    state = goal + history of (action, observation)
    repeat:
        action = policy(state)                 <- the only place a model is involved
        if invalid: record an error observation (bounded number of times)
        observation, done, success = environment.step(action)
        stop if success | environment says done | step budget | time budget | repeated-state loop

The loop, the budgets, the stop conditions and the success test are deterministic code.
An "agent" is this program; the model is one component of it. Every step is appended to
a trace that can be printed or rendered as a diagram.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from mlle.agents.llm import LLM


class Environment(Protocol):
    goal: str

    def reset(self) -> str: ...
    def actions(self) -> list[str]: ...
    def step(self, action: str) -> tuple[str, bool, bool]: ...  # observation, done, success
    def state_key(self) -> str: ...


@dataclass
class Step:
    index: int
    thought: str
    action: str
    observation: str
    status: str  # ok | invalid | success | done


@dataclass
class Episode:
    goal: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = "running"  # success | failed | step_budget | time_budget | loop_detected | too_many_invalid
    seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.outcome == "success"


Policy = Callable[[str, list[Step], list[str]], tuple[str, str]]  # (goal, history, legal actions) -> (thought, action)


def run_agent(env: Environment, policy: Policy, max_steps: int = 20, max_seconds: float = 30.0,
              max_invalid: int = 3, loop_window: int = 3, on_step: Callable[[Step], None] | None = None) -> Episode:
    episode = Episode(goal=env.goal)
    observation = env.reset()
    episode.steps.append(Step(0, "", "(start)", observation, "ok"))
    seen: dict[str, int] = {env.state_key(): 1}
    invalid = 0
    t0 = time.perf_counter()
    for i in range(1, max_steps + 1):
        if time.perf_counter() - t0 > max_seconds:
            episode.outcome = "time_budget"
            break
        thought, action = policy(env.goal, episode.steps, env.actions())
        if action not in env.actions():
            invalid += 1
            step = Step(i, thought, action, f"invalid action. Legal actions: {', '.join(env.actions())}", "invalid")
            episode.steps.append(step)
            if on_step:
                on_step(step)
            if invalid >= max_invalid:
                episode.outcome = "too_many_invalid"
                break
            continue
        observation, done, success = env.step(action)
        step = Step(i, thought, action, observation, "success" if success else ("done" if done else "ok"))
        episode.steps.append(step)
        if on_step:
            on_step(step)
        if success:
            episode.outcome = "success"
            break
        if done:
            episode.outcome = "failed"
            break
        key = env.state_key()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > loop_window:  # the same state again and again: the agent is going in circles
            episode.outcome = "loop_detected"
            break
    else:
        episode.outcome = "step_budget"
    episode.seconds = time.perf_counter() - t0
    return episode


# ----------------------------------------------------------------------------
# A ReAct-style policy backed by an LLM

REACT_TEMPLATE = """Goal: {goal}
Legal actions: {actions}
Current state: {state}
Recent history:
{history}
Reply in exactly this format:
Thought: <one sentence>
Action: <one legal action>"""

_ACTION_RE = re.compile(r"Action:\s*(.+)")
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)")


def react_policy(llm: LLM, history_limit: int = 6) -> Policy:
    """Build a policy that shows the model the goal, the legal actions, the current state and
    the recent history, and parses 'Thought: ... / Action: ...' from its reply.

    `history_limit` is the context management: only the most recent steps are shown. The
    CURRENT STATE is passed separately and always: a truncated history can consist entirely
    of error messages, and a model that cannot see where it is cannot act (an earlier
    version of this function had exactly that bug)."""

    def policy(goal: str, history: list[Step], actions: list[str]) -> tuple[str, str]:
        shown = history[-history_limit:]
        lines = "\n".join(f"  {s.index}. action={s.action} -> {s.observation}" for s in shown)
        state = next(s.observation for s in reversed(history) if s.status != "invalid")  # last real observation
        reply = llm.complete(REACT_TEMPLATE.format(goal=goal, actions=", ".join(actions), state=state, history=lines), max_tokens=64)
        thought = (_THOUGHT_RE.search(reply) or [None, ""])[1].strip() if _THOUGHT_RE.search(reply) else ""
        match = _ACTION_RE.search(reply)
        return thought, (match.group(1).strip() if match else reply.strip())

    return policy


# ----------------------------------------------------------------------------
# Presentation

def format_episode(ep: Episode) -> str:
    lines = [f"goal: {ep.goal}"]
    for s in ep.steps:
        thought = f" | thought: {s.thought}" if s.thought else ""
        lines.append(f"  {s.index:2d}. [{s.status:7s}] {s.action:14s} -> {s.observation}{thought}")
    lines.append(f"outcome: {ep.outcome} after {len(ep.steps) - 1} steps")
    return "\n".join(lines)


def episode_to_mermaid(ep: Episode) -> str:
    """A sequence diagram of the episode, ready to paste into Markdown."""
    clean = lambda t: re.sub(r"[^\w .,=+*/()<>:-]", " ", t)[:60]
    out = ["sequenceDiagram", "    participant P as Policy (model)", "    participant L as Loop (code)", "    participant E as Environment"]
    for s in ep.steps[1:]:
        out.append(f"    L->>P: state, legal actions")
        out.append(f"    P->>L: {clean(s.action)}")
        if s.status == "invalid":
            out.append("    L-->>P: rejected (invalid action)")
        else:
            out.append(f"    L->>E: {clean(s.action)}")
            out.append(f"    E->>L: {clean(s.observation)}")
    out.append(f"    Note over L: stop: {ep.outcome}")
    return "\n".join(out)
