"""
Policies for the agent loop (Module 18): from pure code to simulated models.

    plan_then_execute   compute a full plan by search, then follow it (no model at all)
    greedy_number       a myopic heuristic that sometimes walks in circles
    SimulatedReActLLM   a rule-based stand-in for a model: it reads the ReAct prompt, applies a
                        simple heuristic, and with a given probability makes a mistake (a poor
                        or illegal action, or a malformed reply). It lets the loop's parsing,
                        validation, recovery and stop conditions be exercised deterministically.

To use a real model instead, pass any `LLM` to `mlle.agents.loop.react_policy`.
"""

from __future__ import annotations

import random
import re

from mlle.agents.envs import NumberGame
from mlle.agents.loop import Policy, Step


def plan_then_execute(env: NumberGame) -> Policy:
    plan = env.shortest_solution() or []

    def policy(goal: str, history: list[Step], actions: list[str]):
        done = sum(1 for s in history[1:] if s.status != "invalid")
        return ("following the precomputed plan", plan[done]) if done < len(plan) else ("plan exhausted", actions[0])

    return policy


def greedy_number(goal: str, history: list[Step], actions: list[str]):
    """Myopic rule: double while far below the target, add 3 while below, otherwise subtract 1."""
    target = int(re.search(r"into (-?\d+)", goal).group(1))
    value = int(re.search(r"value=(-?\d+)", history[-1].observation if history[-1].status != "invalid" else history[-2].observation).group(1))
    if value * 2 <= target:
        return "far below: double", "double"
    if value + 3 <= target:
        return "below: add 3", "add3"
    if value < target:
        return "just below: add 3 then come back down", "add3"
    return "above: subtract 1", "sub1"


class SimulatedReActLLM:
    """Stands in for a model inside `react_policy`. NOT a language model."""

    def __init__(self, mistake_rate: float = 0.0, seed: int = 0):
        self.mistake_rate, self.rng, self.calls = mistake_rate, random.Random(seed), 0

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, seed: int | None = None) -> str:
        self.calls += 1
        goal = re.search(r"Goal: (.*)", prompt).group(1)
        actions = [a.strip() for a in re.search(r"Legal actions: (.*)", prompt).group(1).split(",")]
        last_obs = re.search(r"Current state: (.*)", prompt).group(1)
        roll = self.rng.random()
        if roll < self.mistake_rate / 3:
            return "I think we should try something else."  # malformed: no Action line
        if roll < 2 * self.mistake_rate / 3:
            return "Thought: let me try a shortcut.\nAction: teleport"  # illegal action
        if roll < self.mistake_rate:
            return f"Thought: guessing.\nAction: {self.rng.choice(actions)}"  # legal but arbitrary
        thought, action = self._heuristic(goal, last_obs, actions)
        return f"Thought: {thought}\nAction: {action}"

    @staticmethod
    def _heuristic(goal: str, obs: str, actions: list[str]) -> tuple[str, str]:
        if "value=" in obs:  # NumberGame
            target = int(re.search(r"into (-?\d+)", goal).group(1))
            value = int(re.search(r"value=(-?\d+)", obs).group(1))
            if value * 2 <= target:
                return "far below the target, doubling is fastest", "double"
            if value < target:
                return "below the target, adding 3", "add3"
            return "above the target, subtracting 1", "sub1"
        if "key lies here" in obs:
            return "there is a key, take it", "take key"
        if "you carry a key" in obs and "locked" in obs:
            return "I have the key and the door is locked", "unlock door"
        return "keep exploring east", "go east"
