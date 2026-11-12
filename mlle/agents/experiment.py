"""
A bounded, auditable experiment loop (Module 20).

    repeat at most `max_trials` times, within `max_seconds`:
        1. PROPOSE exactly one change to the current configuration
        2. TEST    reject the proposal if it violates a guard rail (bounds, budget)
        3. RUN     train and evaluate on the VALIDATION set
        4. DECIDE  accept only if it beats the incumbent by more than the noise threshold
        5. RECORD  append everything to a ledger (JSON lines), accepted or not
    finally: evaluate the chosen configuration ONCE on a held-out test set

What makes this safe and honest is everything the loop does NOT do: it cannot change its
own code, its search space, its budget or its evaluation; it never sees the test set; and
every decision is on record. It improves a configuration (a workflow artefact). It does
not improve "itself".
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

Config = dict
Objective = Callable[[Config, int], dict]  # (config, seed) -> {"val": float, "test": float}; higher is better
Proposer = Callable[[Config, list, random.Random], tuple[str, object]]  # -> (parameter, new value)


@dataclass
class Trial:
    index: int
    parameter: str
    old: object
    new: object
    passed_tests: bool
    val_score: float | None
    incumbent_val: float
    accepted: bool
    seconds: float
    note: str = ""


@dataclass
class LoopResult:
    best_config: Config
    best_val: float
    test_score: float
    trials: list[Trial] = field(default_factory=list)
    stopped: str = "max_trials"
    noise: float = 0.0


def random_proposer(space: dict[str, list]) -> Proposer:
    """Change one randomly chosen parameter to a different allowed value."""

    def propose(config: Config, history: list, rng: random.Random):
        name = rng.choice(sorted(space))
        return name, rng.choice([v for v in space[name] if v != config[name]])

    return propose


def llm_proposer(llm, space: dict[str, list]) -> Proposer:
    """Ask a model for 'CHANGE: name=value'. Whatever it replies is validated against the
    search space by the loop's tests: a model cannot propose its way out of the sandbox."""

    def propose(config: Config, history: list, rng: random.Random):
        recent = "; ".join(f"{t.parameter}={t.new} -> {'kept' if t.accepted else 'rejected'}" for t in history[-5:])
        reply = llm.complete(f"Current config: {config}\nAllowed values: {space}\nRecent trials: {recent}\nReply with: CHANGE: name=value")
        name, _, value = reply.replace("CHANGE:", "").strip().partition("=")
        name, value = name.strip(), value.strip()
        for candidate in space.get(name, []):
            if str(candidate) == value:
                return name, candidate
        return name, value  # left as-is; the guard rails will reject it

    return propose


def estimate_noise(objective: Objective, config: Config, seeds=(101, 102, 103)) -> float:
    """Spread of the validation score across seeds for one configuration: differences smaller
    than this are not evidence of improvement."""
    scores = [objective(config, s)["val"] for s in seeds]
    return max(scores) - min(scores)


def run_experiment_loop(objective: Objective, config: Config, space: dict[str, list], proposer: Proposer,
                        max_trials: int = 10, max_seconds: float = 120.0, min_improvement: float | None = None,
                        ledger_path: str | Path | None = None, seed: int = 0) -> LoopResult:
    rng = random.Random(seed)
    t_start = time.perf_counter()
    noise = estimate_noise(objective, config) if min_improvement is None else min_improvement
    incumbent = dict(config)
    incumbent_val = objective(incumbent, seed)["val"]
    result = LoopResult(best_config=incumbent, best_val=incumbent_val, test_score=float("nan"), noise=noise)

    def record(trial: Trial):
        result.trials.append(trial)
        if ledger_path is not None:
            Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**asdict(trial), "config_after": dict(incumbent)}) + "\n")

    for i in range(1, max_trials + 1):
        if time.perf_counter() - t_start > max_seconds:
            result.stopped = "time_budget"
            break
        t0 = time.perf_counter()
        name, new = proposer(incumbent, result.trials, rng)
        old = incumbent.get(name)
        # TEST: guard rails. Only known parameters, only allowed values, exactly one change.
        if name not in space or new not in space[name] or new == old:
            record(Trial(i, name, old, new, False, None, incumbent_val, False, time.perf_counter() - t0, "rejected by guard rails"))
            continue
        candidate = {**incumbent, name: new}
        val = objective(candidate, seed)["val"]
        accepted = val > incumbent_val + noise
        note = "improvement exceeds noise" if accepted else "no improvement beyond noise"
        if accepted:
            incumbent, incumbent_val = candidate, val
        record(Trial(i, name, old, new, True, val, incumbent_val if not accepted else val, accepted, time.perf_counter() - t0, note))

    result.best_config, result.best_val = incumbent, incumbent_val
    result.test_score = objective(incumbent, seed + 9999)["test"]  # the ONLY use of the test set, with a fresh seed
    return result


def format_ledger(result: LoopResult) -> str:
    lines = [f"noise threshold: {result.noise:.4f}"]
    for t in result.trials:
        val = "   -   " if t.val_score is None else f"{t.val_score:.4f}"
        lines.append(f"  trial {t.index:2d}: {t.parameter}: {t.old} -> {t.new} | val {val} | {'ACCEPT' if t.accepted else 'reject'} ({t.note})")
    lines.append(f"final: {result.best_config} | val {result.best_val:.4f} | test (once) {result.test_score:.4f} | stopped: {result.stopped}")
    return "\n".join(lines)
