"""
Lab 19 — Is a team of agents better than one agent with the same budget?
Tier A (a few seconds, CPU)

    python labs/agents/19_multi_agent.py

A "team" is several ROLES (prompts) connected by code: a planner splits the problem, an
executor does one step at a time, a reviewer approves or rejects, and the orchestrator
repeats the round if the review fails. All roles here are played by the SIMULATED solver
of Module 17 (stated error rates; not a language model), so that the questions about
ORGANISATION can be answered exactly:

  1  What do the messages of one run look like?
  2  Team versus a single call, and versus a single agent given the SAME number of calls.
  3  What is the reviewer worth: same model, an independent model, an exact tool, none?
  4  What happens when errors are systematic and the reviewer shares the blind spot?
  5  Does validating messages between roles (in code) pay?

Predict before running: the team uses about 6 calls per problem. Will it beat a single
agent that spends 6 calls on majority voting?
"""

from mlle.agents.llm import NoisySolver, evaluate_chain
from mlle.agents.multi import compare_systems, plan_is_valid, run_team
from mlle.agents.reasoning import make_chain_problems
from mlle.utils import REPO_ROOT, JsonlLogger

log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part5" / "19_multi_agent.jsonl")
N = 400

# %% 1. One run, message by message
print("1. one team run (planner -> executor -> reviewer), every message logged")
llm = NoisySolver(p_step=0.3, seed=4)  # an unreliable executor: watch what the reviewer does with its answer
problem = make_chain_problems(1, 4, seed=5)[0]
run = run_team(problem, llm, llm, llm, max_rounds=3)
print(f"   problem: {problem}   (truth: {evaluate_chain(problem)})")
for m in run.messages:
    print(f"   {m.sender:9s} -> {m.recipient:12s} [{m.kind:11s}] {m.content}")
print(f"   answer {run.answer} after {run.rounds} round(s), {run.calls} model calls")

# %% 2. Team against cost-matched single agents
print(f"\n2. accuracy and cost on {N} problems (simulated solver: 5 % error per step, 15 % per operation when solving at once, 10 % wrong reviews)")
for n_ops in (4, 8):
    print(f"   {n_ops} operations")
    for row in compare_systems(make_chain_problems(N, n_ops, seed=1), lambda: NoisySolver(seed=0)):
        print(f"      {row['system']:58s} accuracy {row['accuracy']:.3f}   calls/problem {row['calls_per_problem']:.1f}")
        log.log(part=2, n_ops=n_ops, **row)


def team_score(problems, planner, executor, reviewer=None, tool_review=False, validate_plan=False, max_rounds=3):
    correct = calls = 0
    for p in problems:
        if validate_plan:  # the orchestrator checks the planner's message before anyone acts on it
            planner_checked = _ValidatingPlanner(planner, p)
            r = run_team(p, planner_checked, executor, reviewer, tool_review=tool_review, max_rounds=max_rounds)
            r.calls += planner_checked.extra_calls
        else:
            r = run_team(p, planner, executor, reviewer, tool_review=tool_review, max_rounds=max_rounds)
        calls += r.calls
        correct += str(r.answer) == str(evaluate_chain(p))
    return correct / len(problems), calls / len(problems)


class _ValidatingPlanner:
    """Wraps a planner: re-asks (up to 3 times) until the plan reproduces the problem."""

    def __init__(self, planner, problem):
        self.planner, self.problem, self.extra_calls = planner, problem, 0

    def complete(self, prompt, temperature=0.0, max_tokens=64, seed=None):
        plan = self.planner.complete(prompt, temperature=temperature, seed=seed)
        for retry in range(3):
            if plan_is_valid(self.problem, plan):
                break
            plan = self.planner.complete(prompt, temperature=1.0, seed=1000 + retry)
            self.extra_calls += 1
        return plan


problems = make_chain_problems(N, 6, seed=2)

# %% 3. What is the reviewer worth?
print("\n3. the reviewer (6 operations, up to 3 rounds)")
print(f"   {'reviewer':52s} {'accuracy':>9} {'calls':>6}")
solver = lambda **kw: NoisySolver(seed=0, **kw)  # noqa: E731
rows = [("none (accept the first answer)", dict(reviewer=None)),
        ("the same model (10 % wrong reviews)", dict(reviewer=solver())),
        ("the same model, careless (30 % wrong reviews)", dict(reviewer=solver(p_check_err=0.3))),
        ("an exact tool", dict(tool_review=True))]
for name, kw in rows:
    acc, calls = team_score(problems, solver(), solver(), **kw)
    print(f"   {name:52s} {acc:>9.3f} {calls:>6.1f}")
    log.log(part=3, reviewer=name, accuracy=acc, calls=calls)

# %% 4. Shared blind spots
print("\n4. a solver whose errors are half SYSTEMATIC answers the whole problem; a reviewer approves or sends it back (up to 3 attempts)")
print(f"   {'reviewer':52s} {'accuracy':>9} {'calls':>6} {'wrong answers approved':>23}")


def solve_and_review(reviewer_fn, check_cost):
    correct = calls = approved_wrong = 0
    worker = NoisySolver(systematic=0.5, seed=0)
    for p in problems:
        for attempt in range(3):
            answer = worker.complete(f"SOLVE: {p}", temperature=1.0, seed=attempt)
            calls += 1 + check_cost
            if reviewer_fn(p, answer):
                break
        right = answer == str(evaluate_chain(p))
        correct += right
        approved_wrong += (not right) and reviewer_fn(p, answer)
    return correct / len(problems), calls / len(problems), approved_wrong / len(problems)


same = NoisySolver(systematic=0.5, seed=0)  # the reviewer IS the solver: same weights, same blind spots
other = NoisySolver(systematic=0.5, seed=1, blind_spot=0.0)  # a different model: its own errors, not these
for name, fn, cost in [("the same model as the solver", lambda p, a: same.complete(f"CHECK: {p}={a}") == "yes", 1),
                       ("an independent model", lambda p, a: other.complete(f"CHECK: {p}={a}") == "yes", 1),
                       ("an exact tool", lambda p, a: a == str(evaluate_chain(p)), 0)]:
    acc, calls, bad = solve_and_review(fn, cost)
    print(f"   {name:52s} {acc:>9.3f} {calls:>6.1f} {bad:>23.3f}")
    log.log(part=4, reviewer=name, accuracy=acc, calls=calls, approved_wrong=bad)

# %% 5. Validating messages between roles
print("\n5. a sloppy planner (garbles one step in 20 % of plans); tool reviewer; with and without a code check of the plan message")
for name, validate in (("plan used as received", False), ("plan validated by code, re-requested if wrong", True)):
    acc, calls = team_score(problems, NoisySolver(p_plan=0.2, seed=0), NoisySolver(seed=0), tool_review=True, validate_plan=validate)
    print(f"   {name:52s} accuracy {acc:.3f}   calls/problem {calls:.1f}")
    log.log(part=5, variant=name, accuracy=acc, calls=calls)
print("\nRoles do not create information. What helps is an INDEPENDENT source of it: a tool, a different model, a check in code.")
