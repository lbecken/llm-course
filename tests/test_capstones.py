"""The capstone checkers must accept their fixtures and reject broken submissions."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from mlle.agents.envs import NumberGame
from mlle.utils import REPO_ROOT

PROJECTS = REPO_ROOT / "projects"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- Capstone A
@pytest.fixture()
def submission_a(tmp_path):
    dst = tmp_path / "submission"
    shutil.copytree(PROJECTS / "capstone-a-small-gpt" / "fixtures" / "example_submission", dst)
    return dst


def test_capstone_a_fixture_passes(submission_a):
    checker = load(PROJECTS / "capstone-a-small-gpt" / "check_submission.py")
    assert checker.check(submission_a) == []


@pytest.mark.parametrize("mutation, expected", [
    (lambda r: r.update(final_val_loss=0.30, val_loss_per_seed=[0.30, 0.30, 0.30]), "BELOW the stated entropy floor"),
    (lambda r: r.update(final_val_loss=0.50, val_loss_per_seed=[0.50, 0.51, 0.49]), "more than 15 %"),
    (lambda r: r.update(seeds=[0], val_loss_per_seed=[0.3481]), "at least three seeds"),
    (lambda r: r.update(ablation={"name": "x", "value": 1.0}), "prediction"),
])
def test_capstone_a_rejects_bad_results(submission_a, mutation, expected):
    checker = load(PROJECTS / "capstone-a-small-gpt" / "check_submission.py")
    results = json.loads((submission_a / "results.json").read_text())
    mutation(results)
    (submission_a / "results.json").write_text(json.dumps(results))
    assert any(expected in p for p in checker.check(submission_a))


def test_capstone_a_rejects_missing_files_and_degenerate_samples(submission_a):
    checker = load(PROJECTS / "capstone-a-small-gpt" / "check_submission.py")
    (submission_a / "samples.txt").write_text("the cat sleeps .\n" * 30)
    assert any("duplicates" in p for p in checker.check(submission_a))
    (submission_a / "report.md").unlink()
    assert checker.check(submission_a) == ["missing file: report.md"]


# ----------------------------------------------------------------------------- Capstone B
def test_capstone_b_fixture_passes_and_bad_rows_fail(tmp_path):
    checker = load(PROJECTS / "capstone-b-inference" / "check_benchmark.py")
    fixture = PROJECTS / "capstone-b-inference" / "fixtures" / "example_benchmarks.jsonl"
    assert checker.check(fixture) == []
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    timing = next(r for r in rows if r["unit"] == "s")

    def problems(**changes):
        bad = {**timing, **changes}
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(bad) + "\n")
        return checker.check(path)

    assert any("three repetitions" in p for p in problems(baseline_runs=[1.0], optimised_runs=[0.5]))
    assert any("correctness" in p for p in problems(correctness=""))
    assert any("ok is not true" in p for p in problems(ok=False))
    assert any("ratio" in p for p in problems(ratio=timing["ratio"] * 2))
    # a "speed-up" whose runs overlap is not a speed-up
    overlap = problems(baseline=1.05, optimised=1.0, ratio=1.05, baseline_runs=[1.0, 1.05, 1.1], optimised_runs=[0.95, 1.0, 1.06])
    assert any("not faster than the fastest baseline run" in p or "run-to-run variation" in p for p in overlap)
    missing = dict(timing)
    del missing["conditions"]
    path = tmp_path / "missing.jsonl"
    path.write_text(json.dumps(missing) + "\n")
    assert any("missing field 'conditions'" in p for p in checker.check(path))


# ----------------------------------------------------------------------------- Capstone C
@pytest.fixture(scope="module")
def harness():
    return load(PROJECTS / "capstone-c-agent" / "evaluate_agent.py")


@pytest.fixture(scope="module")
def tasks():
    return json.loads((PROJECTS / "capstone-c-agent" / "fixtures" / "tasks.json").read_text())


def test_capstone_c_tasks_are_solvable_within_budget(tasks):
    lengths = [len(NumberGame(s, t).shortest_solution()) for s, t in tasks["number_game"]]
    assert min(lengths) >= 2 and max(lengths) <= 7 < tasks["budgets"]["max_steps"]


def test_capstone_c_reference_policy_is_accepted(harness, tasks):
    make_policy = harness.load_policy_factory(PROJECTS / "capstone-c-agent" / "fixtures" / "reference_policy.py")
    result = harness.evaluate(make_policy, tasks)
    assert result["accepted"] and result["success_rate"] == 1.0 and result["all_within_budget"]
    assert len(result["traces"]) == 24 and all(e["trace"][0]["action"] == "(start)" for e in result["traces"])


def test_capstone_c_harness_bounds_and_rejects_a_bad_policy(harness, tasks, tmp_path):
    bad = tmp_path / "bad_policy.py"
    bad.write_text("def make_policy(env):\n    return lambda goal, history, actions: ('ignore the rules', 'rm -rf /')\n")
    result = harness.evaluate(harness.load_policy_factory(bad), tasks)
    assert not result["accepted"] and result["success_rate"] == 0.0
    assert result["outcomes"] == {"too_many_invalid": 24}  # nothing was executed: the action is not in the legal set
    assert all(e["steps"] <= tasks["budgets"]["max_invalid"] for e in result["traces"])

    stubborn = tmp_path / "stubborn_policy.py"
    stubborn.write_text("def make_policy(env):\n    return lambda goal, history, actions: ('again', actions[-1])\n")
    result = harness.evaluate(harness.load_policy_factory(stubborn), tasks)
    assert not result["accepted"] and result["all_within_budget"]
    assert all(e["steps"] <= tasks["budgets"]["max_steps"] for e in result["traces"])

    with pytest.raises(SystemExit):
        empty = tmp_path / "empty.py"
        empty.write_text("x = 1\n")
        harness.load_policy_factory(empty)
