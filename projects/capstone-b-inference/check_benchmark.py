"""
Checker for Capstone B. Verifies the mechanical acceptance criteria of a benchmark file.

    python projects/capstone-b-inference/check_benchmark.py results/runs/capstone-b/benchmarks.jsonl

For every row: required fields, at least three repetitions per side, a correctness
statement, stated conditions and environment, consistency of the summary numbers with
the individual runs, and a claimed gain that exceeds the run-to-run variation.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REQUIRED = ["comparison", "kind", "unit", "baseline", "optimised", "ratio", "correctness", "ok", "conditions", "env",
            "baseline_runs", "optimised_runs"]


def check_row(row: dict, where: str) -> list[str]:
    problems = [f"{where}: missing field '{k}'" for k in REQUIRED if k not in row]
    if problems:
        return problems
    if not str(row["correctness"]).strip():
        problems.append(f"{where}: empty correctness statement")
    if row["ok"] is not True:
        problems.append(f"{where}: the row's own correctness check did not pass (ok is not true)")
    if not row["conditions"]:
        problems.append(f"{where}: conditions (model, sizes, batch, precision) are not stated")
    b, o = row["baseline_runs"], row["optimised_runs"]
    if len(b) < 3 or len(o) < 3:
        problems.append(f"{where}: at least three repetitions per side are required")
        return problems
    for name, runs, summary in (("baseline", b, row["baseline"]), ("optimised", o, row["optimised"])):
        if not (min(runs) - 1e-12 <= summary <= max(runs) + 1e-12):
            problems.append(f"{where}: {name} summary {summary} lies outside its own runs {min(runs)}..{max(runs)}")
    if row["optimised"] > 0 and abs(row["ratio"] - row["baseline"] / row["optimised"]) > 0.02 * abs(row["ratio"]):
        problems.append(f"{where}: ratio {row['ratio']} is not baseline / optimised")
    if row["unit"] in ("s", "ms"):  # for timings the two sets of runs must not overlap
        if row["ratio"] > 1 and min(b) <= max(o):
            problems.append(f"{where}: claimed speed-up, but the slowest optimised run ({max(o)}) is not faster than the fastest baseline run ({min(b)})")
        spread = max(statistics.pstdev(b) / statistics.mean(b), statistics.pstdev(o) / statistics.mean(o))
        if abs(row["ratio"] - 1) < 2 * spread:
            problems.append(f"{where}: the claimed change ({row['ratio']:.2f}x) is within twice the run-to-run variation ({spread:.1%})")
    return problems


def check(path: str | Path) -> list[str]:
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        return ["no rows"]
    problems = []
    for i, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"row {i}: invalid JSON ({e.msg})")
            continue
        problems += check_row(row, f"row {i}")
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    found = check(sys.argv[1])
    for p in found:
        print("PROBLEM:", p)
    print("mechanical checks passed; now apply rubrics 2-4" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
