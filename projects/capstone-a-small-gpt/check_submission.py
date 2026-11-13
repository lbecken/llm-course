"""
Checker for Capstone A. Verifies the mechanical acceptance criteria of a submission directory.

    python projects/capstone-a-small-gpt/check_submission.py results/runs/capstone-a

It checks that the required files exist and are well-formed, that the reported numbers
are internally consistent, and that the thresholds of the brief are met. It cannot judge
the report: passing is necessary for acceptance, not sufficient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ["report.md", "config.json", "results.json", "train_log.jsonl", "samples.txt"]
REPORT_SECTIONS = ["computes", "trained", "good", "inside", "nanochat", "not claim"]


def check(directory: str | Path) -> list[str]:
    """Returns a list of problems; empty means the mechanical criteria are met."""
    d = Path(directory)
    problems = [f"missing file: {name}" for name in REQUIRED if not (d / name).is_file()]
    if problems:
        return problems
    try:
        config = json.loads((d / "config.json").read_text())
        results = json.loads((d / "results.json").read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]

    for key in ("seed", "course_commit"):
        if key not in config:
            problems.append(f"config.json lacks '{key}'")
    for key in ("final_val_loss", "reference", "seeds", "val_loss_per_seed", "ablation", "hardware", "measured_on"):
        if key not in results:
            problems.append(f"results.json lacks '{key}'")
    if problems:
        return problems

    seeds, per_seed = results["seeds"], results["val_loss_per_seed"]
    if len(seeds) < 3 or len(seeds) != len(per_seed):
        problems.append("at least three seeds are required, with one validation loss per seed")
    elif not (min(per_seed) - 1e-9 <= results["final_val_loss"] <= max(per_seed) + 1e-9):
        problems.append("final_val_loss is not among (or within the range of) the per-seed losses")
    reference = results["reference"]
    if not {"name", "value"} <= set(reference):
        problems.append("reference needs 'name' and 'value'")
    else:
        if results["final_val_loss"] < reference["value"] and "floor" in reference["name"].lower():
            problems.append("validation loss is BELOW the stated entropy floor: check for leakage or a measurement error")
        if results["final_val_loss"] > 1.15 * reference["value"]:
            problems.append(f"validation loss is more than 15 % above the reference ({reference['value']}); explain in the report or train further")
    if not {"name", "prediction", "value"} <= set(results["ablation"]):
        problems.append("ablation needs 'name', 'prediction' (written before the run) and 'value'")

    steps = []
    for i, line in enumerate((d / "train_log.jsonl").read_text().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"train_log.jsonl line {i + 1} is not JSON")
            break
        if "step" not in row or not ({"val_loss", "val_bpb"} & set(row)):
            problems.append(f"train_log.jsonl line {i + 1} needs 'step' and 'val_loss' (or 'val_bpb')")
            break
        steps.append(row["step"])
    if len(steps) < 3:
        problems.append("train_log.jsonl needs at least three logged evaluations")
    elif steps != sorted(steps):
        problems.append("train_log.jsonl steps are not in increasing order")

    samples = [s for s in (d / "samples.txt").read_text().splitlines() if s.strip()]
    if len(samples) < 20:
        problems.append(f"samples.txt has {len(samples)} samples; at least 20 are required")
    if len(set(samples)) < len(samples) / 2:
        problems.append("more than half of the samples are duplicates: check the decoding settings (Module 10)")

    report = (d / "report.md").read_text().lower()
    for keyword in REPORT_SECTIONS:
        if keyword not in report:
            problems.append(f"report.md has no section mentioning '{keyword}' (see the six required parts in the brief)")
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    found = check(sys.argv[1])
    for p in found:
        print("PROBLEM:", p)
    print("mechanical checks passed; now apply rubrics 1-3" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
