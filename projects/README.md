# Capstone projects

Choose one. Each is two to four weeks of part-time work and is assessed with the rubrics
in [`instructor/grading-rubrics.md`](../instructor/grading-rubrics.md). Each directory
contains the brief, a **checker** that verifies the mechanical part of the acceptance
criteria on your submission, and **fixtures** (a small passing example) that the test
suite runs (`tests/test_capstones.py`).

| | Capstone | You should already have completed | Hardware |
|---|---|---|---|
| A | [Explain and train your own small GPT](capstone-a-small-gpt/README.md) | Parts I–II | CPU version and GPU version |
| B | [Optimize a real inference pipeline](capstone-b-inference/README.md) | Parts I and III | CPU sufficient; GPU extends it |
| C | [From LLM to tool-using agent](capstone-c-agent/README.md) | Parts I, IV (lesson 14) and V | CPU sufficient |

What the checkers do and do not do: they confirm that required files exist, that reported
numbers are internally consistent and meet the stated thresholds, and (Capstone C) they
run your agent under fixed budgets. They cannot judge an explanation or an
interpretation. A passing checker is necessary for acceptance, not sufficient.

Rules common to all three:

1. Every number in your report is labelled measured (hardware, date), reported (source) or estimated (from what).
2. Every comparison has a baseline at equal cost and a statement of run-to-run variation.
3. Work happens in this repository or your fork of it. Nothing is changed in, or submitted to, the upstream nanochat repository.
4. Any loop that acts has explicit limits on iterations, time and errors, and executes no model output as code or shell commands.
