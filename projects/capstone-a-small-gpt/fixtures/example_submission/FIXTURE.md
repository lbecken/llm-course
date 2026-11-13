# This directory is a test fixture, not a model submission

It exists so that `check_submission.py` has a passing example and the test suite can
exercise it. What is real and what is not:

- `train_log.jsonl`, `final_val_loss` and seed 0: the author's measured CPU run
  (`results/reference/minigpt-pocket-cpu.md`).
- The second and third "seeds" in `results.json` repeat the first value. They were not measured.
- `samples.txt` contains sentences drawn from the Pocket grammar itself, not from a model.
- `report.md` is a list of headings.

Do not copy any number from here into a report.
