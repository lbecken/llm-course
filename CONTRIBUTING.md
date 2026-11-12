# Contributing

Corrections, validation reports from real hardware, and improvements are welcome.

## Ground rules

1. **Topic branches and pull requests.** Do not push to `main`. One logical change per PR.
2. **Never modify or vendor nanochat.** It stays external at the pinned commit
   ([docs/upstream-version.md](docs/upstream-version.md)). To study a modification, describe
   it as a patch or wrapper in a lab.
3. **No fabricated results.** A number in the repository is one of:
   - *measured*: with hardware, date, software versions and the exact command (put it in `results/reference/`);
   - *reported*: with a link to the source and its date;
   - *expected*: clearly worded as an expectation.
4. **Every file referenced must exist.** `make check-docs` verifies relative links,
   backticked repository paths and nanochat source links (file and line range at the pin).
5. **Code must be tested.** New functionality in `mlle/` comes with a CPU test in `tests/`
   that runs in seconds. GPU-only behaviour gets a test marked `@pytest.mark.gpu` plus a
   documented manual validation step.
6. **Safety limits are not optional.** Tool execution stays allow-listed and sandboxed;
   agent and experiment loops must have explicit iteration and time budgets.

## Before opening a pull request

```bash
make test          # fast CPU suite
make test-all      # includes slow tests
make check-docs    # links, paths, source references
make test-nanochat # if you touched MiniGPT, the bridge, or Module 06
```

## Reporting hardware validation (especially DGX Spark)

Fill in [results/templates/spark-validation.md](results/templates/spark-validation.md),
attach the JSON from `python scripts/diagnostics/check_env.py --json ...` and the probe
JSON files, and open a PR that adds them under `results/reference/` and updates the
labels in `docs/dgx-spark-setup.md` and `PROGRESS.md` from "[to validate]" to
"validated on <date>, <software versions>".

## Writing style for lessons

- Follow the module template (sections A–M across `lesson.md`, `lab.md`, `quiz.md`, `references.md`, `solutions/`).
- Intuition, then mathematics, then a small numerical example, then code, then nanochat, then experiment.
- Define every symbol. Give tensor shapes at every step. Link formulas to code lines.
- Say what is mathematics, what is a kernel-level optimization and what is system design.
- Avoid describing models as having mental states. Describe computations and measurements.
- In Markdown tables, write special tokens with escaped pipes: `` `<\|bos\|>` ``.

## Updating the nanochat pin

See "Updating the pin" in [docs/upstream-version.md](docs/upstream-version.md).
