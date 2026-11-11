# Experiment record

Copy this file into `results/runs/` (git-ignored) for every experiment you want to keep.
One question per record.

## 1. Question and hypothesis

- **Question:**
- **Prediction (write it BEFORE running):**
- **What result would prove the prediction wrong?**

## 2. Setup

| | |
|---|---|
| Date | |
| Hardware | (`python scripts/diagnostics/check_env.py --json ...` output file) |
| Course commit | (`git rev-parse --short HEAD`) |
| nanochat commit | (must be `92d63d4` unless you are studying a difference) |
| Tier | A / B / C / D |

**Exact command(s):**

```bash
```

## 3. Variables

- **Independent variable (the one thing changed):**
- **Controls (held fixed):** seed, data, tokenizer, steps or tokens, batch size, ...
- **Metric(s):**

## 4. Results

| condition | metric 1 | metric 2 | runtime | peak memory |
|---|---:|---:|---:|---:|
| baseline | | | | |
| variant | | | | |

Number of seeds per condition: ___   Spread between seeds: ___

## 5. Interpretation

- Was the prediction right?
- Is the difference larger than the seed-to-seed spread?
- What else changed that might explain the result?
- Measured vs. expected: which numbers here are measurements, and which are estimates?

## 6. Next step
