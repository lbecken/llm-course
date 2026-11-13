# Capstone B — Optimize a real inference pipeline

**Goal.** Take one inference path, make it measurably cheaper in one resource, prove that
the output is unchanged (or quantify how it changed), and explain the result from the
hardware up. One optimisation done rigorously is the whole project; three done loosely
are not.

## Choose a target

| Target | Baseline | Typical optimisations | Hardware |
|---|---|---|---|
| MiniGPT generation (`mlle.inference`) | naive or cached generation as it stands | batched decoding of several prompts; cache reuse across turns (prefix caching); a ring-buffer cache for sliding windows; GQA retrained and compared; int8 weights with real integer storage | CPU |
| The miniature nanochat model (Module 08) | `Engine.generate` on CPU | reuse the KV cache between chat turns; batch several samples; compare compiled and eager decode | CPU |
| A nanochat checkpoint on a GPU | `scripts/infer_bench.py` numbers | window pattern or GQA at training time, judged on quality *and* cost; decode batching policy; precision of the cache | GPU |

Anything else is welcome if it has a baseline, a correctness criterion and a resource it
targets.

## Deliverables

| File | Content |
|---|---|
| `report.md` | the written report (below) |
| `benchmarks.jsonl` | one JSON object per measured configuration (schema below) |
| the code | in this repository or your fork, with at least one test that fails if the optimisation changes the output beyond its stated tolerance |

A row of `benchmarks.jsonl` (the format written by `scripts/benchmarks/run_all.py`, plus `repeats`):

```json
{"benchmark": "my_suite", "comparison": "decode 256 tokens: one prompt at a time vs batch of 8",
 "kind": "exact", "unit": "s", "baseline": 3.44, "optimised": 0.61, "ratio": 5.6,
 "repeats": 5, "baseline_runs": [3.40, 3.44, 3.51, 3.42, 3.47], "optimised_runs": [0.60, 0.61, 0.63, 0.61, 0.60],
 "correctness": "greedy tokens identical for all 8 prompts; max logit diff 3.1e-06", "ok": true,
 "conditions": {"model": "MiniGPT 4x128", "sequence": 264, "batch": 8, "precision": "fp32"},
 "env": {"...": "from mlle.utils.environment_report()"}}
```

**The report** (5–8 pages) must contain:

1. *The target resource*: arithmetic, memory held or memory moved, and why it binds for your workload (a bound or an estimate, Module 12 D.5).
2. *The optimisation*: what changes, and whether it is exact or approximate.
3. *Correctness*: the test and its result. For approximate methods, the quality metric with its uncertainty and a targeted capability test.
4. *Measurement*: method (warm-up, synchronisation, repetitions), conditions, results with variation.
5. *Analysis*: does the measured gain match the estimate from point 1? Where does the rest go?
6. *Scope*: for which sizes, batch sizes and hardware the result should and should not be expected to hold; one such extrapolation tested.

## Acceptance criteria

- [ ] `python projects/capstone-b-inference/check_benchmark.py <your benchmarks.jsonl>` passes.
- [ ] Every row has a correctness or quality statement, at least three repetitions, stated conditions and the environment.
- [ ] No claimed gain is smaller than the run-to-run variation of its own measurements.
- [ ] The repository contains a test that would catch an incorrect version of the optimisation.
- [ ] Rubrics 2, 3 and 4 at least proficient on every row.

## Suggested plan

1. Measure the baseline three times before writing any code. 2. Write the correctness
test against the baseline's output. 3. Estimate the best possible gain. 4. Implement.
5. Measure across at least two sizes. 6. Explain the gap between estimate and
measurement.

## Starting points

`scripts/benchmarks/run_all.py` (a harness with a correctness column per row),
`mlle/bench/harness.py`, `mlle/inference/kvcache.py`, the challenge exercises of Modules
11 and 12. [`fixtures/example_benchmarks.jsonl`](fixtures/example_benchmarks.jsonl)
contains rows in the required format built from the author's measured CPU runs of the
course suite ([`results/reference/benchmarks-cpu.md`](../../results/reference/benchmarks-cpu.md)).
