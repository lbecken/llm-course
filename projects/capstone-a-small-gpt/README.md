# Capstone A — Explain and train your own small GPT

**Goal.** Train a small GPT, measure it properly, look inside it, and write the
explanation you wish you had been given at the start of the course. The deliverable is as
much the explanation as the model.

## Two versions

| | CPU version | GPU version (e.g. DGX Spark) |
|---|---|---|
| Model | MiniGPT (`mlle.foundations`) or the miniature nanochat pipeline of Module 08 | nanochat at a depth you choose from a measured throughput |
| Data | Pocket, or a small text corpus of your own (state its source and licence) | nanochat's dataset, or your own shards |
| Reference quantity | the entropy floor of Pocket, or an n-gram baseline for your corpus | validation bpb and CORE, with an n-gram or smaller-depth baseline |
| Time | minutes to an hour of training | sized by you; state the estimate before the run |

## Deliverables

Put them in one directory (for example `results/runs/capstone-a/`):

| File | Content |
|---|---|
| `report.md` | the written report (below) |
| `config.json` | the full model and training configuration, the seed, the course commit and (if used) the nanochat SHA |
| `results.json` | the metrics (schema below) |
| `train_log.jsonl` | one JSON object per logged step with at least `step` and `val_loss` (or `val_bpb`) |
| `samples.txt` | at least 20 generated samples, one per line |

`results.json`:

```json
{
  "final_val_loss": 0.3481,
  "reference": {"name": "entropy floor of Pocket", "value": 0.3139, "unit": "nats/char"},
  "baseline": {"name": "5-gram with backoff", "value": 0.37},
  "seeds": [0, 1, 2],
  "val_loss_per_seed": [0.3481, 0.3502, 0.3469],
  "ablation": {"name": "no positional information", "prediction": "agreement accuracy falls to chance; loss rises by > 0.05", "value": 0.4102},
  "hardware": "4-vCPU x86-64, no GPU",
  "measured_on": "2026-11-12"
}
```

**The report** (6–10 pages) must contain:

1. *What the model computes*: the forward pass for one concrete input, with shapes, from text to loss.
2. *How it was trained*: data, tokenizer, objective, optimizer, schedule, and a sizing table made before the run compared with what happened.
3. *How good it is*: the loss in context (floor or baseline), its variation over at least three seeds, samples with an honest reading of them.
4. *What is inside it*: one investigation of your choice that goes beyond looking at attention pictures: an intervention, an ablation or a probe, with a prediction made in advance.
5. *How it relates to nanochat*: a component-by-component map with the pinned revision, including what nanochat does differently and why.
6. *What you would not claim*: the limits of your evidence.

## Acceptance criteria

- [ ] `python projects/capstone-a-small-gpt/check_submission.py <your directory>` passes.
- [ ] The final validation loss is within 15 % of the reference (floor or baseline) or the report explains convincingly why not.
- [ ] At least three seeds; the spread is reported and used when interpreting the ablation.
- [ ] The investigation of point 4 has a prediction written before the run, and its interpretation survives the seed spread.
- [ ] The map to nanochat cites files and lines at the pinned revision.
- [ ] Rubrics 1, 2 and 3 at least proficient on every row.

## Suggested plan

1. Write the sizing table and predictions. 2. Train three seeds. 3. Evaluate; build the
baseline. 4. Run the ablation and the investigation. 5. Write sections 1 and 5 with the
source open. 6. Write section 6 last, and then re-read the others against it.

## Starting points

`labs/foundations/07_train_minigpt.py`, `labs/foundations/05_attention_in_a_model.py`
(the intervention hook), `labs/foundations/06_nanochat_parity.py`,
`scripts/experiments/nanochat_local.sh`, `scripts/experiments/estimate_run.py`. A passing
example of the file formats is in [`fixtures/example_submission/`](fixtures/example_submission/);
its numbers are the author's measured CPU results from
[`results/reference/minigpt-pocket-cpu.md`](../../results/reference/minigpt-pocket-cpu.md),
its report is a skeleton and would not pass the rubric.
