# Solutions — Module 21

## Lab exercises

### Exercise 1: interfaces

1. (a) The tokenizer: pretraining relies on the exact vocabulary and special-token ids, and on the data loader using the same BOS convention. (b) The base checkpoint with its metadata: fine-tuning relies on the model configuration (sequence length, window pattern) and inherits batch sizes and learning rates. (c) The chat template: the renderer must emit exactly the special tokens, order and masking convention used in fine-tuning. (d) The request syntax and the result format: the model was trained on results formatted in one way (`5096`, not `5096.0`) and copies what it sees. (e) Traces and metrics: the experiment loop relies on the success test and the metric being computed identically across trials.
2. Token ids now mean different strings. The model receives ids it interprets with the *old* vocabulary: its input is effectively scrambled, and its output ids are decoded with the new vocabulary. Output is fluent-looking nonsense or degraded text. Nothing raises an error as long as the vocabulary size is unchanged: shapes match. Only a check that hashes the tokenizer and stores the hash with the checkpoint would catch it.
3. One GPU must process the same tokens per optimizer step: 8 accumulation steps instead of 1, about 8 × 1.5 h = 12 h for the stages that scaled with GPU count, if FP8 and the attention kernel are equally available. Not slowed by 8: tokenizer training and data download (CPU and network), single-GPU inference at a given batch size, and anything dominated by start-up or compilation. Evaluation scripts launched with `torchrun` on 8 ranks are slowed by about 8 as well.

### Exercise 2: run it, twice

Reference: lesson D.2 and D.3.

1. Only the timings differ: `seconds` of each stage, `decode_tokens_per_second`, `seconds_per_answer`, `total_seconds`. All losses, accuracies, counts and the data hash are identical.
2. With `--seed 1` a different base model is trained (the cache key includes the seed): the data hash, losses and accuracies all change, typically in the second or third decimal for losses and by a few points for the raw completion accuracy; tool-on accuracy stays at or near 1.000 and tool-off at 0.000; parameter count, vocabulary size and the KV-cache bytes per token are unchanged. Classification: *exact* (given the seed): hashes, counts, losses, accuracies. *Replicable within noise* (across seeds): the same losses and accuracies. *Timing*: everything in seconds or per second.
3. Fine-tuning (about 40 s of 49 s) with the cache; without it, pretraining (about 45 s) and fine-tuning are comparable. In nanochat's reference run pretraining is by far the longest stage: it processes billions of tokens, fine-tuning one pass over under a million conversations. The miniature's fine-tuning set is large relative to its pretraining set, which inverts the proportions.
4. Fine-tuning loss ends higher; in the serving stage the tool-call rate and tool-on accuracy fall (an assistant that has not yet learned the request format reliably), the fabricated-output rate may rise above zero, and the Pocket loss after fine-tuning stays near its base value (less movement, less forgetting). The agent stage is unaffected: it does not use the model. Every change traces to the fine-tuning stage; the serving stage merely reveals it.

### Exercise 3: break an interface on purpose

| Fault | Symptom | Crash? | Cheapest detecting check |
|---|---|---|---|
| Template mismatch (no `<\|user_end\|>`) | accuracy collapses; the model continues the "user" text or emits unrelated tokens | no | render one training conversation and one serving prompt and compare the token sequences position by position |
| Tokenizer mismatch (alphabet reordered) | output is garbage characters; accuracy 0 | no (same vocabulary size) | a tokenizer hash stored with the checkpoint, verified at load |
| Tool format mismatch (`5096.0`) | the model copies what it sees: answers like `5096.0`, or copying errors on the unseen characters; accuracy falls | no | a golden test of the tool: fixed inputs, exact expected output strings |
| Evaluation leak | accuracy *rises* or stays high; nothing looks wrong | no | an assertion that evaluation and fine-tuning question sets are disjoint |

None crashes. Each is caught by a check that takes milliseconds and compares an artefact
with what the consumer was trained on. The fourth is the most dangerous because its
symptom is good news.

### Exercise 4: observability

1. Add `manifest["code"] = {"course_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "nanochat_pinned": PINNED_SHA}` and, in the serving stage, a list of three dicts with `prompt_ids`, `generated_ids`, `mask` and `stop_reason` (`"end_token"`, `"max_tokens"` or `"cache_full"`; the engine needs a small change to report which).
2. Load both manifests, iterate over stages and metrics, and report any numeric metric with `abs(new − old) > tolerance` (skipping timing fields), exiting with status 1 if any is found. Committed alongside a reference manifest, this is a regression test of the whole pipeline: any change to data, model, fine-tuning or serving that alters a metric fails the build and names the stage.
3. One reasonable order: question and claim; code revision(s); data and its hash; exact command; seeds; hardware and software; metrics with sample sizes and uncertainty; baseline run for comparison; what was *not* run or not verified; where the manifest and logs are.

### Exercise 5: nanochat

1. Under `NANOCHAT_BASE_DIR`: `base_data_climbmix/` (dataset shards) → data; `tokenizer/` (the pickled tokenizer and token-byte table) → tokenizer; `base_checkpoints/<tag>/` with `model_*.pt`, `meta_*.json`, `optim_*_rank*.pt` → pretraining; `chatsft_checkpoints/` after fine-tuning; evaluation bundles downloaded by `base_eval`.
2. The metadata holds the model configuration and the user configuration (the parsed arguments), the step, and validation results, so the command line can be reconstructed flag by flag. Not recoverable: the code revision, the hardware, the software versions, the exact data shards present on disk at the time, and wall-clock duration unless logged separately.
3. No reference table (depends on your hardware). Every duration must be marked *measured* (from your own log) or *estimated* (tokens ÷ your measured tokens/s).
4. Provided: per-step training log, optional experiment tracker, validation bpb and CORE at intervals, sample generations, checkpoint metadata, the inference benchmark with JSON output. To add first for a week-long run: periodic checkpoints with resume tested *before* the long run starts, an alert on stalled or diverging loss, and disk-space and temperature monitoring of the machine.

### Challenge: the nanochat model in the course's loops

Outline. `complete` renders the prompt as a single user message
(`tokenizer.render_for_completion`), calls `engine.generate_batch(tokens, num_samples=1,
max_tokens=..., temperature=..., seed=...)`, and decodes the tokens after the prompt up to
`<|assistant_end|>`. Expected with a small model: the JSON tool protocol of Module 16 is
mostly beyond it (outcome `error_budget`: malformed calls), while its *native* calculator
syntax works because it was trained on it; in the agent loop of Module 18, most episodes
end as `too_many_invalid`. The exercise shows that orchestration code is model-agnostic
and that the format a model was trained on is part of its interface. *Not run by the
author on a full-size nanochat model.*

### Written questions

1. The tokenizer (ids are meaningless without it) and the chat template with its special-token conventions (the model's behaviour is conditional on that exact format). Also valuable: the tool definitions and result formats the model was trained with.
2. (a) Diff the manifests of the last good run and the bad one to find the first stage whose metrics differ. (b) Re-run that stage alone on the old and new code with the same inputs and seed. (c) Check that stage's invariant directly (round-trip, cached = uncached, rendered prompt = training format, tool golden test) before reading the diff of the refactoring.
3. The time and cost per stage for the workloads that matter (training hours per experiment, serving cost per answer at the required latency), and the cost of being wrong: how often do evaluation results mislead decisions? If experiments are slow, pretraining speed; if users wait or serving dominates cost, inference; if decisions flip between runs or do not predict use, evaluation, which is usually the highest-value and least glamorous of the three.

## Quiz

1. Data preparation → dataset shards and a held-out shard; tokenizer training → tokenizer; pretraining → base checkpoint; fine-tuning (with conversation data and a template) → chat checkpoint; optional RL/preference training → chat checkpoint; inference engine with tools → generated tokens and tool results; strategies and agent loops → answers and traces; experiment loops → changed recipes, with ledgers.
2. The tokenizer and the chat template (special tokens and rendering). Symptom: fluent but wrong, garbled or never-ending output without any error message.
3. Fine-tuning: the end token was not supervised (masked out, or conversations were cropped so it was never seen). Serving: the engine's stop condition does not check for the end token, or checks the wrong id (tokenizer mismatch).
4. Pretraining: arithmetic throughput and training memory. Prefill: arithmetic throughput. Decode at batch 1: memory bandwidth. A 30-step agent task: decode throughput times the number of steps, plus tool latency.
5. Total 46 h. Doubling pretraining speed saves 20 h (43 %). Doubling fine-tuning speed saves 1.5 h (3 %).
6. Reproducible: same code, data and seed give the same numbers. Replicable: a different seed (or fresh data) gives numbers that agree within noise. Reproducibility can be checked automatically: the lab ran the pipeline twice with one seed and compared manifests; only timings differed.
7. Any six: seeds; data hash; full configuration; code revision; software and hardware environment; per-stage metrics with settings; durations and status; the exact command.
8. Because the model does not see the user's text; it sees tokens produced by a template, special-token insertion, truncation and tool results. Problems of format, truncation and injection are visible only in the rendered sequence.
9. Evaluation stage: test questions resemble or overlap the fine-tuning data. Serving stage: the production prompt template, decoding settings or tool versions differ from those in evaluation. Usage: real inputs are longer, out of the trained range, or multi-turn, so context limits and miscopied requests appear that the evaluation never exercised.
10. `model_<step>.pt` (weights), `meta_<step>.json` (metadata), `optim_<step>_rank<r>.pt` (optimizer state per rank). `chat_sft.py` reads the model configuration to rebuild the model and inherits sequence length, device and total batch size, and learning rates from the pretraining run's configuration unless overridden.
11. Same: total batch size in tokens, number of optimizer steps, learning rates, data, model: the optimisation recipe. Changes: 8 times the accumulation steps per optimizer step (or 16 with device batch 8 if memory requires), bf16 instead of FP8 (slower matmuls, different rounding), and about an order of magnitude more wall-clock time, to be estimated from measured tokens per second.
12. Any two: SFT without replay "worked" (loss fell, chat accuracy 88 %) while the language skill was destroyed (Module 14); a tool-trained model run without its tool produced output with no error and 0 % accuracy (Module 16); an agent loop reports completion when the model judges success (Module 18); a search "improved" a validation score with pure noise (Module 20); a KV cache with a wrong position offset generates plausible text (Module 11).
