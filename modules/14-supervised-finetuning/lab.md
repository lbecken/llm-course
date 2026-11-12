# Module 14 — Lab: teaching a base model to converse, with controls

Scripts: [`labs/finetuning/14_chat_sft.py`](../../labs/finetuning/14_chat_sft.py) ·
[`labs/finetuning/14_lora.py`](../../labs/finetuning/14_lora.py) ·
Code under study: [`mlle/finetune/chat.py`](../../mlle/finetune/chat.py), [`mlle/finetune/sft.py`](../../mlle/finetune/sft.py), [`mlle/finetune/lora.py`](../../mlle/finetune/lora.py), [`mlle/finetune/base.py`](../../mlle/finetune/base.py) ·
Solutions: [`solutions/14-supervised-finetuning.md`](../../solutions/14-supervised-finetuning.md)

The first run pretrains and caches the shared base model of Parts IV–V (about 45 s on a
4-core CPU) under `results/runs/part4/`.

## G. Exercises

### Exercise 1 (by hand): render and mask

Conversation: user `2+2`, assistant `4`, user `3+3`, assistant `6`.

1. Write the full token sequence (characters as single tokens) and the mask.
2. How many positions contribute to the loss? Which *input* position is responsible for predicting the first `<|assistant_end|>`?
3. For `render_for_completion` of this conversation: which tokens are returned, and what is the model expected to generate?
4. A LoRA adapter of rank 4 is added to the query and value projections (each $96\times96$) of a 2-layer model. How many trainable parameters? How many for rank 4 on *all six* matrices of each block (four $96\times96$ attention matrices, and MLP matrices of $384\times96$ and $96\times384$)?

**Success criterion:** items 1–3 match `ChatTokenizer.render_conversation` / `render_for_completion`; item 4 matches `count_trainable`.

### Exercise 2 (experimental): the five-row table

```bash
python labs/finetuning/14_chat_sft.py
```

1. Answer the lesson's "predict before you run" question, then record the table.
2. For each of the following claims, name the pair of rows that supports it: (a) fine-tuning installed the chat behaviour; (b) fine-tuning destroyed the completion interface; (c) the addition skill came mostly from pretraining; (d) part of the improved raw accuracy after replay is simply more training.
3. After "SFT on conversations only", prompt the model with `<|bos|>the cat` and generate 30 characters (use `complete` from `mlle.finetune.sft`). Describe what a Pocket loss of 11 looks like in samples.
4. Run with less data: `--n-convs 100` and `--n-convs 20`. How does chat accuracy on held-out problems depend on the number of conversations? What happens to the "random initialisation" control? What does that say about how much data is needed to *elicit* versus to *teach*?

**Success criterion:** three tables (3 000, 100, 20 conversations) and written answers to items 2 and 4.

### Exercise 3 (coding): break the mask

Each change is one line in a scratch copy of `build_sft_rows` or `render_conversation`.
Fine-tune (300 steps suffice) and observe.

1. **Train on everything:** do not mask (all targets kept except padding). Compare chat accuracy; then sample *without* a stop token for 30 tokens after an answer. What does the model produce after `<|assistant_end|>`, and why is that a problem in deployment?
2. **Forget the end token:** set the mask of `<|assistant_end|>` to 0. What happens to `chat_accuracy`, and what do the raw generations look like?
3. **Crop instead of pad:** pack conversations into the token stream and cut rows at fixed length (as pretraining does). Which fraction of rows now starts in the middle of a conversation? Name one concrete way that hurts.

**Success criterion:** one sentence per bug stating the symptom and the mechanism.

### Exercise 4 (experimental): LoRA

```bash
python labs/finetuning/14_lora.py
```

1. Record the table. Verify the "trainable" column by hand for the rank-8, all-matrices row.
2. The attention-only adapters fail. Propose a variant that adapts attention plus *one* MLP matrix, predict whether it will work, and test it (edit the `variants` list).
3. Run the rank-8 and rank-32 rows with `--lr 3e-3 --seed 0`, `--seed 1`, `--seed 2`. Report mean and range. Would a single run have led you to the right ranking?
4. With the adapter on, the Pocket loss is worse than the base model's. Explain why "the base weights are frozen" does not imply "old behaviour is preserved while the adapter is active".
5. Propose (do not implement) a way to obtain both behaviours from one deployed model with one copy of the base weights.

**Success criterion:** the table, a 3-seed comparison and answers to items 4–5.

### Exercise 5 (nanochat): fine-tune your base model

On the machine with your Milestone 2 base model (GPU; network access needed once for the data sets):

```bash
cd external/nanochat
python -m scripts.chat_sft --model-tag <your tag> --num-iterations 20      # probe: measure tokens/s first
python -m scripts.chat_sft --model-tag <your tag>                          # one pass over the mixture
python -m scripts.chat_eval -i sft
python -m scripts.chat_cli -i sft
```

1. From the probe, estimate the duration of the full pass *before* starting it (rows in the mixture are printed at start-up; nanochat prints progress as a fraction of the epoch).
2. Record validation bpb before and after, and ChatCORE.
3. Chat with the model. Find one question it answers correctly, one it answers fluently and wrongly, and one where it fails to stop or loses the turn structure.
4. Print one training conversation with `tokenizer.visualize_tokenization(ids, mask)` and check the colours against D.2.

On a CPU-only machine: do item 4 with the miniature tokenizer of Module 08 and a hand-written conversation, and read `chat_sft.py` L180–L298 against `build_sft_rows`.

**Success criterion:** a completed record. *Awaiting validation on GPU: no reference values for items 1–3.*

### Challenge (optional): merge and stack

Implement `merge_lora(model)` that folds every adapter into its base matrix and removes
the wrappers, and verify identical logits. Then train two adapters on the same base (one
for the short answer style, one for the long style of `mlle.finetune.data`) and show that
swapping adapters switches behaviour without touching the base weights. What goes wrong
if you naively *add* both adapters at once?

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Separate what fine-tuning adds from what pretraining supplied; measure forgetting; compare full fine-tuning with low-rank adapters |
| **Hypotheses** | (H1) The base model answers 0 % of chat-formatted questions although it completes the same facts. (H2) SFT brings chat accuracy on held-out problems close to the base model's raw accuracy or above. (H3) SFT without replay destroys raw completion and raises the Pocket loss above that of an untrained model. (H4) The same SFT from random initialisation stays below 10 % on held-out problems. (H5) A LoRA adapter on all matrices recovers most of the chat accuracy with under 15 % of the parameters, and switching it off restores the base loss exactly. |
| **Hardware / software** | CPU; course package. Exercise 5: your nanochat environment and a GPU |
| **Tier** | A: lab 14a ≈ 3–4 min, lab 14b ≈ 4 min on a 4-core CPU. Exercise 5: tier B–C |
| **Metrics** | exact-match chat accuracy (seen / held-out problems), raw completion accuracy, loss on held-out Pocket sentences, trainable parameters |
| **Independent variables** | fine-tuning data (conversations, replay, none), initialisation, number of conversations; adapter rank, target matrices, seed |
| **Controls** | one base checkpoint for all variants; same steps, learning rate, batch size and evaluation questions; "more pretraining" and "random initialisation" rows |
| **Interpretation** | 300 questions per cell: standard error up to 0.03. One seed per variant in lab 14a; lab 14b's seed dependence is part of the exercise. |
| **Common failure modes** | Reporting chat accuracy on problems that were in the fine-tuning set and calling it generalisation. Attributing all gains to fine-tuning without the "more training" control. Evaluating only the new behaviour. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.4 and D.6.

| Variant | chat (seen) | chat (held-out) | raw (held-out) | Pocket loss | your prediction |
|---|---:|---:|---:|---:|---|
| base model | | | | | |
| SFT only | | | | | |
| SFT + replay | | | | | |
| more pretraining only | | | | | |
| SFT from random init | | | | | |

| Conversations | chat (held-out), SFT + replay | chat (held-out), random init |
|---:|---:|---:|
| 3 000 | | |
| 100 | | |
| 20 | | |

| LoRA variant | trainable | chat (held-out) | Pocket, adapter on | adapter off |
|---|---:|---:|---:|---:|
| r=8, q and v, frozen token rows | | | | |
| r=8, all matrices + token rows | | | | |
| r=32, all matrices + token rows | | | | |

**Questions to answer in writing**

1. A colleague fine-tunes a model for customer support and reports only the support benchmark. Which two additional measurements do you ask for, and why?
2. Why is `<|assistant_end|>` supervised while `<|assistant_start|>` is not?
3. In your own words: what is the difference between a model *knowing* something and a model *being able to say it in the format you ask for*? Which rows of the table show each?
