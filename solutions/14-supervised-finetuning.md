# Solutions — Module 14

## Lab exercises

### Exercise 1: render and mask

1. 17 tokens:

   ```text
   idx    0       1          2 3 4      5              6           7        8             9         10 11 12     13              14          15        16
   tok  <|bos|> <|user_start|> 2 + 2 <|user_end|> <|assistant_start|> 4 <|assistant_end|> <|user_start|> 3 + 3 <|user_end|> <|assistant_start|> 6 <|assistant_end|>
   mask   0       0          0 0 0      0              0           1        1             0          0 0 0       0               0          1         1
   ```
2. Four positions contribute (the targets `4`, `<|assistant_end|>`, `6`, `<|assistant_end|>`). The first `<|assistant_end|>` (index 8) is predicted from input position 7, the token `4`: having produced the answer, predict "stop".
3. Tokens 0–14: everything up to and including the second `<|assistant_start|>` (the last assistant message is removed, then the start token is appended). The model is expected to generate `6` and `<|assistant_end|>`.
4. Rank 4 on one $96\times96$ matrix: $4\cdot(96+96) = 768$. Query and value in 2 layers: $4\times768 = $ **3 072**. All six matrices per block: four attention matrices $4\times768 = 3\,072$, plus two MLP matrices at $4\cdot(384+96) = 1\,920$ each: 6 912 per block, **13 824** for two blocks. (Check against the lab: rank 8 gives 27 648, plus 1 536 for the token rows = 29 184.)

```python
from mlle.finetune.data import make_tokenizer
tok = make_tokenizer()
conv = {"messages": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"},
                     {"role": "user", "content": "3+3"}, {"role": "assistant", "content": "6"}]}
ids, mask = tok.render_conversation(conv)
print(tok.visualize(ids, mask)); print(len(ids), sum(mask), len(tok.render_for_completion(conv)))
```

### Exercise 2: the five-row table

Reference table: lesson D.4.

2. (a) rows 1 and 2 (chat accuracy 0.000 → 0.88). (b) rows 1 and 2 (raw 0.770 → 0.000), with row 3 showing it was avoidable. (c) rows 2 and 5: identical fine-tuning, with and without pretraining (0.880 versus 0.033 on held-out problems). (d) rows 3 and 4: replay alone raises raw accuracy to 0.903 without any conversation.
3. Typical output: fragments of digits and end tokens in place of words. The model has learned that after any context the next token is a digit or `<|assistant_end|>`; a loss of 11 nats per character means it assigns probabilities around $e^{-11}$ to the actual next letters. It is not "rusty" at Pocket; it is confidently wrong.
4. Reference (chat accuracy on held-out problems, 1 000 steps):

   | Conversations | SFT only | SFT + replay | random initialisation |
   |---:|---:|---:|---:|
   | 3 000 | 0.880 | 0.880 | 0.033 |
   | 100 | 0.093 | 0.080 | 0.000 |
   | 20 | 0.003 | 0.010 | 0.003 |

   With 100 conversations the pretrained model reaches 8–9 %; with 20, nothing. One thousand steps over 100 conversations is 320 passes: the model memorises them instead of connecting the chat format to its addition skill. The random-initialisation control is at zero throughout. So in this model: *eliciting* needs thousands of examples (not millions), and *teaching from scratch* is not achieved by any of these budgets. The gap between the two columns is what pretraining is worth. Do not carry the absolute numbers to large models, which acquire formats from far fewer examples.

A wrong approach worth naming: evaluating chat accuracy on the fine-tuning conversations
themselves. With 100 conversations that would report close to 100 % (they are
memorised) and hide the failure completely.

### Exercise 3: break the mask

Measured with 300 steps, replay on (reference with the correct mask: 0.787 on held-out problems).

1. **Train on everything.** Chat accuracy 0.727 (slightly lower; within about two standard errors). The qualitative change appears when generation is not stopped: after its answer the model writes `<|assistant_end|><|bos|><|user_start|>9+4<|user_end|><|assistant_start|>13<|assistant_end|>…`: it invents the *user's* next question and answers it. Having been trained to predict user turns, it produces them. In deployment that is a model that can continue a dialogue with itself if the stop condition is missed, and whose capacity was partly spent on modelling users. (With the correct mask, what follows the end token is untrained noise, which is harmless because the application stops there.)
2. **End token unsupervised.** Accuracy falls to 0.157. Outputs such as `166<|bos|>the children see the…`: the model never emits `<|assistant_end|>`, so the answer runs on into further digits and then into pretraining-style text. Most answers are "wrong" only because they do not stop. The end token is a behaviour that must be taught like any other.
3. **Crop instead of pad.** Packing 3 000 conversations into a stream and cutting rows of 64: 92 % of rows start in the middle of a conversation, and 18 % of conversations are cut by a row boundary. Harms: a row that begins mid-answer supervises answer tokens whose question is not in the context (the model is trained to guess); a conversation cut before its end token never teaches stopping; and the BOS-at-start assumption of the format is violated.

### Exercise 4: LoRA

Reference table: lesson D.6.

1. Rank 8, all matrices: $2\times[4\times8\cdot(96+96) + 2\times8\cdot(384+96)] = 2\times[6\,144 + 7\,680] = 27\,648$; chat-token rows $8\times96\times2 = 1\,536$; total **29 184**.
2. Measured (rank 8, learning rate 1e-3, seed 0, with chat-token rows):

   | Adapted matrices | trainable | chat (held-out) |
   |---|---:|---:|
   | q, v | 7 680 | 0.040 |
   | MLP input matrix only (`c_fc`) | 9 216 | 0.357 |
   | q, v + `c_fc` | 15 360 | 0.430 |
   | `c_fc` + both output projections (`c_proj`) | 19 968 | 0.490 |
   | all six | 29 184 | 0.740 |

   One MLP matrix alone does far more than the attention projections alone; the MLPs are where this task's adaptation happens. No subset matches adapting everything.
3. Reference at learning rate 3e-3: rank 8: 0.727, 0.843, 0.790 (mean 0.787, range 0.12); rank 32: 0.790, 0.863, 0.887 (mean 0.847, range 0.10). At 1e-3: rank 8: 0.740, 0.727, 0.743; rank 32: 0.823, 0.813, 0.827. With single runs at 3e-3 one could have observed rank 8 at 0.843 and rank 32 at 0.790 and concluded the opposite ranking. The ordering is reliable only at the lower learning rate or with several seeds.
4. "Frozen" refers to the stored base matrices. The *effective* matrix while the adapter is on is $W + \frac{\alpha}{r}BA$, and it is applied to every input, Pocket text included. Fine-tuning optimised $BA$ only for conversations, so on other inputs the correction is arbitrary. Freezing guarantees recoverability, not unchanged behaviour.
5. Keep one copy of the base weights and select per request: adapter off for plain completion, adapter on for chat (or several adapters for several behaviours). This requires *not* merging. Serving systems batch requests with different adapters by adding the low-rank term per row.

### Exercise 5: nanochat

No reference values for items 1–3 (GPU run not performed by the author). Sanity anchors:
the mixture size is printed at start-up ("Training mixture: … rows"), about 0.79 million
conversations with default settings; validation bpb is measured on *conversation* data
and is not comparable with the pretraining bpb; a model of depth 12–20 will give many
fluent wrong answers, which is expected at this scale. Item 4: assistant text, tool-call
parts and `<|assistant_end|>` are green; everything else, including
`<|assistant_start|>` and tool outputs, is red.

### Challenge: merge and stack

Outline. `merge_lora`: for each `LoRALinear`, set
`base.weight += scale * (B @ A)` and put `base` back in place of the wrapper; for the
token rows, add the deltas into the corresponding rows of the embedding and output
tables (possible only if you accept modifying those tables; keep a copy to undo). Logits
must agree to about 1e-6. Two adapters: train each from the same frozen base; switching
is replacing the wrapper parameters. Adding both at once applies
$W + \Delta_1 + \Delta_2$: each adapter was trained assuming the other absent, so their
sum is a point neither was optimised for; behaviour is typically a degraded mixture, and
with adapters trained for *conflicting* styles there is no reason to expect either
style cleanly.

### Written questions

1. (a) An evaluation of general capability the base model had (other tasks, general knowledge), to detect forgetting. (b) Controls for attribution: the base model on the same benchmark with the same prompt format, and ideally a "same amount of training without the support data" row. Also reasonable: evaluation on support questions *not* resembling the fine-tuning set.
2. `<|assistant_end|>` is a decision of the model: whether to stop is part of the answer. `<|assistant_start|>` is inserted by the application to request a turn; the model never has to choose it, and training it would teach the model to open its own turns.
3. Knowing: the information is recoverable from the weights by *some* prompt (row 1: raw completion 0.77). Being able to say it in the requested format: the route from that format to the knowledge exists (row 1: chat 0.00; row 3: chat 0.88). Fine-tuning mostly builds the route.

## Quiz

1. Not defective. It models the distribution of documents, in which a question is often followed by more text of the same kind. Missing is the *behaviour* "a question is followed by its answer, then stop", which is installed by fine-tuning on data of that shape (or approximated by prompting with a document format in which answers follow questions).
2. `<|assistant_start|>`: not a target, application. Text: target, model. `<|python_start|>` expression `<|python_end|>`: targets, model. `<|output_start|>` result `<|output_end|>`: not targets, application (tool). Final text: target, model. `<|assistant_end|>`: target, model.
3. At inference time the tool produces them. If they were targets the model would learn to generate plausible results itself, i.e. to fabricate tool outputs, and capacity would be spent imitating the tool.
4. 12 positions; $12/63 = 19\,\%$. (All 12 masked-in tokens are targets of some input position, since none is the first token.)
5. A cropped conversation can lose its answer or end token, or start mid-turn without role context; the supervised tokens would then be conditioned on a broken prefix. Documents in pretraining have no such structure to break. Padding wastes some positions (masked) but keeps every conversation whole.
6. (a) The base model lacks the knowledge. (b) It has the knowledge but not the interface (does not recognise the format or never stops). Distinguish by testing the base model through an interface it does know (raw completion, or few-shot prompting) and by fine-tuning a randomly initialised model on the same data.
7. The loss of previously learned capabilities when training continues on a different distribution. The fine-tuning loss is computed only on fine-tuning data; it contains no term for the old data, so it can fall monotonically while old capabilities are destroyed.
8. Replay or mixing old data (cost: compute and the need to keep the data); lower learning rate or fewer steps (cost: less adaptation); freezing weights and training adapters (cost: capacity, and the adapter still changes behaviour while active); regularising towards the old weights (cost: tuning, weaker adaptation); evaluating old tasks and stopping early (cost: evaluation effort). Any three.
9. $16\cdot(1024+1024) = 32\,768$, which is 3.1 % of 1 048 576. $B = 0$ makes the correction zero at the start, so training begins exactly at the pretrained model instead of at a random perturbation of it; $A$ must be non-zero or neither factor would receive a gradient.
10. Any two: it cannot give meaning to tokens whose embedding rows were never trained; it cannot make the output layer emit tokens whose rows it does not adapt; on a task needing new computation it was insufficient without the MLP matrices (2–4 % against 74 % for all matrices at the same rank); it does not preserve old behaviour while active.
11. Repeat with several seeds (adapter initialisation and batch order), report mean and spread, and check the learning rate: an unstable rate produces occasional failures. Only then compare ranks.
12. The application encodes user text with the ordinary text encoder, which maps those characters to ordinary character or sub-word tokens; the control tokens are separate ids that only `encode_special` produces. The model therefore sees literal text, inside a user turn. The mistake that breaks this: building the prompt as one string and tokenising it with a function that recognises special-token strings (or otherwise allowing user text to be parsed for them). The remaining risk, that the model *obeys* instructions written in user or tool text, is not a tokenisation issue and is the subject of Module 16.
