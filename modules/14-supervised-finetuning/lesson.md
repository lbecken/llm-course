# Module 14 — Instruction tuning and supervised fine-tuning

**Part IV · From Base Model to Assistant** | Tier A–C | Study time: 7–8 hours

> A pretrained model continues documents. An assistant answers questions, stops when it
> has answered, and follows a turn structure. Nothing in the architecture distinguishes
> the two: the difference is training data. This module shows how a conversation becomes
> a token sequence, which tokens carry loss, what fine-tuning changes and what it
> destroys, and how to adapt a model by training a few percent of its parameters.

## A. Learning objectives

After this module you can:

1. Render a conversation into tokens with special tokens and a loss mask, and explain why each token is or is not supervised.
2. State what changes between pretraining and supervised fine-tuning (SFT) in data, targets and packing, and what does not change.
3. Design controls that separate what pretraining contributed from what fine-tuning contributed.
4. Measure catastrophic forgetting and counter it with replay.
5. Explain LoRA, count its parameters, and say what it can and cannot adapt.
6. Read nanochat's `render_conversation` and `chat_sft.py`.

## B. Prerequisites

- [Module 03](../03-tokenization/lesson.md) (special tokens), [Module 08](../08-data-and-pretraining/lesson.md) (packing: best-fit crop versus pad), [Module 10](../10-evaluation-and-generation/lesson.md) (evaluation with uncertainty).

## C. Motivation and intuition

Give a pretrained model the text `What is the capital of France?`. It was trained to
continue documents, and documents that begin with a question continue in many ways: with
another question, with a quiz's answer key three lines later, with a forum reply. The
model has the *knowledge*; it lacks the *behaviour* "the next thing I produce is the
answer, and then I stop".

Fine-tuning supplies that behaviour by continuing training on text that has the desired
shape:

```text
<|bos|><|user_start|>What is the capital of France?<|user_end|><|assistant_start|>Paris.<|assistant_end|>
```

The same next-token objective, on data in which a question is always followed by an
answer and an answer is always followed by an end marker. After enough of it, the most
probable continuation of `...<|assistant_start|>` is an answer.

Three ideas organise the module:

1. **A chat is a document with structure tokens.** The "chat API" of any language model
   is a rendering function from a list of messages to one token sequence.
2. **Only the assistant's tokens are targets.** The model reads the user; it should
   learn to produce the assistant.
3. **Fine-tuning moves all the weights** (unless you prevent it), so it can damage what
   pretraining built. What you do not rehearse or measure, you may lose.

## D. Rigorous technical explanation

### D.1 Rendering: from messages to tokens

A conversation is a list of messages with alternating roles. Rendering inserts special
tokens that mark who speaks and where a turn ends. nanochat's nine special tokens
(Module 03) give this layout:

| Position | Tokens | Who writes them at inference time |
|---|---|---|
| start | `<\|bos\|>` | the application |
| user turn | `<\|user_start\|>` text `<\|user_end\|>` | the application (text from the user) |
| assistant turn | `<\|assistant_start\|>` | the application: this *asks* the model to speak |
| | text | **the model** |
| | `<\|python_start\|>` expression `<\|python_end\|>` | **the model** (a tool request, Module 16) |
| | `<\|output_start\|>` result `<\|output_end\|>` | the application (the tool's result) |
| | `<\|assistant_end\|>` | **the model**: "I am done" |

Two properties matter.

**Special tokens cannot be forged by text.** They are single token ids that the ordinary
text encoder never emits (Module 03 D.6). A user who types the characters
`<|assistant_end|>` gets a sequence of ordinary character tokens, not the control token.
The turn structure is therefore under the application's control. This is the first line
of defence against prompt injection, and it is only as strong as the discipline of
encoding user text with the ordinary encoder.

**The end token is how generation stops.** The model does not "know" it is finished; it
emits a token, and the application's loop stops when that token is `<|assistant_end|>`.
A model that was never trained to produce it never stops.

### D.2 The loss mask

For every token of the rendered sequence, a mask bit says whether the model should be
trained to produce it:

```text
token   <|bos|> <|user_start|>  7  1  +  5  6  <|user_end|> <|assistant_start|>  1  2  7  <|assistant_end|>
mask       0          0         0  0  0  0  0       0                0           1  1  1         1
```

Mask 1: the assistant's text, its tool requests, and `<|assistant_end|>`. Mask 0:
everything the application or the user supplies, including tool outputs. Implementation:
targets with mask 0 are replaced by −1, which the cross-entropy ignores (Module 02).

Why not train on everything, as in pretraining?

- **User text is not the model's job.** Training on it spends capacity on predicting
  what users say, and teaches the model to *produce* user turns, i.e. to continue the
  conversation on the user's behalf.
- **Tool outputs must not be learned as outputs.** If the result of a calculator call
  were a target, the model would learn to write a plausible result itself: to
  hallucinate the tool (Module 16 demonstrates this failure).
- **Structure tokens are deterministic given the application.** `<|assistant_start|>` is
  always inserted by the application, never chosen by the model.

The mask refers to the *target*: position $t$ contributes loss if token $t+1$ has mask 1.
In the lab's data 28 % of positions are supervised; the rest is context and padding.

### D.3 What changes from pretraining, and what does not

| | Pretraining | Supervised fine-tuning |
|---|---|---|
| Objective | next-token cross-entropy | the same |
| Model, optimizer | — | the same (often lower learning rate, no weight decay) |
| Data | large, raw | small, curated conversations |
| Targets | every token | assistant tokens only |
| Packing | crop documents to fill rows (Module 08) | **pad**, never crop |
| Epochs | less than one | about one over the mixture |

Padding instead of cropping: a cropped conversation may lose its answer or its end
token, and a conversation that starts mid-way has no well-defined roles. Conversations
are packed whole into rows (best-fit), and the remainder of a row is filled with padding
whose targets are masked.

### D.4 An experiment with controls

The lab's world (`mlle/finetune/data.py`): the base model was pretrained on plain text
of two kinds, Pocket sentences and addition facts written as `47+38=85`. It has never
seen a chat token other than `<|bos|>`. Fine-tuning data: 3 000 conversations
`user: 47+38` → `assistant: 85`, built from problems whose facts were in the
pretraining text. Evaluation uses four measurements, because "the fine-tuned model
works" is not one claim:

| Measurement | Question it answers |
|---|---|
| chat accuracy, seen problems | did it learn the format? |
| chat accuracy, **held-out** problems | does it compute, or recite? |
| raw completion `47+38=` on held-out problems | does the pretraining interface still work? |
| loss on held-out Pocket sentences | does it still know the language? |

Measured (`python labs/finetuning/14_chat_sft.py`; 232 k-parameter model, 1 000
fine-tuning steps, 300 questions per column, one seed):

| Variant | chat (seen) | chat (held-out) | raw (held-out) | Pocket loss |
|---|---:|---:|---:|---:|
| base model, no fine-tuning | 0.000 | 0.000 | 0.770 | 0.4034 |
| SFT on conversations only | 0.877 | 0.880 | **0.000** | **11.17** |
| SFT + replay of pretraining batches | 0.870 | 0.880 | 0.967 | 0.4028 |
| control: more pretraining, no conversations | 0.000 | 0.000 | 0.903 | 0.4496 |
| control: SFT from random initialisation | 0.083 | 0.033 | 0.000 | 18.51 |

Reading the table row by row:

1. **The base model has the skill and lacks the interface.** It completes `47+38=`
   correctly 77 % of the time and answers 0 % of chat-formatted questions: the chat
   tokens are noise to it, and it has never produced `<|assistant_end|>`.
2. **SFT installs the interface**: 88 % in chat format, equally on seen and held-out
   problems (it computes; it does not recite). **And it destroys everything not
   rehearsed**: raw completion falls from 77 % to 0 %, and the Pocket loss rises from
   0.40 to 11.2, far worse than a model that had never seen Pocket (the uniform loss is
   $\ln 56 = 4.0$): the model is now confidently wrong about ordinary text. This is
   **catastrophic forgetting**. Nothing in the fine-tuning loss mentions the old skills,
   so nothing protects the weights that implemented them.
3. **Replay prevents it.** Mixing one batch of pretraining text into every step keeps
   the Pocket loss at its original value and the raw interface intact, at no cost in
   chat accuracy.
4. **The control "more pretraining"** shows where the gain in row 3's raw accuracy (0.97)
   comes from: 1 000 additional steps on pretraining text improve addition by
   themselves (0.90), without teaching a single thing about chat (0.000). Without this
   row one might credit fine-tuning with an improvement that is merely more training.
5. **The control "random initialisation"** shows what pretraining contributed: the same
   3 000 conversations and steps yield 3 % on held-out problems. The conversations
   suffice to *elicit* a skill, not to *create* it in that budget.

A caveat on row 2 versus row 1: fine-tuned chat accuracy (0.88) exceeds the base model's
raw accuracy (0.77). The conversations are themselves 3 000 correct addition examples,
so SFT also trains the skill. Elicitation and teaching are mixed in any fine-tuning run;
only controls like rows 4 and 5 separate them.

How much data does eliciting need? Repeating row 3 with fewer conversations
(`--n-convs`): 3 000 conversations give 0.880 on held-out problems, 100 give 0.080, and
20 give 0.010. This small model, for which even the chat tokens are new, memorises a
hundred conversations rather than connecting the format to its addition skill. Large
pretrained models pick up a format from far fewer examples because the pattern "question,
then answer" is already represented in them; do not transfer the number, only the
method of finding it.

### D.5 Forgetting in practice

Real pipelines seldom replay pretraining text. They handle forgetting by other means:

- **Mixtures.** The fine-tuning set is a mixture of tasks, so each rehearses the others.
  nanochat mixes general conversations with multiple-choice questions and arithmetic
  word problems; a task is weighted by including it several times.
- **Small steps.** Lower learning rates and about one epoch limit the distance moved
  from the pretrained weights.
- **Not caring.** After SFT nobody prompts the model in raw-completion style; forgetting
  that interface is harmless. What must survive is *knowledge* reachable through the
  chat interface, and that is what chat evaluations measure.
- **Freezing most weights** (next section).

The lesson of D.4 is not "always replay" but "what you do not measure can vanish
without the training loss telling you". The fine-tuning loss of row 2 looked excellent.

### D.6 LoRA: low-rank adaptation

**Idea.** Freeze the pretrained matrix $W \in \mathbb{R}^{m\times n}$ and learn a
correction of low rank:

$$W' = W + \frac{\alpha}{r}\,B A,\qquad A\in\mathbb{R}^{r\times n},\quad B\in\mathbb{R}^{m\times r},\quad r \ll \min(m,n)$$

- $A$ is initialised randomly and $B$ at **zero**, so $W' = W$ at the start: the adapted
  model begins as the base model.
- Trainable numbers: $r(m+n)$ instead of $mn$. For a 768 × 768 matrix and $r=8$: 12 288
  instead of 589 824 (2 %).
- Only $A$ and $B$ receive gradients and optimizer state, so training memory for
  parameters shrinks accordingly (activations do not: Module 13).
- After training, $BA$ can be **merged** into $W$ (no inference cost) or kept separate
  (one base model, many small adapters).
- The base weights are untouched: switching the adapter off restores the base model
  *exactly*. Forgetting becomes reversible.

**Numerical example.** $W$ is 4 × 4, $r=1$, $\alpha=1$: $A = (1, 0, -1, 0)$,
$B = (0.5, 0, 0, 0)^\top$. Then $BA$ has one non-zero row, $(0.5, 0, -0.5, 0)$, in row 1:
the adapter can only add multiples of one input direction ($x_1 - x_3$) to one output
direction. Rank $r$ means $r$ such (direction-in, direction-out) pairs per matrix.

**What an adapter cannot do.** The chat tokens were never trained: their embedding rows
are initialisation noise and their output rows were pushed towards "never predict this".
An adapter on attention matrices can re-route existing features but cannot give meaning
to an input row that has none, nor make the output layer emit a token whose row it does
not touch. New tokens need trainable rows. The course implementation adds a small
correction vector per new token to the embedding and output layers, keeping the base
tables frozen (`EmbeddingWithNewRows`, `LinearWithNewRows`).

Measured (`python labs/finetuning/14_lora.py`; same task and data as D.4, 1 000 steps,
learning rate 1e-3, seed 0; "chat-token rows" = 8 tokens × 96 dimensions × 2 tables):

| Variant | trainable | % of model | chat (held-out) | Pocket loss, adapter on | adapter off |
|---|---:|---:|---:|---:|---:|
| full fine-tuning (no replay) | 231 744 | 100 | 0.880 | 11.17 | n/a |
| LoRA r=8 on q, v; token rows frozen | 6 144 | 2.7 | 0.017 | 1.21 | 0.4034 |
| LoRA r=8 on q, v + chat-token rows | 7 680 | 3.3 | 0.040 | 0.75 | 0.4034 |
| LoRA r=2 on all matrices + chat-token rows | 8 448 | 3.6 | 0.330 | 1.02 | 0.4034 |
| LoRA r=8 on all matrices + chat-token rows | 29 184 | 12.6 | 0.740 | 0.64 | 0.4034 |
| LoRA r=32 on all matrices + chat-token rows | 112 128 | 48.4 | 0.823 | 2.48 | 0.4034 |

- **Attention-only adapters fail here** (2–4 %), with or without trainable token rows.
  The task needs more than re-routing: the answer must be produced in a new position
  after new tokens, which involves the MLPs. Which matrices to adapt is a design choice
  that depends on the task; "query and value projections" is a common default from the
  original paper, not a law.
- **Adapters on all matrices work**, and capacity matters: rank 2 reaches 33 %, rank 8
  74 %, rank 32 82 %, against 88 % for full fine-tuning. With 13 % of the parameters
  trainable the adapter recovers most of the effect.
- **The adapter still disturbs old behaviour while it is on** (Pocket loss 0.64 to 2.5,
  against 11.2 for full fine-tuning): a low-rank change to shared matrices affects all
  inputs, not only chat inputs. But **with the adapter off the loss is 0.4034, exactly
  the base model's**: nothing was destroyed. That guarantee, not the parameter count, is
  often the practical reason to use adapters.
- **Seeds matter more than one would like.** Three seeds of the rank-8 variant gave
  0.740, 0.727, 0.743 at learning rate 1e-3, and 0.727, 0.843, 0.790 at 3e-3; rank 32
  gave 0.823, 0.813, 0.827 at 1e-3, and 0.790, 0.863, 0.887 at 3e-3. An unseeded run at
  3e-3 once gave 0.37. The higher learning rate is better on average and much less
  reliable. Report several seeds before ranking adapter configurations.

In a model of this size the percentages are unflattering to LoRA (the eight token rows
alone are 1 536 parameters). In a large model the same construction trains well under
1 % of the weights.

### D.7 What instruction tuning does not do

- It does not add knowledge the fine-tuning data lacks. Facts come from pretraining (or
  from tools and retrieval: Parts V and VI).
- It does not make the model truthful. It makes answer-shaped text the likely
  continuation; a wrong answer in the right format is still wrong (the base model's
  23 % errors survive in chat form).
- It does not make special tokens a security boundary by itself: the boundary is the
  application's refusal to let text become control tokens.

## E. Minimal implementation

Rendering ([`mlle/finetune/chat.py`](../../mlle/finetune/chat.py), abridged):

```python
add(sp["<|bos|>"], 0)
for i, message in enumerate(conversation["messages"]):
    if message["role"] == "user":
        add(sp["<|user_start|>"], 0); add(self.encode(content), 0); add(sp["<|user_end|>"], 0)
    else:
        add(sp["<|assistant_start|>"], 0)
        for part in parts:
            if part["type"] == "text":            add(tokens, 1)
            elif part["type"] == "python":        add(sp["<|python_start|>"], 1); add(tokens, 1); add(sp["<|python_end|>"], 1)
            elif part["type"] == "python_output": add(sp["<|output_start|>"], 0); add(tokens, 0); add(sp["<|output_end|>"], 0)
        add(sp["<|assistant_end|>"], 1)
```

Masked targets ([`mlle/finetune/sft.py`](../../mlle/finetune/sft.py)):

```python
inputs, targets = ids[:, :-1], ids[:, 1:].clone()
targets[mask[:, 1:] == 0] = -1          # the mask of the TARGET token decides
loss = model(inputs, targets)           # cross-entropy with ignore_index=-1
```

LoRA ([`mlle/finetune/lora.py`](../../mlle/finetune/lora.py)):

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8.0):
        self.base = base                                    # frozen
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: starts as the base model
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```

> **Predict before you run** `python labs/finetuning/14_chat_sft.py`: after fine-tuning
> on conversations only, will the model still complete `47+38=`? Will its loss on
> ordinary sentences be a little worse, or much worse than that of an untrained model?

## F. nanochat code walkthrough

**Rendering**
([`tokenizer.py` L140–L223](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py#L140-L223)).
The course's `render_conversation` is a transcription of this function. Details worth
reading in the original:

- A leading **system message** is merged into the first user message
  (L155–L163): nanochat has no system role token; system text is user text.
- Roles must alternate, enforced by an assertion "to prevent footguns" (L179–L181).
- The comment on tool outputs (L209–L214): "none of these tokens are supervised because
  the tokens come from Python at test time".
- The result is truncated to `max_tokens` (L221–L223).

`visualize_tokenization` (L226–L239) prints the tokens green or red by mask: the
quickest check of a chat format, and the model for the course's `visualize`.
`render_for_completion` (L241–L258) removes the last assistant message and appends
`<|assistant_start|>`: the prompt used for evaluation and for the reinforcement learning
of Module 15.

**The fine-tuning script**
([`scripts/chat_sft.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_sft.py)).

*Starting point* (L93–L115): loads the base checkpoint and inherits sequence length,
batch sizes and learning rates from the pretraining run unless overridden.

*Optimizer* (L130–L160): the same Muon/AdamW combination, weight decay 0. By default it
is warm-started from the pretraining optimizer state (momentum buffers), with learning
rates reset, because "pretraining warmdown brings LRs to ~0". The initial rate is 0.8 of
the base rate (`--init-lr-frac`), constant for the first half and decaying linearly to
zero over the second (`--warmdown-ratio 0.5`).

*Data mixture*
([L161–L173](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_sft.py#L161-L173)):

```python
train_tasks = [
    SmolTalk(split="train"), # 460K rows of general conversations
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
]
train_dataset = TaskMixture(train_tasks)
```

General conversation, multiple choice (three passes: "teaches Multiple Choice") and
arithmetic word problems with calculator calls (four passes: "teaches Math and Tool
Use"). Oversampling by repetition is the mixture-weighting mechanism of D.5. One pass
over the mixture is the default length of training (`--num-iterations -1`).

*Packing and masking*
([L180–L298](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_sft.py#L180-L298)):
best-fit packing as in pretraining, but when nothing fits the row is **padded** with
BOS tokens "instead of cropping", and

```python
mask_targets = mask_tensor[:, 1:].to(device=device)
targets[mask_targets == 0] = -1
...
targets[i, content_len-1:] = -1      # padding positions
```

*Evaluation during training*: validation bits per byte on a held-out mixture, and the
ChatCORE metric of Module 10 at intervals (`--chatcore-every`).

**What nanochat does not do:** replay pretraining data, LoRA or any other
parameter-efficient method, a system-role token. LoRA in this module is a course
extension.

**On the DGX Spark:** SFT is the cheap stage. Its cost is one pass over roughly
0.8 million conversations; measure tokens per second with a short run
(`--num-iterations 20`) before the full pass, as for pretraining. The data sets are
downloaded on first use from the Hugging Face hub (network required once).
*No timings are given here: this stage was not run on GPU hardware by the author.*

## M. Key takeaways

1. A chat is one token sequence: messages plus special tokens that only the application can insert.
2. Only assistant tokens are targets; user text, structure tokens and tool outputs are masked. The end token is what makes generation stop.
3. SFT is pretraining's objective on differently shaped data. It elicits and reshapes skills; with little data it does not create them (random-initialisation control: 3 %).
4. Fine-tuning forgets what it does not rehearse, and the fine-tuning loss does not show it. Measure old skills explicitly; counter with mixtures or replay.
5. LoRA trains low-rank corrections to frozen matrices: a few percent of the parameters, mergeable, and removable to recover the base model exactly. New tokens need trainable rows that an adapter on other matrices cannot provide.
6. Attribute improvements with controls: "more training" and "no pretraining" are the two that every fine-tuning claim needs.

**Where this sits in the full system:** SFT turns the base model of Part II into the
chat model that every later module uses. Module 15 continues from this checkpoint with
training signals that are not reference answers: rewards and preferences.

Next: [lab](lab.md) · [quiz](quiz.md) · [references](references.md) · then [Module 15 — Preference optimization and reinforcement learning](../15-rl-and-preferences/lesson.md)
