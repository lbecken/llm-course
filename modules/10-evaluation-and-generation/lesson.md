# Module 10 — Evaluation and generation

**Part II · Training a Modern LLM** | Tier A–B | Study time: 6–8 hours

> A trained model is a function from context to a distribution. This module is about the
> two things you do with that distribution: *measure* it (is this model better than that
> one?) and *sample* from it (what text does it produce?). Both are easy to do badly.
> Scores get compared across incompatible setups, a lucky seed is reported as a result,
> and a decoding setting is mistaken for a property of the model. The aim here is a set
> of habits that make your numbers mean what you say they mean.

## A. Learning objectives

After this module you can:

1. Compute and convert cross-entropy, perplexity and bits per byte, and say when two losses are comparable.
2. Explain how nanochat's CORE and ChatCORE metrics are computed and what they do and do not measure.
3. Implement greedy, temperature, top-k and top-p decoding and predict their effect on quality and diversity.
4. Explain repetition and hallucination in terms of the training objective and the sampling loop.
5. Quantify uncertainty: seed variance, standard error of an accuracy, bootstrap intervals, paired comparisons.
6. Design a comparison between two models that would survive a sceptical reader.

## B. Prerequisites

- [Module 02](../02-language-modeling/lesson.md) (entropy, perplexity, generation loop), [Module 08](../08-data-and-pretraining/lesson.md) (splits, contamination).
- [Mathematics primer](../../docs/mathematics-primer.md) §5–7.

## C. Motivation and intuition

Three situations you will be in constantly:

1. *"My change lowered the loss from 0.3554 to 0.3514."* Is that an improvement or the
   difference between two random seeds?
2. *"Model A scores 61 % on the benchmark, model B 58 %."* On how many questions? Would
   the order survive a different sample of questions?
3. *"The model writes nonsense."* At which temperature? Greedy decoding and sampling at
   1.5 can make the same model look brilliant or broken.

The common thread: a number is the outcome of a *procedure* (data, tokenizer, seed,
sample size, decoding settings), and it inherits every source of variation in that
procedure. Evaluation is the discipline of knowing which.

## D. Rigorous technical explanation

### D.1 Intrinsic metrics: how well does the model predict text?

| Metric | Definition | Comparable across… |
|---|---|---|
| Cross-entropy (nats/token) | $-\frac1N\sum\log q(x_t\mid x_{<t})$ | runs with the **same tokenizer and data pipeline** only |
| Perplexity | $e^{\text{cross-entropy}}$ | same |
| **Bits per byte (bpb)** | $\dfrac{\sum_t -\log q(x_t\mid x_{<t})}{\ln 2\,\cdot\,\sum_t \text{bytes}(x_t)}$ | **different tokenizers**, same text |

Why bpb: a tokenizer with larger tokens makes each prediction harder and each token
worth more text. Dividing the *total* loss by the *total* number of UTF-8 bytes removes
the tokenizer from the unit. nanochat's implementation
([`loss_eval.py` L9–L65](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/loss_eval.py#L9-L65)):

```python
loss2d = model(x, y, loss_reduction='none')      # per-token losses, nats
num_bytes2d = token_bytes[y]                     # bytes of each target token; 0 for special tokens
total_nats += (loss2d * (num_bytes2d > 0)).sum() # special tokens are excluded from the loss ...
total_bytes += num_bytes2d.sum()                 # ... and from the byte count
...
bpb = total_nats / (math.log(2) * total_bytes)
```

Three details: it sums before dividing (a ratio of sums, not a mean of ratios); special
tokens such as `<|bos|>` contribute neither loss nor bytes; ignored targets (−1) are
handled without indexing errors. `token_bytes` is the table written by `tok_train.py`
(Module 03).

**What intrinsic metrics cannot tell you.** Lower bpb means better prediction of *that
distribution of text*. It does not say whether the model answers questions correctly,
follows instructions, or is useful. Two models with equal bpb can differ in capabilities
that occupy a tiny share of tokens. And as Module 08 showed, bpb is comparable only when
the evaluation text and its packing are the same.

### D.2 Extrinsic metrics: can the model do tasks?

**CORE** (from the DCLM paper) is nanochat's main quality number for *base* models. It
is computed without generating any text
([`core_eval.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/core_eval.py),
[`base_eval.py` L57–L123](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_eval.py#L57-L123)):

1. 22 tasks, each a set of items with a few in-context examples (few-shot prompts built from other items of the same task).
2. Each item is scored by comparing **losses**, in one of three formats:

   | Task type | Prompts built | Prediction |
   |---|---|---|
   | multiple choice | same context + each answer option | the option whose tokens have the lowest mean loss |
   | schema | each context option + the same continuation | the context under which the continuation has the lowest loss |
   | language modeling | context + continuation | correct only if the arg-max prediction equals every continuation token |

3. Per-task accuracy is **centred** against the task's random-guessing baseline:
   $\text{centred} = (\text{acc} - \text{baseline})/(1 - \text{baseline})$, so 0 means
   chance and 1 means perfect.
4. CORE is the mean of the 22 centred accuracies.

Reference points from nanochat's README: the original GPT-2 (1.6 B parameters) scores
0.2565; the leaderboard's goal is to exceed that in the least wall-clock time.

Why loss comparison instead of generation? It is deterministic (no sampling), fast (one
forward pass per option), and works for a base model that has never seen a chat format.

**ChatCORE** is the analogue for chat models
([`chat_eval.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_eval.py)):
ARC-Easy, ARC-Challenge, MMLU (multiple choice), GSM8K and HumanEval (generative), again
as a mean of centred accuracies. Two evaluation loops:

- *Categorical* (multiple choice): the conversation is rendered up to
  `<|assistant_start|>`, one forward pass is run, and the logits at that position are
  **restricted to the tokens of the answer letters**; the arg-max letter is the answer
  ([L87–L152](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_eval.py#L87-L152)).
  The source notes that this "helps the evaluation a lot": the model cannot fail by
  answering in the wrong format. It measures "which letter is most likely", not "does
  the model produce the right answer unprompted".
- *Generative*: sample a completion and check it with a program (the final number after
  `####` for GSM8K; running unit tests in a sandbox for HumanEval)
  ([L28–L80](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_eval.py#L28-L80)).

**Read every benchmark number with four questions:**

| Question | Why it matters | nanochat example |
|---|---|---|
| How many items? | sets the noise floor (D.5) | CORE during training uses at most 500 items per task (`--core-metric-max-per-task`); the leaderboard requires the full set |
| What is chance? | 25 % on 4-way multiple choice is zero skill | hence centring |
| Could the model have seen the test items, or their format? | contamination; format learning | nanochat's SFT trains on MMLU's *auxiliary_train* split and GSM8K's *train* split to teach the formats, then evaluates on the *test* splits: legitimate, but part of the measured gain is "learned the format" |
| How is the answer extracted? | restricting to letters, regex on `####`, running tests: each is more or less forgiving | categorical evaluation is lenient by design |

### D.3 Decoding: from a distribution to a token

Let $\mathbf{z}$ be the logits for the next token.

| Strategy | Rule | Parameter |
|---|---|---|
| Greedy | pick $\arg\max_i z_i$ | — |
| Temperature | sample from $\mathrm{softmax}(\mathbf{z}/\tau)$ | $\tau<1$ sharpens, $\tau>1$ flattens; $\tau=1$ is the model's own distribution |
| Top-k | keep the $k$ largest logits, renormalise, sample | $k$ |
| Top-p (nucleus) | keep the smallest set of tokens whose probability sums to at least $p$, renormalise, sample | $p$ |
| Repetition penalty | reduce logits of tokens already generated | penalty $>1$ |

**Worked example.** Logits $(2, 1, 0, -1)$:

| | token A | B | C | D |
|---|---:|---:|---:|---:|
| $\tau=1$ | 0.644 | 0.237 | 0.087 | 0.032 |
| $\tau=0.5$ (logits doubled) | 0.865 | 0.117 | 0.016 | 0.002 |
| $\tau=2$ (logits halved) | 0.455 | 0.276 | 0.167 | 0.102 |
| top-k = 2 at $\tau=1$ | 0.731 | 0.269 | 0 | 0 |
| top-p = 0.8 at $\tau=1$ | 0.731 | 0.269 | 0 | 0 |
| top-p = 0.9 at $\tau=1$ | 0.665 | 0.245 | 0.090 | 0 |

Top-k keeps a fixed *number* of candidates; top-p keeps a fixed *mass*, so it adapts:
one candidate when the model is sure, many when it is not. Both exist to cut off the
long tail of individually unlikely tokens, which together carry enough probability that
pure sampling will eventually pick one and derail the text.

nanochat implements temperature and top-k
([`engine.py` L140–L156](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py#L140-L156));
the chat CLI defaults to temperature 0.6 and top-k 50, evaluation to temperature 0.
Top-p and the repetition penalty are course code
([`mlle/inference/sampling.py`](../../mlle/inference/sampling.py)).

### D.4 Measured: the quality–diversity trade-off

For real text "quality" is a matter of judgement. For Pocket it is not: the grammar
gives every generated sentence an exact verdict
(`pocket.sentence_logprob`). `python labs/pretraining/10_decoding.py` generates 64
continuations of 60 characters per setting from one trained model (validation loss
0.3497) and scores every complete sentence. Measured, CPU, seed 0:

| Decoding | valid | unique | nats/char under the true grammar |
|---|---:|---:|---:|
| greedy ($\tau=0$) | 1.00 | **0.02** | 0.264 |
| $\tau=0.5$ | 1.00 | 0.94 | 0.301 |
| $\tau=0.8$ | 0.99 | 0.99 | 0.310 |
| $\tau=1.0$ (the model's distribution) | 0.96 | 0.96 | 0.317 |
| $\tau=1.5$ | 0.65 | 0.93 | 0.323 |
| $\tau=2.5$ | **0.06** | 1.00 | 0.338 |
| $\tau=1.5$, top-k 3 | 0.90 | 0.92 | 0.294 |
| $\tau=1.5$, top-p 0.9 | 0.99 | 0.97 | 0.315 |
| $\tau=1.0$, top-p 0.9 | 1.00 | 0.97 | 0.312 |

*valid* = grammatical including agreement; *unique* = share of distinct sentences among
the valid ones; the last column is the average surprise of the valid sentences under the
real grammar (the language itself: 0.314).

- **Greedy decoding is flawless and useless**: every sentence valid, 2 % unique. It emits
  the same most-probable sentence over and over. Its text is also *atypical*: at 0.264
  nats/char it is more predictable than the language. Maximising probability at each
  step does not produce representative text.
- **At $\tau=1$ the samples have the statistics of the language** (0.317 ≈ 0.314), and
  4 % of sentences are wrong: the model's own small errors, sampled faithfully.
- **High temperature destroys validity**: 65 % at 1.5, 6 % at 2.5, with misspelt words.
- **Truncation rescues high temperature**: at $\tau=1.5$, top-p 0.9 restores validity from
  0.65 to 0.99 while keeping diversity. The errors lived in the tail.
- **$\tau=1$ with top-p 0.9** gives 100 % valid, 97 % unique: the tail removed, the body
  of the distribution intact. This is why nucleus sampling at moderate temperature is a
  common default.

There is no universally best setting. Low temperature suits tasks with one right answer
(nanochat evaluates at 0); moderate temperature with truncation suits open-ended text.
**A decoding configuration is part of the system being evaluated and must be reported.**

### D.5 Repetition and hallucination

**Repetition.** Greedy and low-temperature decoding follow the single most probable
path. Once a phrase has occurred, the model (trained on text where recently seen
material often recurs) finds it likely again; a deterministic decoder then has no way
out of the loop. Sampling breaks the cycle by occasionally choosing another branch;
repetition penalties do it by force. In the table above the loop is visible as
*unique = 0.02*.

**Hallucination**: fluent output that is false or unsupported. No extra mechanism is
needed to explain it:

- The training objective rewards *plausible continuation*. A model that has not stored a
  fact still has a distribution over what such a fact would look like, and decoding
  picks from it. There is no built-in representation of "I do not know" unless training
  data makes such responses likely in the right contexts.
- Sampling adds random errors on top (the 4 % at $\tau=1$).
- Exposure bias (Module 02): after one wrong token the model continues consistently with
  its own mistake.

In Pocket terms, a hallucination is an *invalid but fluent* sentence: "the cats that
like the fox that see the bird". Lowering the temperature reduces the random component
(validity 0.96 → 1.00) and cannot fix systematic errors: what the model believes with
high confidence, greedy decoding repeats faithfully. Real remedies add information or
checking from outside the model: retrieval (Module 22), tools (16), verification (17).
nanochat's README invites the experiment: "Ask it to tell you who you are to see a
hallucination."

### D.6 How sure are you? Four tools

`python labs/pretraining/10_eval_methodology.py` (measured, CPU):

**1. Seed variance.** The same configuration, five seeds, 400 steps:

| seed | val loss | agreement with attractor |
|---:|---:|---:|
| 0 | 0.3514 | 1.000 |
| 1 | 0.3554 | 0.243 |
| 2 | 0.3551 | 0.173 |
| 3 | 0.3531 | 1.000 |
| 4 | 0.3535 | 0.693 |
| mean ± std | 0.3537 ± 0.0016 | 0.622 ± 0.399 |

The loss varies in the third decimal; the behavioural metric swings from 17 % to 100 %.
So: (a) the question from section C, "0.3554 → 0.3514?", is answered: that difference is
within one standard deviation of seed noise; (b) a capability can be far noisier than
the loss, especially while it is being learned; (c) "we ran it once" supports no claim
about a difference smaller than the seed spread. **Three seeds is a minimum; report
mean and spread.**

**2. Sample size.** An accuracy $a$ measured on $n$ independent items has standard error
$\sqrt{a(1-a)/n}$. For the run that scored 0.693:

| items | standard error | ≈ 95 % interval |
|---:|---:|---:|
| 30 | 0.084 | ± 0.165 |
| 100 | 0.046 | ± 0.090 |
| 300 | 0.027 | ± 0.052 |
| 3 000 | 0.008 | ± 0.017 |

Halving the error bar takes four times the items. A 3-point difference between two
models on a 100-question benchmark is noise. This is also why nanochat's log calls CORE
"a little bit noisy" and why ChatCORE with `--chatcore-max-sample 24` generative
problems per task during training is a trend indicator, not a measurement.

**3. Bootstrap intervals.** When no formula applies (a mean of per-item scores, a ratio,
a median), resample the items with replacement many times and read the spread off the
resampled means. For the 300 items above: [0.640, 0.747], matching the formula.

**4. Paired comparison.** Two models scored on the *same* 300 sentences (negative
log-probability per sentence; lower is better):

| | mean | 95 % interval |
|---|---:|---|
| model A (600 steps) | 10.074 | [9.648, 10.496] |
| model B (450 steps) | 10.116 | [9.679, 10.545] |
| **paired difference B − A** | **+0.042** | **[+0.012, +0.070]**, P(A better) = 0.998 |

The individual intervals overlap almost completely, because sentences differ enormously
in difficulty (standard deviation 3.8 nats): that shared variation swamps a 0.04
difference. Comparing the two models *item by item* removes it, and the difference
becomes clear. **When two systems are evaluated on the same items, analyse the per-item
differences.** "The error bars overlap" is not a valid argument against a paired effect.

### D.7 A checklist for a comparison you can defend

1. **Write the prediction first** (and what would refute it).
2. **Change one thing.** Same data, tokenizer, evaluation pipeline, decoding settings.
3. **Equalise the right budget**: tokens, FLOPs or wall-clock, stated explicitly (Module 09).
4. **Use held-out data** that has not guided earlier decisions; check for contamination (Module 08).
5. **Several seeds** per arm; report mean and spread.
6. **Enough items**; compute the standard error before interpreting.
7. **Pair** when the same items are used.
8. **Report negative results and the full configuration.** nanochat's `dev/LOG.md` is a
   model here: many entries end "Not adopted".
9. **Distinguish measured from expected** in what you write down.

## E. Minimal implementation

- Decoding: [`mlle/inference/sampling.py`](../../mlle/inference/sampling.py). Nucleus filtering, the one non-obvious function:

  ```python
  def top_p_filter(logits, p):
      sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
      probs = torch.softmax(sorted_logits, dim=-1)
      cumulative = probs.cumsum(dim=-1)
      drop_sorted = (cumulative - probs) >= p     # drop a token if the mass BEFORE it already reaches p
      drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
      return logits.masked_fill(drop, float("-inf"))
  ```

- Statistics: [`mlle/evaluation.py`](../../mlle/evaluation.py): `bits_per_byte`,
  `accuracy_stderr`, `mean_std`, `bootstrap_ci`, `paired_bootstrap`, each a few lines.

- Ground truth for Pocket: `pocket.sentence_logprob` in
  [`mlle/data/pocket.py`](../../mlle/data/pocket.py), a recursive-descent recogniser that
  returns the exact log-probability of a sentence or `None` if it is not in the language.

> **Predict before you run** `python labs/pretraining/10_decoding.py`: greedy decoding
> will produce only valid sentences. What fraction of them will be distinct: about 90 %,
> 50 %, or under 5 %?

## F. nanochat code walkthrough

| Task | Command | What it runs |
|---|---|---|
| bits per byte, samples (base) | `python -m scripts.base_eval --eval bpb,sample` | `evaluate_bpb` on train and validation loaders; 7 fixed prompts at temperature 0 and 8 unconditioned samples at temperature 1 |
| CORE (base) | `python -m scripts.base_eval --eval core --max-per-task 100` | downloads the evaluation bundle once, then the three task formats of D.2; writes a CSV per model |
| ChatCORE (chat) | `python -m scripts.chat_eval -i sft` | categorical and generative loops; `-a MMLU` selects a task, `-x 100` limits problems |
| inference cost | `python -m scripts.infer_bench -i base` | latency, throughput and memory (Module 11–12); GPU only |

Points to read in the source:

- **Sampling**
  ([`engine.py` L140–L156](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py#L140-L156)):
  temperature 0 short-circuits to arg-max; with top-k, the k largest logits are selected
  *first* and only those are divided by the temperature and sampled, which avoids a
  softmax over the whole vocabulary.
- **Seeds**: `Engine.generate` takes a `seed` and creates its own generator, so
  generation is reproducible and independent of global random state. The RL script
  derives a fresh seed for every sampling call for the same reason (Module 15).
- **Few-shot construction for CORE**
  ([`core_eval.py` L167–L241](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/core_eval.py#L167-L241)):
  the in-context examples for item `idx` are drawn with `random.Random(1234 + idx)`,
  excluding the item itself: deterministic, and no item is its own example.
- **The source's own caveat**: the file header notes one task (SQuAD) on which nanochat's
  implementation gets 31 % where the reference reports 37 %, cause unknown. An evaluation
  harness is software and has bugs and discrepancies like any other; absolute numbers
  from different harnesses are not interchangeable.
- **The leaderboard rule**
  ([`dev/LEADERBOARD.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LEADERBOARD.md)):
  a claimed improvement must report CORE on the full task set *and* validation bpb,
  "because the CORE metric can be a little bit noisy", and must hold across depths, not
  for one configuration.

CORE requires real-world knowledge and is meaningless for a model trained on Pocket,
which is why the miniature pipeline of Module 08 disables it. You will run it on your
Milestone 2 model:

```bash
cd external/nanochat
python -m scripts.base_eval --model-tag d6-first --eval core,bpb,sample --max-per-task 100 --device-batch-size 16
```

(*Awaiting validation on GPU.* Expect a CORE far below GPT-2's 0.2565 for a depth-6
model; the number is your baseline for later comparisons, not a target.)

## M. Key takeaways

1. Cross-entropy and perplexity are comparable only within one tokenizer and data pipeline; bits per byte removes the tokenizer. None of them measures task ability.
2. CORE and ChatCORE turn tasks into centred accuracies. Know the number of items, the chance level, the extraction rule and the contamination risk of every benchmark number you read.
3. Decoding is a choice outside the model. Greedy text is repetitive and atypical; high temperature breaks validity; truncating the tail (top-k, top-p) buys diversity without the errors.
4. Hallucination and repetition follow from the objective and the loop; temperature controls only the random part.
5. Seeds, sample size and pairing decide whether a difference is real. Behavioural metrics can be far noisier than the loss.
6. A comparison is credible when its prediction, controls, budget, seeds and negative results are all on the page.

**Where this sits in the full system:** evaluation is the feedback signal for every other
stage: it decides which data, architecture, optimizer and fine-tuning recipe survive.
Part III keeps the model fixed and asks how to run it faster.

## Milestone 2

*Train and evaluate at least one small nanochat model, log configuration and results,
and investigate one variable with a controlled experiment.* The pieces are Module 08
Exercise 6 (the model), Module 09 Exercise 5 (the experiment) and this module's
methodology. Checklist and rubric:
[instructor/assessments.md](../../instructor/assessments.md#milestone-2).

Next: [lab](lab.md) · [quiz](quiz.md) · [references](references.md) · then [Module 11 — The KV cache](../11-kv-cache/lesson.md)
