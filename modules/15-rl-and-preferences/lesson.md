# Module 15 — Preference optimization and reinforcement learning

**Part IV · From Base Model to Assistant** | Tier A–C | Study time: 7–9 hours

> Supervised fine-tuning needs the right answer for every example. Often there is none
> to give: only a program that can check an answer, or a person who can say which of two
> answers is better. This module derives the two standard ways of learning from such
> signals, the policy gradient and direct preference optimisation, runs each on the
> assistant of Module 14, and spends as much time on how they fail as on how they work,
> because both optimise exactly what they are given and nothing else.

## A. Learning objectives

After this module you can:

1. Distinguish three training signals (label, reward, preference) by what they tell the model and what they cost to obtain.
2. Derive the policy gradient and explain the role of a baseline.
3. Implement REINFORCE for a language model and recognise it as cross-entropy on the model's own samples, weighted by advantage.
4. Explain why learning from rewards is less efficient than learning from labels, and what "no informative samples" means.
5. Demonstrate reward hacking with a flawed reward and state how to detect it.
6. Derive the DPO loss, implement it, and diagnose likelihood displacement.
7. Read nanochat's `chat_rl.py` and say which parts of PPO-style algorithms it deliberately omits.

## B. Prerequisites

- [Module 14](../14-supervised-finetuning/lesson.md) (chat format, masked loss, forgetting).
- [Module 02](../02-language-modeling/lesson.md) (sequence probability as a sum of token log-probabilities), [Module 10](../10-evaluation-and-generation/lesson.md) (sampling, uncertainty).

## C. Motivation and intuition

| Signal | What the trainer provides | What the model is told | Typical source |
|---|---|---|---|
| **Label** (SFT) | the answer | "produce exactly this" | curated data, human writers |
| **Reward** (RL) | a score for an answer the model produced | "that attempt was this good" | a checker, unit tests, a game outcome |
| **Preference** | which of two answers is better | "more like this one, less like that one" | human comparisons, an automatic judge |

A label says *what* to produce. A reward says only *how good* a particular attempt was,
and says nothing about attempts the model did not make. A preference says which of two
given answers is better, and nothing about answers not shown. Moving down the table the
signal gets cheaper to obtain and weaker per example.

Why bother with the weak signals?

- For many tasks nobody can write the best answer, but a program can verify one (a
  proof checks, tests pass, the sum is right).
- Humans compare more reliably than they compose or score.
- Rewards let a model exceed its demonstrations: it can find answers nobody wrote.

And one warning that the module will make concrete: **an optimiser given a number will
raise that number.** If the number is not exactly what you want, you will get the number.

## D. Rigorous technical explanation

### D.1 The policy gradient

Call the model a *policy* $\pi_\theta(y\mid x)$: the probability of answer $y$ (a token
sequence) given prompt $x$. A reward function $R(x, y)$ scores an answer. We want to
maximise the expected reward of the model's own samples:

$$J(\theta) = \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\big] = \sum_y \pi_\theta(y\mid x)\,R(x,y)$$

$R$ is a black box (a program): it cannot be differentiated. But $\pi_\theta$ can. Using
$\nabla\pi = \pi\,\nabla\log\pi$:

$$\nabla_\theta J = \sum_y \nabla\pi_\theta(y\mid x)\,R(x,y) = \sum_y \pi_\theta(y\mid x)\,\nabla\log\pi_\theta(y\mid x)\,R(x,y) = \mathbb{E}_{y\sim\pi_\theta}\big[R(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\big]$$

An expectation over the model's samples, so it can be estimated by sampling: draw $K$
answers, and average $R(y_k)\,\nabla\log\pi_\theta(y_k)$. In words: **make each sampled
answer more probable in proportion to its reward.** This estimator is called REINFORCE.

For a language model $\log\pi_\theta(y\mid x) = \sum_t \log\pi_\theta(y_t\mid x, y_{<t})$:
the sum of the token log-probabilities of the sampled answer, i.e. minus the
cross-entropy of the model on its own sample.

### D.2 The baseline

With rewards in $\{0, 1\}$ the estimator above only ever pushes probabilities *up*. That
is correct in expectation but noisy. Subtracting any quantity $b$ that does not depend on
the sample leaves the expectation unchanged,

$$\mathbb{E}_y\big[b\,\nabla\log\pi_\theta(y)\big] = b\,\nabla\sum_y\pi_\theta(y) = b\,\nabla 1 = 0$$

and can reduce the variance greatly. The simplest choice: the mean reward of the $K$
samples for the same prompt. The quantity $A_k = R(y_k) - \bar R$ is called the
*advantage*: better-than-average samples are made more likely, worse-than-average ones
less likely.

**Numerical example.** Four samples with rewards $(1, 0, 0, 1)$: mean 0.5, advantages
$(+0.5, -0.5, -0.5, +0.5)$. Four samples with rewards $(0, 0, 0, 0)$, or $(1, 1, 1, 1)$:
advantages all zero. **A prompt whose samples all receive the same reward contributes no
gradient.** If the model never samples a correct answer for a prompt, RL cannot teach it
that prompt: the *exploration* problem.

### D.3 The loss, and its relation to SFT

$$\mathcal{L} = -\frac{1}{N_\text{tokens}}\sum_{k} A_k \sum_{t} \log\pi_\theta\big(y_{k,t}\mid x, y_{k,<t}\big)$$

This is the masked cross-entropy of Module 14, computed on the model's *own samples*
instead of reference answers, with each sample weighted by its advantage. SFT is the
special case "the only sample is the reference, and its weight is 1". Everything else
(rendering, masking the prompt, the optimizer) is unchanged.

One training step:

```text
for each of P prompts:
    sample K answers from the current model            (no gradient; uses the KV cache)
    score each with the reward function
    advantages = rewards - mean(rewards)
loss = - sum over samples of advantage x (sum of log-probs of its sampled tokens) / number of sampled tokens
one optimizer step; then sample again FROM THE UPDATED model       ("on-policy")
```

### D.4 What larger algorithms add (conceptual)

REINFORCE takes one gradient step per batch of fresh samples. Widely used refinements,
none of which nanochat implements:

| Addition | Purpose |
|---|---|
| Learned value function (actor–critic) | a better baseline than the batch mean |
| Importance ratio and clipping (PPO) | reuse one batch of samples for several gradient steps without the policy running away from the data |
| KL penalty to a reference policy / trust region | keep the policy near the starting model; limits forgetting and collapse |
| Reward normalisation (dividing by the standard deviation) | make step sizes insensitive to reward scale |
| GRPO | PPO-style updates with the group of samples per prompt as baseline, popular for language-model reasoning tasks |

### D.5 Experiment: rewards versus labels

Starting point: the assistant of Module 14, deliberately under-trained (300 SFT steps:
about half of its answers are right). 300 training prompts; per step 16 prompts × 8
samples; 300 steps; learning rate 3e-4. Measured with
`python labs/finetuning/15_reinforce.py` (one seed; 300 questions per cell):

| | step 0 | 75 | 150 | 225 | 300 |
|---|---:|---:|---:|---:|---:|
| **A: reward = exact answer** — accuracy, training prompts | 0.533 | 0.577 | 0.597 | 0.620 | **0.637** |
| accuracy, held-out prompts | 0.523 | 0.537 | 0.537 | 0.510 | 0.517 |
| **C: SFT with reference answers**, same prompts and steps — training prompts | 0.533 | 0.687 | 0.760 | 0.823 | **0.843** |
| held-out prompts | 0.523 | 0.537 | 0.543 | 0.540 | 0.557 |

1. **Learning from a reward alone works.** No reference answer entered run A; accuracy
   on the prompts it practised rose steadily by ten points.
2. **Labels are far more efficient**: +31 points from the same prompts and steps. A
   label states the answer; a 0/1 reward on a sample conveys at most one bit, and only
   about answers the model happened to try.
3. **Much of the sampling is uninformative.** In the logged steps, only 31–62 % of the
   prompts had samples with differing rewards; the rest contributed nothing (D.2).
4. **Neither generalised to held-out prompts** (differences within the standard error
   of 0.03). Three hundred problems, each visited sixteen times, is practice on those
   problems. Generalisation needs coverage, with rewards exactly as with labels.
5. **It is fragile.** With the learning rate raised to 1e-3 (`--lr 1e-3`), accuracy on
   the training prompts first *fell* to 0.45, held-out accuracy fell to 0.37 before
   recovering to 0.49, and the loss on ordinary Pocket text rose from 0.41 to 2.6: the
   policy gradient has no term that keeps the model near where it started. Negative
   advantages push sampled answers' probabilities down without bound.

When to use which: if you can write the answers, use SFT. RL is for when you can only
check them, or when the goal is to exceed the available demonstrations.

### D.6 Reward hacking

Run B of the same lab replaces the reward by a lazy checker: **1 if the last digit and
the number of digits are right**, 0 otherwise. Easy to write, and correlated with
correctness: at the start the checker approves 94 % of greedy answers while 53 % are
actually right.

| | step 0 | 75 | 150 | 225 | 300 |
|---|---:|---:|---:|---:|---:|
| **B: reward = lazy checker** — mean sampled reward | 0.81 | 0.82 | 0.90 | 0.91 | |
| lazy checker's score, greedy | 0.943 | 0.957 | 0.970 | 0.980 | **0.987** |
| true accuracy, training prompts | 0.533 | 0.497 | 0.500 | 0.480 | **0.477** |
| true accuracy, held-out prompts | 0.523 | 0.483 | 0.450 | 0.457 | **0.460** |

Every number the training loop can see improves: the reward rises from 0.81 to 0.91 and
the checker's approval to 98.7 %. The thing we wanted gets worse: true accuracy falls by
six points, on training and held-out prompts alike. The mechanism is not mysterious.
Samples with the right last digit and wrong leading digits receive reward 1, so their
*whole* token sequence, wrong digits included, is reinforced; samples that are nearly
right but miss the last digit are suppressed.

This is **reward hacking** (also: specification gaming; an instance of Goodhart's law).
It does not require an adversarial model, only an optimiser and a gap between the
measured and the intended objective. Practical consequences:

- Keep an evaluation the optimiser never sees, and make it *stricter* than the reward.
- Read samples. A rising reward with odd-looking outputs is the typical signature.
- Prefer rewards that check the outcome itself (exact answer, tests that cover the
  behaviour) to proxies for it (length, format, a keyword, a judge's impression).
- Expect the same effect with learned reward models and human raters: a model trained
  on approval learns what earns approval (Module 20 returns to this for systems that
  improve themselves).

### D.7 Learning from preferences

**Reward models (RLHF, conceptual).** The classical pipeline: collect human comparisons
between pairs of model outputs; train a *reward model* $r_\phi(x, y)$ so that the
preferred output scores higher, with the Bradley–Terry likelihood
$P(y_w \succ y_l) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$; then run RL (usually
PPO with a KL penalty to the starting model) against $r_\phi$. Three models, two training
loops, and a learned reward that can be hacked as in D.6.

**DPO** removes the reward model and the RL loop. Derivation sketch. The KL-regularised
objective $\max_\pi\ \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_\text{ref})$
has the closed-form solution $\pi^*(y\mid x) \propto \pi_\text{ref}(y\mid x)\,e^{r(x,y)/\beta}$.
Solve for the reward: $r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \text{const}(x)$.
Insert into Bradley–Terry (the constant cancels in the difference) and treat $\pi^*$ as
the model being trained:

$$\mathcal{L}_\text{DPO} = -\log\sigma\Big(\beta\Big[\underbrace{\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}}_{\text{gain of the preferred}} - \underbrace{\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}}_{\text{gain of the rejected}}\Big]\Big)$$

A supervised loss on triples (prompt, preferred, rejected), with a frozen copy of the
starting model as reference. No sampling during training.

**Numerical example.** At the start $\pi_\theta = \pi_\text{ref}$: both gains are 0, the
loss is $-\log\sigma(0) = \ln 2 = 0.693$. If training raises the preferred answer's
log-probability by 1 and lowers the rejected one's by 2 (with $\beta=1$): margin 3, loss
$-\log\sigma(3) = 0.049$. If instead *both fall*, the preferred by 2 and the rejected by
5: margin 3, loss 0.049. **The loss cannot tell these apart.**

### D.8 Experiment: DPO succeeds at the preference, and what else happens

Setup A, a style preference (`python labs/finetuning/15_dpo.py`). The starting model was
fine-tuned on a 50/50 mixture of two answer styles, `85` and `it is 85`. Preference
data: 300 triples in which the long style is preferred over the short one *for the same
correct answer*. 100 DPO steps. Reported: the fraction of sampled answers in the
preferred style (what was optimised), exact-answer accuracy (what was not), and the mean
log-probabilities of both answers.

| Variant | long style | accuracy | log p preferred | log p rejected |
|---|---:|---:|---:|---:|
| start | 0.685 | 0.735 | −1.29 | −2.46 |
| DPO, β = 1, learning rate 3e-4 | 0.990 | **0.325** | **−3.04** | −24.24 |
| DPO, β = 5, learning rate 1e-4 | 0.981 | 0.685 | −0.87 | −7.31 |
| DPO, β = 1, 3e-4, plus 0.3 × SFT on the preferred answer | 0.998 | 0.775 | −0.19 | −12.16 |

All three variants achieve the preference: 98–100 % of answers in the long style. In the
first, **accuracy collapses from 73 % to 33 %** while the loss falls to nearly zero. The
log-probability columns show why: the rejected answer was pushed down by 22 nats, and
the preferred answer went down by 1.8 with it. This is **likelihood displacement**. The
two answers share their content (`85`); suppressing the short answer suppresses the
digits wherever they occur, and since the loss only constrains the *difference*, nothing
stops the preferred answer's probability from falling too. Probability mass moves to
other outputs: long-style answers with wrong numbers.

Two remedies, both visible in the table: stay closer to the reference (larger β, smaller
learning rate), or add an anchor that explicitly keeps the preferred answer likely.
Either preserves accuracy while achieving the preference.

Setup B, a correctness preference with near-miss negatives (the right sum against a sum
off by 1, 2 or 10), from the under-trained model of D.5:

| Variant | accuracy | log p preferred | log p rejected |
|---|---:|---:|---:|
| start | 0.505 | −1.98 | −8.97 |
| DPO, β = 1, 3e-4 | 0.380 | −2.86 | −12.20 |
| DPO + 0.3 × SFT on the preferred answer | 0.495 | −1.82 | −10.81 |

Preference training for *correctness* made the model less correct. The rejected answers
were already improbable (−9 nats); pushing them further taught nothing useful and
displaced the correct ones. Even with the anchor there is no gain. Preference
optimisation is a tool for choosing among behaviours the model already has, and it works
best when the two answers differ in the property of interest and in little else that
matters. It is not a substitute for a correctness signal.

The general lesson, again: the preference metric said "success" in every row. Only
measuring what was *not* being optimised revealed the damage.

### D.9 Choosing a method

| You have | Use | Watch for |
|---|---|---|
| reference answers | SFT | forgetting (Module 14) |
| an exact checker, no answers | policy gradient | instability; exploration; cost of sampling |
| an approximate checker or judge | policy gradient, cautiously | reward hacking: hold out a stricter evaluation |
| pairwise comparisons | DPO (or reward model + RL) | likelihood displacement; measure untargeted qualities |

## E. Minimal implementation

REINFORCE ([`mlle/finetune/rl.py`](../../mlle/finetune/rl.py)):

```python
def reinforce_step(model, optimizer, batches):          # batches: (inputs, targets, rewards) per prompt
    for inputs, targets, rewards in batches:
        advantages = rewards - rewards.mean()                                   # baseline: this prompt's mean
        logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)   # (K, T); 0 where target == -1
        objective = (logp * advantages[:, None]).sum() / (targets >= 0).sum().clamp(min=1)
        (-objective / len(batches)).backward()
    optimizer.step()
```

`targets` is −1 everywhere except on the tokens the model *sampled*: the prompt is not an
action. DPO ([`mlle/finetune/dpo.py`](../../mlle/finetune/dpo.py)):

```python
def dpo_loss(model, reference, prompt, preferred, rejected, beta=1.0, sft_weight=0.0):
    with torch.no_grad():
        ref_w, ref_l = sequence_logprob(reference, prompt, preferred), sequence_logprob(reference, prompt, rejected)
    pol_w, pol_l = sequence_logprob(model, prompt, preferred), sequence_logprob(model, prompt, rejected)
    return -F.logsigmoid(beta * ((pol_w - ref_w) - (pol_l - ref_l))) - sft_weight * pol_w
```

> **Predict before you run** `python labs/finetuning/15_reinforce.py`: in run B the
> reward is "last digit and length correct". The reward will rise. Will true accuracy
> rise with it, stay flat, or fall? Write down a mechanism for your answer.

## F. nanochat code walkthrough

[`scripts/chat_rl.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py)
trains the chat model on GSM8K word problems with a 0/1 reward. Its docstring states the
design
([L1–L10](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py#L1-L10)):

```text
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.
```

Each line removes one row of the table in D.4. What remains is D.3.

**Rollouts**
([`get_batch`, L84–L145](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py#L84-L145)).
For one training question: `render_for_completion`; sample `--num-samples` (16) answers
with the cached `Engine` at temperature 1.0 and top-k 50, up to 256 new tokens; score
each with `train_task.reward`; pad to equal length; and

```python
targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
# NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
# So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
mu = rewards.mean()
advantages = rewards - mu
```

The second comment matters for Module 16: when the engine *forces* a calculator result
into the sample, those tokens were not chosen by the policy and receive no gradient.

**The reward**
([`tasks/gsm8k.py` L109–L116](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/tasks/gsm8k.py#L109-L116))
re-uses the evaluation: 1.0 if the number after the `####` marker equals the reference,
else 0.0. An exact check of the outcome, not a proxy; the source notes it "could be made
more complex (e.g. format matching etc.)", which is precisely where the risk of D.6
would enter.

**The update**
([L263–L273](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py#L263-L273)):

```python
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
# Note, there is no need to add PPO ratio+clip because we are on policy
loss = -pg_obj
```

the course's `reinforce_step`, line for line. Gradients are accumulated over
`--examples-per-step` (16) questions before one optimizer step.

**Step size.** The learning rates start at 5 % of their base values (`--init-lr-frac
0.05`) and decay linearly to zero
([L204–L212](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py#L204-L212)).
With no trust region, a small step is the only thing keeping the policy near the SFT
model (compare D.5, item 5).

**Evaluation**
([L225–L238](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py#L225-L238)):
pass@k on held-out test questions: the fraction of questions for which at least one of
$k$ samples is correct, for $k = 1 \dots$ device batch size. Pass@1 at temperature 1 is
the sampled accuracy; the gap to pass@8 shows how often the model *can* produce the
answer without doing so reliably, which is exactly the headroom RL exploits.

**What nanochat does not have:** preference optimisation of any kind, a reward model, a
KL term, a value function. DPO in this module is a course extension.

**On the DGX Spark:** RL is dominated by *generation* (16 × 16 samples of up to 256
tokens per step), so its speed depends on decode throughput (Modules 11–12), not on
training throughput. *Not run on GPU hardware by the author; no timings are given.*

## M. Key takeaways

1. Labels say what to produce, rewards how good an attempt was, preferences which of two is better: cheaper to obtain, weaker per example.
2. The policy gradient is $\mathbb{E}[A\,\nabla\log\pi]$: cross-entropy on the model's own samples weighted by advantage. A baseline reduces variance without bias.
3. Only prompts whose samples differ in reward teach anything. RL cannot reinforce what the model never tries.
4. With the same prompts and steps, labels gave +31 points and rewards +10. Use RL when you can check but cannot demonstrate.
5. An optimiser raises the number it is given. A flawed reward raised reward and lowered accuracy; a preference loss achieved the preference and destroyed accuracy. Always measure what you care about separately from what you optimise.
6. DPO turns preferences into a supervised loss on log-probability *differences*; the difference can grow while both terms fall (likelihood displacement). Anchoring to the reference prevents it.
7. nanochat's RL is REINFORCE with a per-question mean baseline, no trust region, and a very small learning rate.

**Where this sits in the full system:** this completes the path from base model to
assistant. Part V keeps the weights fixed and improves answers by other means: tools,
additional computation at inference time, and loops. Before continuing, complete
[Milestone 4](../../instructor/assessments.md#milestone-4).

Next: [lab](lab.md) · [quiz](quiz.md) · [references](references.md) · then [Module 16 — Teaching LLMs to use tools](../16-tool-use/lesson.md)
