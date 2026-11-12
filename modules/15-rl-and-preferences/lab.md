# Module 15 — Lab: rewards, flawed rewards, and preferences

Scripts: [`labs/finetuning/15_reinforce.py`](../../labs/finetuning/15_reinforce.py) ·
[`labs/finetuning/15_dpo.py`](../../labs/finetuning/15_dpo.py) ·
Code under study: [`mlle/finetune/rl.py`](../../mlle/finetune/rl.py), [`mlle/finetune/dpo.py`](../../mlle/finetune/dpo.py) ·
Solutions: [`solutions/15-rl-and-preferences.md`](../../solutions/15-rl-and-preferences.md)

Both labs start from checkpoints derived from the Module 14 base model and cache them
under `results/runs/part4/`.

## G. Exercises

### Exercise 1 (by hand): policy gradients on a three-armed bandit

A "model" chooses one of three answers with logits $(0, 0, 0)$. Rewards: answer 1 → 1,
answers 2 and 3 → 0.

1. Compute $\nabla_{z}\log\pi(a)$ for each action $a$ (the gradient with respect to the three logits).
2. Compute the exact policy gradient $\sum_a \pi(a)\,R(a)\,\nabla\log\pi(a)$. In which direction do the logits move?
3. You sample four times and get actions $(1, 2, 2, 3)$. Compute the REINFORCE estimate without a baseline and with the mean reward as baseline. Compare both with the exact gradient.
4. You sample four times and get $(2, 3, 3, 2)$. What is the estimate with baseline? What does this illustrate?
5. For a DPO triple with $\beta = 2$: the policy has raised the preferred answer's log-probability by 0.5 over the reference and lowered the rejected one's by 0.5. Compute the loss. Then give a second, different pair of changes with exactly the same loss in which the preferred answer became *less* likely.

**Success criterion:** numerical answers; both estimates of item 3 sum to zero over the three logits (explain why they must).

### Exercise 2 (experimental): rewards versus labels

```bash
python labs/finetuning/15_reinforce.py
```

1. Answer the "predict before you run" question of the lesson. Record the three tables.
2. Using the standard error of an accuracy on 300 questions, which differences between step 0 and step 300 are clearly real, which are not?
3. From the "informative prompts" column: roughly what fraction of the sampling effort produced no gradient? Propose two changes that would raise it, and say what each risks.
4. Run with `--lr 1e-3`. Describe what happens to accuracy and to the Pocket loss. Which missing ingredient of D.4 addresses this?
5. Run with `--num-samples 2` (and `--prompts-per-step 64` to keep the number of samples per step equal). Predict the effect on the informative fraction and on learning.

**Success criterion:** tables for the default run and one variation, with a written comparison.

### Exercise 3 (analysis): catch the reward hack

Use the output of run B.

1. Plot or tabulate the reward, the lazy checker's score and true accuracy against the step. At which step would you have noticed a problem if you had only monitored (a) the reward, (b) held-out accuracy?
2. Print 20 greedy answers of the model before and after run B for prompts where the two differ (write 15 lines using `complete`). Classify the changes: how many went from right to wrong with the last digit preserved?
3. Design a *different* flawed reward for this task that looks reasonable (examples: "the answer has the right number of digits"; "the answer is within 10 of the truth"; "the answer is at least as large as each operand"). Predict what the model will learn, then run it by passing your function to `reinforce_train`.
4. State a rule for choosing the held-out evaluation when the reward is known to be approximate.

**Success criterion:** a described hack for a reward of your own design, with measured reward and true accuracy.

### Exercise 4 (experimental): preferences

```bash
python labs/finetuning/15_dpo.py
```

1. Answer the "predict before you run" question in the script's docstring. Record both tables.
2. For each variant of part 1 compute the change in "log p preferred" and in "log p rejected". In which variants did the preferred answer become more likely in absolute terms?
3. Run with `--steps 20` and `--steps 300`. How early is the style preference achieved, and how does the damage develop with further training? What does this suggest about early stopping, and what would you monitor to decide when to stop?
4. Part 2 fails to improve correctness even with the anchor. Explain why near-miss negatives carry so little useful information for this model, using the starting log-probabilities.
5. Design preference pairs for this task that you would expect to *help*, and say what you would measure.

**Success criterion:** recorded tables for three step counts and written answers to items 2, 4 and 5.

### Exercise 5 (nanochat): RL on your chat model

GPU, with your SFT model from Module 14:

```bash
cd external/nanochat
python -m scripts.chat_rl --model-tag <your tag> --eval-every 30 --num-epochs 1
```

1. Before starting: how many sequences are sampled per optimizer step with the defaults? Up to how many tokens? Use your decode throughput from Module 11 to estimate the time per step.
2. Record pass@1 and pass@8 at the first evaluation and at the last. Which moved more?
3. In `get_batch`, find where tool-forced tokens are excluded from the loss. Why would including them be wrong?
4. Stop after a few evaluations if time is short: this stage is the most expensive per unit of improvement in the pipeline.

On a CPU-only machine: answer items 1 and 3 from the source, and compare `get_batch` with `rollout_batch` line by line.

**Success criterion:** written answers; measured values where a GPU is available. *Awaiting validation on GPU: no reference values.*

### Challenge (optional): a KL penalty

Add a term $\lambda\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_\text{ref}\big)$, estimated on the
sampled tokens as $\lambda\sum_t\big[\log\pi_\theta(y_t) - \log\pi_\text{ref}(y_t)\big]$, to
`reinforce_step`, with a frozen copy of the starting model as reference. Show that with
`--lr 1e-3` the collapse of Exercise 2.4 is prevented or reduced for a suitable $\lambda$,
and report the effect on the Pocket loss and on the final accuracy.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Show that a 0/1 reward can train a language model; compare its efficiency with labels; demonstrate reward hacking and likelihood displacement with measurements |
| **Hypotheses** | (H1) With an exact reward, accuracy on the training prompts rises. (H2) SFT on the same prompts with labels rises at least twice as much. (H3) With the lazy reward, the reward rises while true accuracy does not rise (it may fall). (H4) DPO raises the share of the preferred style above 95 %. (H5) Plain DPO lowers the preferred answer's log-probability in at least one configuration; an anchor or a larger β prevents the loss of accuracy. |
| **Hardware / software** | CPU; course package. Exercise 5: nanochat environment and a GPU |
| **Tier** | A: lab 15a ≈ 3–4 min, lab 15b ≈ 3–5 min on a 4-core CPU. Exercise 5: tier C |
| **Metrics** | exact-match accuracy on training and held-out prompts; lazy checker's score; mean sampled reward; fraction of informative prompts; share of long-style answers; mean log-probability of preferred and rejected answers; Pocket loss |
| **Independent variables** | reward function; training signal (reward, label, preference); learning rate; β; anchor weight; number of steps |
| **Controls** | all runs from the same checkpoint, same prompts, same number of steps; evaluation by a checker that the training never sees (exact match) |
| **Interpretation** | 300 questions per accuracy cell (200 in lab 15b): standard error about 0.03. One seed. Trends over five checkpoints are more trustworthy than any single cell. |
| **Common failure modes** | Monitoring only the reward. Evaluating with the training reward. Concluding from run A that "RL generalises" or from held-out accuracy that "RL does not work": it improved what it practised. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.5, D.6, D.8.

| Run | metric | step 0 | 75 | 150 | 225 | 300 |
|---|---|---:|---:|---:|---:|---:|
| A exact reward | accuracy, training prompts | | | | | |
| | accuracy, held-out | | | | | |
| B lazy reward | lazy checker | | | | | |
| | accuracy, training prompts | | | | | |
| C labels | accuracy, training prompts | | | | | |

| DPO variant | long style | accuracy | Δ log p preferred | Δ log p rejected |
|---|---:|---:|---:|---:|
| β = 1, lr 3e-4 | | | | |
| β = 5, lr 1e-4 | | | | |
| β = 1 + anchor | | | | |

**Questions to answer in writing**

1. Your team's model is trained with a reward computed by another model that judges answer quality. List three checks you would require before trusting a reported improvement.
2. Why does nanochat's RL script use a learning rate twenty times smaller than its fine-tuning script?
3. In one paragraph: what do reward hacking and likelihood displacement have in common?
