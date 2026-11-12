# Solutions — Module 15

## Lab exercises

### Exercise 1: a three-armed bandit

Logits $(0,0,0)$, so $\pi = (\tfrac13, \tfrac13, \tfrac13)$.

1. For a softmax, $\nabla_z\log\pi(a) = \mathbf{e}_a - \pi$. Action 1: $(\tfrac23, -\tfrac13, -\tfrac13)$; action 2: $(-\tfrac13, \tfrac23, -\tfrac13)$; action 3: $(-\tfrac13, -\tfrac13, \tfrac23)$.
2. Only action 1 has reward: $\tfrac13\cdot1\cdot(\tfrac23, -\tfrac13, -\tfrac13) = (0.222, -0.111, -0.111)$. Logit 1 rises, the others fall equally.
3. Samples $(1,2,2,3)$, rewards $(1,0,0,0)$.
   - Without baseline: $\tfrac14\cdot(\tfrac23, -\tfrac13, -\tfrac13) = (0.167, -0.083, -0.083)$.
   - With baseline 0.25, advantages $(0.75, -0.25, -0.25, -0.25)$:
     $\tfrac14\big[0.75(\tfrac23,-\tfrac13,-\tfrac13) - 0.25\cdot2\,(-\tfrac13,\tfrac23,-\tfrac13) - 0.25(-\tfrac13,-\tfrac13,\tfrac23)\big] = (0.188, -0.125, -0.063)$.

   Both point the right way (raise logit 1). The baseline version also distinguishes the unrewarded actions: action 2 was tried twice and failed twice, so it is pushed down more than action 3. Both estimates sum to zero because every $\nabla_z\log\pi(a) = \mathbf{e}_a - \pi$ sums to zero: adding a constant to all logits does not change a softmax, so no gradient can point in that direction.
4. All rewards are 0, all advantages 0: the estimate is exactly zero, although the true gradient is not. The rewarded action was never sampled, so nothing was learned: the exploration problem in its smallest form.
5. Margin $= 0.5 - (-0.5) = 1$; loss $= -\log\sigma(2\cdot1) = \log(1 + e^{-2}) = 0.127$. Same loss with the preferred answer's log-probability changed by $-1$ and the rejected one's by $-2$ (margin again 1): the loss is identical and the preferred answer is now *less* likely than under the reference.

### Exercise 2: rewards versus labels

Reference tables: lesson D.5 and D.6.

2. Standard error of an accuracy near 0.55 on 300 questions: $\sqrt{0.55\cdot0.45/300} = 0.029$; of a difference between two such figures, about 0.04 if treated as independent (they are measured on the same questions, so a paired comparison would be tighter). Clearly real: A on training prompts (+0.104, and monotonic over five checkpoints), C on training prompts (+0.31), B's fall on training prompts (−0.056, monotonic) and its rise in the lazy score. Not distinguishable from zero: every held-out change in A and C (+0.03 or less).
3. In the logged steps 38–69 % of prompts had identical rewards for all eight samples and produced no gradient. To raise the informative fraction: (a) more samples per prompt (risk: cost, and it still fails where the right answer has negligible probability); (b) a higher sampling temperature (risk: samples from a distribution the model will not use at deployment, and more junk); (c) choose prompts of intermediate difficulty for the current model, a curriculum (risk: the hard prompts are never learned); (d) partial-credit rewards (risk: exactly the reward-hacking exposure of D.6).
4. Reference: accuracy on training prompts 0.533 → 0.453 → 0.497 → 0.570 → 0.593, held-out 0.523 → 0.373 → 0.420 → 0.483 → 0.493, Pocket loss 0.41 → 2.58. An early collapse, partial recovery, and serious forgetting of unrelated text. The missing ingredient is a constraint tying the policy to the starting model: a KL penalty or trust region (the challenge exercise), or simply the small step size nanochat uses.
5. Reference (`--num-samples 2 --prompts-per-step 64`): informative fraction 0.12–0.31 instead of 0.31–0.62, as expected: with two samples a prompt is informative only if exactly one is right. Accuracy on training prompts nevertheless reached 0.630 (versus 0.637) and held-out 0.570 (versus 0.517; within noise). With the number of samples per step fixed, fewer samples on more prompts lost per-prompt signal and gained prompt coverage; here the two roughly cancelled. A two-sample baseline is also noisier. The lesson: the split between prompts and samples is a tunable trade, not a matter of principle.

### Exercise 3: catch the reward hack

1. (a) Never: the reward and the lazy checker improve at every checkpoint. (b) At the first evaluation, step 75 (held-out 0.523 → 0.483), with reasonable confidence by step 150 (0.450). A held-out evaluation with a stricter criterion is the only place the problem is visible.
2. Measured on the 300 training prompts (greedy answers before and after run B): 167 unchanged; 87 changed from one wrong answer to another whose last digit and length are correct; **30 went from right to wrong with the last digit preserved** (659+7: 666 → 766; 62+5: 67 → 77; 348+387: 735 → 765); 13 went from wrong to right (429+6: 421 → 435); 3 other. Net 17 fewer correct answers (0.533 → 0.477), while 297 of the 300 final answers satisfy the lazy checker. The model learned precisely what was paid for.
3. Examples and what to expect. "Right number of digits": reward saturates quickly (the model already satisfies it for most prompts), accuracy drifts, since no pressure distinguishes right from wrong digits. "Within 10 of the truth": a milder proxy; leading digits are protected, the last digit is free to drift, so exact accuracy tends to fall while the proxy rises. "At least as large as each operand": satisfied by almost every sum-like answer and by many wrong ones; the model can raise it by biasing answers upward. In each case: reward up, exact accuracy flat or down, and the failure concentrated on whatever the reward does not check.
4. The held-out evaluation must be at least as strict as the intended objective and must not share the reward's blind spots: ideally the exact outcome check, computed by different code from the reward, on prompts the training does not see. If no exact check exists, use several independent approximate ones and human inspection of samples.

### Exercise 4: preferences

Reference: lesson D.8, and for other step counts:

| Variant (part 1) | 20 steps: style / accuracy | 100 steps | 300 steps |
|---|---|---|---|
| β = 1, lr 3e-4 | 0.990 / 0.445 | 0.990 / 0.325 | 0.986 / 0.345 |
| β = 5, lr 1e-4 | 0.944 / 0.700 | 0.981 / 0.685 | 0.988 / 0.655 |
| β = 1 + 0.3 × SFT anchor | 0.995 / 0.650 | 0.998 / 0.775 | 1.000 / 0.725 |

(start: 0.685 / 0.735)

1. Two ways to increase the difference: raise the preferred answer's log-probability, or lower the rejected one's. The second has no floor.
2. At 100 steps: β = 1: preferred −1.75, rejected −21.8 (preferred *less* likely). β = 5: preferred +0.42, rejected −4.9. Anchor: preferred +1.10, rejected −9.7. Only the last two made the preferred answer more likely in absolute terms.
3. The preference is achieved within 20 steps in every variant (94–99 %). For plain DPO the damage is already present at 20 steps (accuracy 0.445) and deepens (0.325 at 100): **early stopping does not rescue it at this learning rate**, because style shift and damage happen together. For the conservative variants accuracy stays within about 0.03–0.08 of the start throughout (differences of that size are within two standard errors on 200 questions). What to monitor: the log-probability of the *preferred* answers under the policy (it should not fall), and a held-out measure of every quality you are not optimising.
4. The rejected near misses start at −8.97 nats on average: the model almost never produces them (probability about $10^{-4}$). A preference "right answer over near miss" is therefore already satisfied; the loss can only be reduced further by pushing an already negligible probability lower, and because near misses share most of their tokens with the right answer, the push lands on the right answer too. The pairs that would carry information are those where the rejected answer is something the model actually says.
5. For example: for each prompt, sample from the current model; where it produces a wrong answer with non-negligible probability, use (right answer, that wrong answer) as the pair, with an anchor term, and refresh the pairs as the model changes. Measure exact accuracy on held-out prompts, the log-probability of the preferred answers, and the Pocket loss. (In the author's trial of exactly this without an anchor, accuracy still fell, from 0.53 to 0.40: on-policy negatives do not remove displacement by themselves.) Expect a correctness *label* to beat any such scheme; preferences are the tool when no label exists.

A wrong approach worth naming: reporting "the DPO loss went from 0.69 to 0.03" as the
result. The loss measures the margin; in the first row a margin of 20 nats coexisted
with accuracy halved.

### Exercise 5: nanochat

1. $16 \times 16 = 256$ sampled sequences per optimizer step, each up to 256 new tokens: up to 65 536 generated tokens per step, in decode batches of `--device-batch-size` (8), plus the same sequences once more through a training forward/backward pass. Time per step ≈ generated tokens ÷ your measured decode throughput at batch 8, plus the training pass. *No reference timing: not run on GPU by the author.*
2. No reference values. Expected pattern: pass@8 is well above pass@1 at the start; RL mainly raises pass@1 towards pass@8 (it makes answers the model could already sample more reliable) rather than raising pass@8.
3. In `get_batch`: `targets[mask_ids[:, 1:] == 0] = -1`, with the comment that the engine returns mask 0 for prompt tokens *and* for tool-forced tokens. The calculator's output tokens were inserted by the engine, not sampled from the policy; they are observations, not actions. Applying the policy gradient to them would train the model to predict tool outputs, i.e. to hallucinate the tool (Module 14 D.2), and would credit or blame the policy for choices it did not make.

### Challenge: a KL penalty

Outline. Keep `reference = copy.deepcopy(model)` frozen. In `reinforce_step`, compute
the reference's per-token log-probabilities on the same `(inputs, targets)` under
`no_grad`, and add `lam * ((logp - logp_ref) * mask).sum() / num_tokens` to the loss.
This is a single-sample estimate of the KL divergence along the sampled sequences. With
`--lr 1e-3`, a penalty around 0.1–1 should reduce or remove the early collapse and keep
the Pocket loss near its starting value, at the price of a smaller final gain; too large
a penalty freezes the policy. Note what the penalty does *not* constrain: the model's
behaviour on text the rollouts never visit is tied to the reference only indirectly.

### Written questions

1. For example: (a) an independent, stricter evaluation the training never used (exact checks or human review of samples) shows the improvement too; (b) the judge's blind spots have been probed, e.g. do longer, more confident or more flattering answers score higher regardless of correctness; (c) untargeted qualities were measured before and after (accuracy on other tasks, refusals, length); (d) results hold across seeds. Any three.
2. Without a trust region or KL term, step size is the only thing limiting how far one noisy update moves the policy; a policy that moves too far samples worse data, which produces worse updates (lesson D.5 item 5). Fine-tuning on fixed labelled data has no such feedback loop.
3. In both, the optimiser reduced the stated objective by a route that was permitted but unintended: the lazy reward did not say "and the other digits must stay right"; the DPO loss did not say "and the preferred answer must stay likely". The damage was invisible in the optimised metric and obvious in an independent one. The common remedy is to measure, and where possible constrain, what the objective leaves free.

## Quiz

1. Label: the exact output wanted for this input. Reward: how good one particular sampled output was (nothing about other outputs). Preference: which of two given outputs is better (nothing about outputs not shown, and nothing about how good either is in absolute terms).
2. $\nabla J = \sum_y \nabla\pi_\theta(y)R(y) = \sum_y\pi_\theta(y)\nabla\log\pi_\theta(y)R(y) = \mathbb{E}_{y\sim\pi_\theta}[R(y)\nabla\log\pi_\theta(y)]$, using $\nabla\pi = \pi\nabla\log\pi$. The gradient acts only on $\log\pi_\theta$; $R$ appears as a number per sample, so it need not be differentiable or even known in closed form, and the expectation can be estimated from samples.
3. $\mathbb{E}_y[b\,\nabla\log\pi(y)] = b\,\nabla\sum_y\pi(y) = b\,\nabla1 = 0$ for any $b$ independent of the sample. It changes the variance of the estimate (and makes below-average samples receive negative weight).
4. Mean 0.25: advantages $(0.75, 0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25)$, summing to 0. With all rewards 0, all advantages are 0 and the gradient is exactly zero.
5. The loss is the masked cross-entropy on token sequences; in REINFORCE the sequences are sampled from the current model rather than taken from a data set, and each sequence's loss is multiplied by its advantage (possibly negative). What changes in the step: where the batch comes from (generation, then scoring) and the per-sample weight. Rendering, masking of the prompt, forward pass and optimizer are unchanged.
6. Any two: a 0/1 reward carries at most one bit per sample, a label identifies the answer among all possible ones; the reward informs only about answers the model sampled, so nothing is learned where it never samples a good one; the gradient estimate is high-variance; each update requires fresh generation, which is expensive.
7. (1) No trust region / KL regularisation to a reference: would keep the policy close to the starting model. (2) No PPO importance ratio and clipping: would allow several gradient steps on the same samples safely. (3) Token-level instead of sequence-level normalisation of the objective. (4) Advantage is $r - \mu$ without dividing by the standard deviation: z-scoring would make the step size independent of reward scale. Any three.
8. Reward hacking (specification gaming): the reward measures something that can be increased without, or at the expense of, the intended behaviour. First step: read samples with high reward and wrong answers to see what the reward is paying for; then fix the reward or constrain training, and keep the exact-match evaluation as the criterion.
9. Gains: preferred $-4-(-2) = -2$; rejected $-9-(-3) = -6$; margin $+4$; loss $\log(1+e^{-4}) = 0.018$. A very low loss, and the model is *worse* at producing the preferred answer (log-probability −4 instead of −2).
10. The reference defines "no change": the loss rewards changes in log-probability relative to it, which implements a KL-regularised objective and keeps the policy near the starting model. A larger β saturates the loss at a smaller change in the log-ratios, i.e. it keeps the policy closer to the reference.
11. For example: factual accuracy or task success (the content of answers), and behaviour on inputs unrelated to politeness; also length, refusal rate, honesty when the polite answer and the true answer differ. This module showed a style preference achieved at 99 % while exact accuracy fell from 0.73 to 0.33: optimising one property can silently damage others unless they are measured.
12. Can conclude: the model's outputs now score higher under that judge on the evaluated prompts. Cannot conclude: that answers are better by the intended standard. The judge was the optimisation target, so its score is the one number guaranteed to rise; whether quality rose requires an evaluation independent of the judge.
