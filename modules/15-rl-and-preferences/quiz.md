# Module 15 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/15-rl-and-preferences.md`](../../solutions/15-rl-and-preferences.md).

1. **(Concept)** Name the three training signals of this module and state, for each, what a single example tells the model.

2. **(Derivation)** Starting from $J(\theta) = \sum_y \pi_\theta(y)\,R(y)$, derive $\nabla J = \mathbb{E}_{y\sim\pi_\theta}[R(y)\,\nabla\log\pi_\theta(y)]$. Why is this form useful when $R$ is a program?

3. **(Concept)** Why does subtracting a baseline not change the expected gradient? What does it change?

4. **(Calculation)** Eight samples for one prompt receive rewards $(1,1,0,0,0,0,0,0)$. Give the advantages. What is the sum of the advantages, and what would the gradient be if all eight rewards were 0?

5. **(Concept)** Explain the statement "REINFORCE is supervised fine-tuning on the model's own samples, weighted by advantage". Which lines of an SFT training step change?

6. **(Concept)** Give two reasons why a 0/1 reward is a less efficient training signal than a reference answer.

7. **(Code reading)** nanochat's `chat_rl.py` lists four simplifications relative to GRPO/PPO. Name three and say what each omitted component is for.

8. **(Diagnosis)** During RL the mean reward rises from 0.6 to 0.9 while a held-out exact-match evaluation falls from 0.5 to 0.4. Name the phenomenon, the most likely cause, and your first diagnostic step.

9. **(Calculation)** DPO with $\beta = 1$. Log-probabilities under the reference: preferred −2, rejected −3. Under the policy: preferred −4, rejected −9. Compute the loss. Is the model better at producing the preferred answer than before?

10. **(Concept)** What is the role of the reference model in DPO, and what does a larger $\beta$ do?

11. **(Design)** You have 5 000 pairs of answers ranked by users for *politeness*. After DPO the model is more polite. Name two properties you must measure before deploying it, and why this module says so.

12. **(Critical thinking)** "We trained with RL on a reward from an automatic judge and the judge's score improved by 20 %." What can and cannot be concluded?

## Mastery criteria

- [ ] Derive the policy gradient and explain baselines and advantages.
- [ ] Implement a REINFORCE step for a language model and relate it to masked cross-entropy.
- [ ] Explain why and when RL is preferable to SFT, and why it is less efficient.
- [ ] Recognise reward hacking and likelihood displacement in measurements, and name countermeasures.
- [ ] Write the DPO loss and compute it for given log-probabilities.
- [ ] Complete **Milestone 4**: a base model turned into an assistant, with controls and an evaluation the training never saw.
- [ ] At least 10 of 12 quiz questions correct.
