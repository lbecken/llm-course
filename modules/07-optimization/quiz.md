# Module 07 — Knowledge check

Closed book, about 35 minutes. Solutions: [`solutions/07-optimization.md`](../../solutions/07-optimization.md).

1. **(Concept)** Through which two routes does the gradient arriving at the output of a residual block continue backwards? Why does this matter for deep models?

2. **(Derivation)** For one row of attention with weights $\mathbf{a}$ and upstream gradient $\mathbf{g}_A$ on those weights, the gradient on the scores is $g_{S,j} = a_j(g_{A,j} - \sum_k a_k g_{A,k})$. Use it to show that a perfectly one-hot attention row passes no gradient to its scores.

3. **(Calculation)** Adam, first step: $g_1 = -0.004$, $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon \approx 0$. What is the update? What if $g_1 = -400$?

4. **(Concept)** What is the difference between L2 regularisation added to the gradient and AdamW's decoupled weight decay, and why does it matter under Adam?

5. **(Concept)** In one sentence each: what does Muon do to an update matrix, how does it avoid computing an SVD, and why is it not used for embedding tables?

6. **(Code reading)** nanochat sets the embedding learning rate to 0.3 and the output-head learning rate to 0.008 (before width scaling). Give a reason these differ by a factor of almost 40.

7. **(Schedules)** Sketch nanochat's learning-rate multiplier over a run of 1 000 steps with default settings (`warmup_steps=40`, `warmdown_ratio=0.65`, `final_lr_frac=0.05`). At which step does the decay start? What is the multiplier at step 0?

8. **(Initialisation)** nanochat initialises `attn.c_proj` and `mlp.c_proj` to zero. What is the model's function at step 0, and what is its loss?

9. **(Diagnosis)** A character-level run on Pocket shows validation loss 2.75 from step 200 onwards. Another shows 3.22 forever. Diagnose each.

10. **(Experimental reasoning)** A blog post reports "optimizer X reaches lower loss than AdamW with the default learning rate 3e-4". List what you would need to know before believing that X is better.

11. **(Stability)** Name three mechanisms in nanochat that bound activations or scores, and the failure each prevents.

12. **(Synthesis)** List, in order, everything that happens in one training step of nanochat's `base_train.py`, from obtaining data to the weights having changed.

## Mastery criteria

- [ ] Compute Adam updates by hand and explain bias correction.
- [ ] Explain what Muon changes about a matrix update and which parameters it applies to in nanochat.
- [ ] Explain warm-up, decay, weight decay, clipping and zero-initialised projections by the failure each prevents.
- [ ] Train MiniGPT to within 0.06 nats/char of the entropy floor and explain the remaining gap (**Milestone 1**).
- [ ] Recognise uniform and unigram collapse from a loss value.
- [ ] At least 10 of 12 quiz questions correct.
