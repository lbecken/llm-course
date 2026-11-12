# Module 07 — Lab: train a small GPT and understand its learning curve

Script: [`labs/foundations/07_train_minigpt.py`](../../labs/foundations/07_train_minigpt.py) ·
Code under study: [`mlle/foundations/optim.py`](../../mlle/foundations/optim.py), [`mlle/foundations/train.py`](../../mlle/foundations/train.py) ·
Solutions: [`solutions/07-optimization.md`](../../solutions/07-optimization.md)

This lab is the **Part I milestone**. Keep the records it writes to `results/runs/minigpt.jsonl`.

## G. Exercises

### Exercise 1 (mathematical): Adam by hand

One scalar parameter $\theta_0 = 1.0$, $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 0$, no weight decay.
Gradients arrive as $g_1 = 2$, $g_2 = -2$, $g_3 = 2$.

1. Compute $m_t$, $v_t$, $\hat m_t$, $\hat v_t$ and $\theta_t$ for $t = 1, 2, 3$.
2. Compare the size of the three updates. What does Adam do when gradients disagree in sign?
3. What would SGD with the same $\eta$ have done over these three steps?
4. Check with `mlle.foundations.optim.AdamW` (set `weight_decay=0`, `eps=0` is not safe in general but works here).

**Success criterion:** your $\theta_3$ matches the code to 4 decimals.

### Exercise 2 (experimental): the milestone run

```bash
python labs/foundations/07_train_minigpt.py
python labs/foundations/07_train_minigpt.py --style gpt2
```

1. Record for each: parameters, final validation loss, gap to the entropy floor, both agreement accuracies, tokens/s.
2. Explain the first loss value printed (step 0) for each style.
3. The gap to the floor is about 0.03 nats. Give two reasons it is not zero that have nothing to do with the model being too small.
4. Generate with the trained model (the script prints a sample). Find one sentence in your sample with a relative clause and check its agreement by hand.

**Success criterion (defaults):** validation loss within 0.06 of the floor; agreement with attractors ≥ 0.95.

### Exercise 3 (experimental): the learning-rate landscape

Predict first: at which learning rate will the GPT-2-style model fail, and what will its
loss be when it does?

```bash
for lr in 1e-4 1e-3 1e-2 1e-1 3e-1; do
  python labs/foundations/07_train_minigpt.py --steps 600 --lr $lr --sentences 10000 | grep "validation loss"
done
for lr in 3e-3 3e-1; do
  python labs/foundations/07_train_minigpt.py --steps 600 --lr $lr --sentences 10000 --style gpt2 | grep "validation loss"
done
```

Fill in the table in section I and compare with the lesson's table (D.6).

**Success criterion:** you observe the collapse of the GPT-2 style at 0.3 and can identify the value it collapses to.

### Exercise 4 (experimental): comparing optimizers fairly

```bash
python labs/foundations/07_train_minigpt.py --steps 600 --sentences 10000 --optimizer muon
python labs/foundations/07_train_minigpt.py --steps 600 --sentences 10000 --lr 3e-3
python labs/foundations/07_train_minigpt.py --steps 600 --sentences 10000 --lr 1e-2
```

Run each with `--seed 0`, `1`, `2`. Then answer: does Muon beat AdamW on this problem?
State precisely which comparison supports your answer.

**Success criterion:** a 3×3 table (configuration × seed) and a conclusion that mentions tuning.

### Exercise 5 (coding): see the gradient's path

Write a script that builds an untrained `MiniGPT` (GPT-2 style, 6 layers, `n_embd=64`),
runs one forward/backward pass on a random batch, and prints the gradient norm of
`block.attn.c_proj.weight` for each layer. Then repeat with residual connections removed
(patch `Block.forward` as in Module 06's lab). Which layers receive gradient in each
case? Relate the result to the "identity path" of lesson D.2.

**Success criterion:** two columns of six numbers and a two-sentence explanation.

### Challenge (optional): learning-rate warm-up in isolation

Using `TrainConfig(warmup_steps=...)`, compare warm-up of 1, 20 and 100 steps at
learning rates 3e-3 and 3e-2, three seeds each, 600 steps, GPT-2 style and nanochat
style. For which (style, rate) combination does warm-up matter? Explain with the
bias-corrected Adam update at $t=1$ and the initial state of the model.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Train a complete small causal Transformer; measure its loss against the known optimum; characterise sensitivity to learning rate, optimizer and initialisation |
| **Hypotheses** | (H1) The default run ends within 0.06 nats/char of the entropy floor and reaches ≥ 0.95 agreement with attractors. (H2) Final loss is flat over a wide range of learning rates and degrades on both sides. (H3) An excessive learning rate collapses the GPT-2-style model to the unigram loss but not the nanochat-style model. (H4) Muon and AdamW are indistinguishable on this task once both are tuned. |
| **Hardware / software** | CPU (or `--device cuda`); course package |
| **Tier** | A: default run ≈ 35 s on 4 cores; 600-step runs ≈ 15 s |
| **Inputs** | Pocket corpus, 20 000 sentences (default), character tokenizer, last 10 % held out |
| **Metrics** | validation loss (nats/char) on a fixed set of 20 batches; gap to floor 0.3139; agreement accuracy on 200 + 200 minimal pairs; tokens/s |
| **Independent variables** | learning rate; optimizer; style; seed (one at a time) |
| **Controls** | data, model size, steps, batch size, schedule shape, evaluation batches |
| **Interpretation** | Seed spread of the final loss is about ±0.003: differences below 0.005 need more seeds. A final loss near 2.75 is the unigram solution (context unused); near 3.22 is the uniform solution (nothing learned). |
| **Common failure modes** | Running several jobs at once on a small CPU multiplies run time. `--device cuda` needs a CUDA build of PyTorch. With `--text-file`, the entropy floor and agreement tests are not available (they are specific to Pocket). |

## I. Results and interpretation

Reference (author, CPU, 2026-11-12, seed 0; 600 steps, 10 000 sentences unless noted):

| Configuration | final val loss |
|---|---:|
| nanochat style, default run (1 500 steps, 20 000 sentences) | 0.3481 |
| GPT-2 style, default run | 0.3492 |
| nanochat style, lr 1e-4 / 1e-3 / 1e-2 / 1e-1 / 3e-1 | 1.672 / 0.373 / 0.350 / 0.358 / 0.392 |
| GPT-2 style, lr 3e-3 / 3e-1 | 0.366 / 2.751 |
| Muon 0.02 + AdamW 3e-3, seeds 0/1/2 | 0.3499 / 0.3487 / 0.3525 |
| AdamW 3e-3, seeds 0/1/2 | 0.3574 / 0.3553 / 0.3542 |

Your measurements:

| Configuration | seed | final val loss | gap to floor | agreement (attractor) | tokens/s |
|---|---:|---:|---:|---:|---:|
| | | | | | |

**Questions to answer in writing**

1. Which of H1–H4 did your data support?
2. Your tokens/s differs from the reference. Does that affect any conclusion above? Which kind of result is hardware-dependent and which is not?
3. Write three sentences for a colleague explaining what "collapsed to the unigram solution" means and how to recognise it from a loss value.
