# Module 01 — Lab: rebuild, verify, and repurpose the MLP

Scripts: [`labs/foundations/01_digits_mlp.py`](../../labs/foundations/01_digits_mlp.py),
[`labs/foundations/01b_mlp_next_char.py`](../../labs/foundations/01b_mlp_next_char.py) ·
Code under study: [`mlle/foundations/mlp.py`](../../mlle/foundations/mlp.py) ·
Solutions: [`solutions/01-neural-network-refresher.md`](../../solutions/01-neural-network-refresher.md)

## G. Exercises

### Exercise 1 (mathematical): the gradient by hand

A 3-class classifier outputs logits $\mathbf{z} = [1, 3, 0]$ for an example whose correct class is 2 (the last one).

1. Compute the probabilities $\mathbf{q}$ and the loss.
2. Compute $\partial L / \partial \mathbf{z}$.
3. After one SGD step *on the logits themselves* with learning rate 1, what are the new logits and the new loss?
4. Without computing: if the correct class had been 1 instead, would the gradient have been larger or smaller in magnitude? Why?

**Success criterion:** your numbers match the solution to 3 decimals, and you can explain item 4 in one sentence.

### Exercise 2 (coding): break the backward pass, catch it with the check

1. Open `mlle/foundations/mlp.py`. In `backward`, change `dh_pre = dh * (1.0 - h**2)` to `dh_pre = dh * (1.0 - h)` (a plausible typo).
2. Predict: which of the four gradient checks in the lab script will now fail (`W1`, `b1`, `W2`, `b2`)?
3. Run `python labs/foundations/01_digits_mlp.py`. Record the reported differences. Does the network still train? How well?
4. Restore the line, and run `make test`.

**Success criterion:** you predicted the failing checks correctly *and* can explain why the network still learns something with a wrong gradient.

### Exercise 3 (experimental): capacity and noise

Run the digit lab over a small grid and fill in the table in section I.

```bash
for h in 2 4 8 32 128; do python labs/foundations/01_digits_mlp.py --hidden $h | tail -3; done
for p in 0.0 0.08 0.2 0.3 0.4; do python labs/foundations/01_digits_mlp.py --flip-prob $p | tail -3; done
```

Before running, predict: (a) the smallest hidden size that reaches 95 % test accuracy;
(b) the accuracy at `--flip-prob 0.4`; (c) whether train and test loss will differ more
for large or small hidden sizes.

**Success criterion:** a completed table and one sentence per prediction saying whether it held.

### Exercise 4 (experimental): context length in the window model

```bash
for k in 1 2 3 5 8; do python labs/foundations/01b_mlp_next_char.py --context $k | tail -6; done
```

Record the final validation loss and the count-based baseline for each `K`. At which `K`
does the MLP start to beat counting, if at all? Why might a neural model beat counting
for long contexts but not for short ones? (Module 02 gives the full answer; write your
hypothesis first.)

**Success criterion:** table filled; hypothesis written before reading Module 02.

### Challenge (optional): a second hidden layer

Extend `ManualMLP` to two hidden layers, by hand (no autograd). Add a gradient check for
the new parameters. Requirements: all finite-difference checks below 1e-6 in float64,
and agreement with autograd below 1e-9.

Hint: the backward pass repeats steps 3–4 of the existing one once more. The only new
idea is that the "input" of the second layer is the first layer's activation, and its
gradient is what the first layer receives.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Establish that a hand-written backward pass is correct, observe learning curves for the digit classifier, and observe the limits of a fixed-window language model |
| **Hypotheses** | (H1) analytic, numeric and autograd gradients agree to ~1e-8 or better in float64. (H2) Initial loss ≈ ln(number of classes). (H3) The window model's loss approaches the same-context count-based model and stays above the source entropy. |
| **Hardware / software** | Any CPU. Python ≥ 3.10, `pip install -e .` |
| **Tier** | A: each command finishes in seconds (lab 01b: about 20 s) |
| **Inputs** | Synthetic 5×7 digits (`mlle/data/digits.py`); Pocket corpus (`mlle/data/pocket.py`); both generated from fixed seeds |
| **Commands** | `python labs/foundations/01_digits_mlp.py` and `python labs/foundations/01b_mlp_next_char.py`, plus the loops in Exercises 3–4 |
| **Expected output** | Gradient-check table; loss/accuracy every 50 steps; final accuracy; for 01b the loss curve, two reference values and a text sample |
| **Metrics** | cross-entropy (nats), test accuracy, gradient discrepancy |
| **Independent variables** | hidden size, pixel-flip probability, context length `K` (one at a time) |
| **Controls** | seeds (data and init), number of steps, learning rate, batch size |
| **Interpretation** | Compare initial loss with ln(C). Compare train and test loss (gap = overfitting). In 01b compare against the count-based model (what the data supports at that context length) and the entropy floor (what any model could reach with unlimited context). |
| **Common failure modes** | Loss `nan` after raising `--lr` far above 1 (divergence: reduce it). Gradient check "fails" at 1e-3 in float32 (precision, not a bug: the script casts to float64 for this reason). Accuracy stuck at 10 % (all-zero init or a broken backward pass). |

## I. Results and interpretation

Reference measurements by the course author (CPU sandbox, 2026-11-11, seed 0) are in the
lesson, section E. Record your own here (copy to `results/runs/` if you want to keep them).

**Exercise 3: capacity**

| hidden | final train loss | final test loss | test acc (2000 fresh) | prediction held? |
|---:|---:|---:|---:|---|
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 32 | | | | |
| 128 | | | | |

Author's reference points: hidden 32 → 0.9925; hidden 4 → 0.913.

**Exercise 3: noise**

| flip prob | test acc | most common confusion |
|---:|---:|---|
| 0.0 | | |
| 0.08 | | |
| 0.2 | | |
| 0.3 | | |
| 0.4 | | |

**Exercise 4: context length**

| K | MLP val loss | count-based, same K | entropy floor |
|---:|---:|---:|---:|
| 1 | | | 0.3139 |
| 2 | | | 0.3139 |
| 3 | | | 0.3139 |
| 5 | | | 0.3139 |
| 8 | | | 0.3139 |

Author's reference point: K = 3 → MLP 0.4597, count-based 0.4381.

**Questions to answer in writing**

1. Which numbers above are measurements, and which are theoretical values?
2. With `--flip-prob 0.4`, is the remaining error the network's fault? What is the best any classifier could do when 40 % of pixels are flipped? (You do not need the exact number; give the argument.)
3. In Exercise 4, what happens to the *number of parameters* in $W_1$ as `K` grows? What happens to the number of distinct contexts a count-based model would need to store?
