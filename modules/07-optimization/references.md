# Module 07 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Kingma & Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980), ICLR 2015 | paper | intermediate | The Adam update, bias correction and its invariance to gradient scale (Algorithm 1 and section 2). |
| Loshchilov & Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), ICLR 2019 | paper | intermediate | AdamW: why L2-in-the-gradient and weight decay differ under adaptive optimizers. Cited in nanochat's `optim.py`. |
| Keller Jordan, [*Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/) (2024) | blog post | intermediate | The primary description of Muon: orthogonalised momentum, the Newton–Schulz coefficients used in this course, and guidance on which parameters to apply it to. |
| Amsel, Persson, Musco, Gower, [*The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*](https://arxiv.org/abs/2505.16932) (2025) | paper | advanced | The orthogonalisation iteration nanochat uses in place of plain Newton–Schulz; source of the coefficient table in `optim.py`. |
| Liang, Chen, Liu, Liu, [*Cautious Optimizers: Improving Training with One Line of Code*](https://arxiv.org/abs/2411.16085) (2024) | paper | intermediate | The sign-agreement masking idea behind nanochat's cautious weight decay. |
| Andrej Karpathy, [*Building makemore Part 3: Activations & Gradients, BatchNorm*](https://www.youtube.com/watch?v=P6sfmUTpUmc) (2022) | video | introductory–intermediate | Initialisation, activation statistics, and how to read the health of a network from histograms of activations and gradients. |
| Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/) (2019) | blog post | introductory | A practitioner's checklist: verify the initial loss, overfit one batch, add complexity one piece at a time. The methodological companion to this module. |
| Glorot & Bengio, [*Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html), AISTATS 2010 | paper | intermediate | The variance-preservation argument for initialisation. |
| nanochat, [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LOG.md), entries 2026-01-08 (gradient clipping), 2026-01-10 (Muon upgrades, cautious weight decay), 2026-01-19–22 (optimizer hyperparameter sweep), 2026-02-05 (batch-size scaling) | engineering log | intermediate | The experiments behind the pinned optimizer configuration, including what did not help. |

Historical versus current: SGD with momentum (1980s–2010s) and Adam (2015) are the
foundations; AdamW (2019) has been the default for Transformers for years. Muon (2024)
and its refinements are recent, are used by nanochat and by speed-running code bases,
and their behaviour at very large scale is an active research topic rather than settled
practice.
