# Module 13 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740), ICLR 2018 | paper | intermediate | The origin of the rule in D.2: half-precision computation, full-precision master weights, loss scaling. Short and still accurate. |
| PyTorch, [*Automatic Mixed Precision*](https://pytorch.org/docs/stable/amp.html) and [*CUDA Automatic Mixed Precision examples*](https://pytorch.org/docs/stable/notes/amp_examples.html) | documentation | introductory | `autocast` and `GradScaler` as most code bases use them, including gradient accumulation with a scaler. nanochat's explicit casts replace `autocast`; read these to recognise the conventional form. |
| Kalamkar et al., [*A Study of BFLOAT16 for Deep Learning Training*](https://arxiv.org/abs/1905.12322) (2019) | paper | intermediate | Evidence that bf16 computation with fp32 accumulation trains a range of models without loss scaling: the empirical basis for bf16 as the default. |
| Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) (2022) | paper | advanced | Defines the e4m3 and e5m2 formats and their intended use for weights/activations versus gradients, as used in `nanochat/fp8.py`. |
| nanochat, [`nanochat/fp8.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/fp8.py) (module docstring) | source code | advanced | A 60-line explanation of FP8 training with dynamic per-tensor scaling, more direct than most write-ups. |
| Chen, Xu, Zhang, Guestrin, [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (2016) | paper | intermediate | The analysis behind activation checkpointing: trading recomputation for memory, and how far the trade can be pushed. |
| PyTorch, [*Introduction to `torch.compile`*](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | tutorial | introductory | What is traced, what triggers recompilation, how to diagnose slow compilation. |
| Chowdhery et al., [*PaLM: Scaling Language Modeling with Pathways*](https://arxiv.org/abs/2204.02311) (2022), Appendix B | paper | intermediate | Defines model FLOPs utilisation (MFU) and argues for it over hardware utilisation; nanochat's FLOP estimate cites this paper. |
| Rajbhandari, Rasley, Ruwase, He, [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054), SC 2020 | paper | advanced | Sharding optimizer state and gradients across data-parallel ranks; nanochat's optimizer describes itself as "ZeRO-2 style". |
| PyTorch, [*Distributed Data Parallel*](https://pytorch.org/docs/stable/notes/ddp.html) | documentation | intermediate | The standard mechanism nanochat chose *not* to use. Understanding it makes nanochat's optimizer-level synchronisation easier to read. |

Historical versus current: the mixed-precision paper (2018) and checkpointing (2016) are
unchanged foundations. FP8 training and compiler stacks are recent and move quickly:
check the versions pinned in nanochat's lockfile before relying on any behaviour
described in a tutorial.
