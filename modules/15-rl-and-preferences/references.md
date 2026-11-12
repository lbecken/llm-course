# Module 15 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Williams, [*Simple statistical gradient-following algorithms for connectionist reinforcement learning*](https://link.springer.com/article/10.1007/BF00992696), Machine Learning 8, 1992 | paper | advanced | The original REINFORCE paper, including the baseline. Readers who trained networks in the 1990s will recognise the notation. |
| Sutton and Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition (2018), chapter 13 | book (free online) | intermediate | The standard derivation of policy-gradient methods, baselines and actor–critic. Chapter 13 is self-contained for the purposes of this module. |
| Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017) | paper | intermediate | The ratio-and-clip objective that nanochat's script omits. Read sections 2–3 to understand what is being left out and why on-policy single-step updates do not need it. |
| nanochat, [`scripts/chat_rl.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py) | source code | intermediate | The primary source for section F: REINFORCE on a chat model in about 300 lines, with its simplifications stated in the docstring. |
| Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), NeurIPS 2022 | paper | intermediate | The reward-model-plus-PPO pipeline (RLHF) in full: data collection, reward model, KL penalty. The conceptual background of D.7. |
| Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), NeurIPS 2023 | paper | intermediate | The derivation sketched in D.7, in full. Section 4 is the derivation; section 5 its interpretation. |
| Razin et al., [*Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*](https://arxiv.org/abs/2410.08847) (2024) | paper | advanced | Names and analyses the failure measured in D.8: the preferred response's probability decreasing during DPO, especially when preferred and rejected responses are similar. |
| Krakovna et al., [*Specification gaming: the flip side of AI ingenuity*](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) (DeepMind blog, 2020) | article | introductory | A catalogue of reward hacking across reinforcement learning, with the general argument that it is a property of optimisation under an imperfect objective. |
| Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300) (2024), section 4 | paper | advanced | Introduces GRPO, the algorithm nanochat's docstring refers to: group-relative advantages with PPO-style clipping and a KL term. |

Historical versus current: REINFORCE (1992) and PPO (2017) are stable foundations. The
practice of preference optimisation for language models is recent and changing: DPO has
many variants addressing the failure shown in this module. Treat specific recipes as
provisional and the measurement discipline as permanent.
