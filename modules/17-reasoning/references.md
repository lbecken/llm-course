# Module 17 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Nye et al., [*Show Your Work: Scratchpads for Intermediate Computation with Language Models*](https://arxiv.org/abs/2112.00114) (2021) | paper | intermediate | The experiment of D.1 at larger scale: Transformers trained to emit intermediate computation solve algorithmic tasks they fail to solve directly. |
| Wei et al., [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903), NeurIPS 2022 | paper | introductory | Shows that large pretrained models can be *prompted* to produce intermediate steps, with large gains on multi-step problems. Read it for the phenomenon; the explanation in terms of serial computation is in the next entry. |
| Feng et al., [*Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective*](https://arxiv.org/abs/2305.15408), NeurIPS 2023 | paper | advanced | Proves that bounded-depth Transformers cannot solve certain arithmetic and equation problems directly but can with generated intermediate steps: the formal version of section C. |
| Wang et al., [*Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171), ICLR 2023 | paper | introductory | Majority voting over sampled reasoning chains. Note that its gains are reported for long sampled chains, the situation D.4 identifies as favourable. |
| Yao et al., [*Tree of Thoughts: Deliberate Problem Solving with Large Language Models*](https://arxiv.org/abs/2305.10601), NeurIPS 2023 | paper | intermediate | Search over partial solutions with model-based evaluation (D.6), with an honest account of its cost. |
| Turpin et al., [*Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*](https://arxiv.org/abs/2305.04388), NeurIPS 2023; Lanham et al., [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702) (2023) | papers | intermediate | Evidence for D.2: answers influenced by cues the stated reasoning omits, and answers that survive corruption of the reasoning. Essential before treating reasoning text as explanation. |
| Snell et al., [*Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*](https://arxiv.org/abs/2408.03314) (2024) | paper | advanced | A systematic study of how to spend a fixed inference budget across sampling, verification and search, and when that beats a larger model: the equal-cost viewpoint of D.5 on real models. |
| DeepSeek-AI, [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948) (2025) | paper | advanced | A published account of training a reasoning model with reinforcement learning on final-answer rewards (D.7), including what emerged without supervision of the reasoning itself. |

Historical versus current: scratchpads (2021) and self-consistency (2022) are stable
ideas. Which inference-time strategy is best for current models changes with each
generation; models trained for reasoning have absorbed much of what prompting strategies
used to add. Re-measure with the checklist of D.8 rather than trusting older results.
