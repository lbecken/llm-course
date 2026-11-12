# Module 19 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Wu et al., [*AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*](https://arxiv.org/abs/2308.08155) (2023) | paper | intermediate | A representative multi-agent framework: roles as conversable agents, orchestration patterns, example applications. Read it for the design vocabulary, and apply D.3's baseline questions to its results. |
| Du et al., [*Improving Factuality and Reasoning in Language Models through Multiagent Debate*](https://arxiv.org/abs/2305.14325), ICML 2024 | paper | intermediate | The debate pattern with reported gains. A good exercise: identify the budget of the single-agent baseline and whether the debaters are independent. |
| Cemri et al., [*Why Do Multi-Agent LLM Systems Fail?*](https://arxiv.org/abs/2503.13657) (2025) | paper | intermediate | An empirical taxonomy of failures in multi-agent systems (specification problems, inter-agent misalignment, verification failures) from annotated traces. The systematic version of D.7. |
| Kapoor et al., [*AI Agents That Matter*](https://arxiv.org/abs/2407.01502) (2024) | paper | intermediate | Shows on public benchmarks that simple baselines often match complex agent architectures once cost is controlled: the empirical case for D.3. |
| Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (2024) | article | introductory | Workflow patterns (chaining, routing, parallelisation, orchestrator–workers, evaluator–optimizer) with advice on when each is warranted. Matches D.1's table and D.8's ordering. |
| Saltzer and Schroeder, [*The Protection of Information in Computer Systems*](https://www.cs.virginia.edu/~evans/cs551/saltzer/), Proceedings of the IEEE 63(9), 1975 | paper | intermediate | The origin of least privilege and separation of privilege. D.7's security row and D.8's last bullet are these principles applied to agents; the paper is older than every model in this course and entirely current. |
| Panickssery et al., [*LLM Evaluators Recognize and Favor Their Own Generations*](https://arxiv.org/abs/2404.13076) (2024) | paper | intermediate | Evidence that models used as judges are biased towards their own outputs: an empirical counterpart of D.5's shared blind spots on real models. |

Historical versus current: orchestration frameworks appear and disappear quickly; none
is needed to understand or build the patterns here. The durable material is
methodological (equal-cost baselines, failure taxonomies) and the security principles of
1975.
