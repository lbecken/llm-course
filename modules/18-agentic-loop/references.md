# Module 18 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629), ICLR 2023 | paper | intermediate | The thought–action–observation loop used in this module, with experiments on when interleaved reasoning helps acting. |
| Russell and Norvig, [*Artificial Intelligence: A Modern Approach*](https://aima.cs.berkeley.edu/), 4th edition (2020), chapters 2–4 | book | introductory | The classical definition of agents, environments, and search. Everything in D.1–D.2 is this framework with a language model as one possible policy. Chapter 3 covers the breadth-first search the lab uses as its reference policy. |
| Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (2024) | article | introductory | A practitioner's taxonomy: fixed workflows versus model-directed loops, with the recommendation to use the simplest structure that works. Agrees with D.2's "least intelligent component". |
| Kapoor et al., [*AI Agents That Matter*](https://arxiv.org/abs/2407.01502) (2024) | paper | intermediate | A critique of agent evaluation practice: cost must be reported with accuracy, simple baselines are often as good, and benchmarks are easily overfitted. The methodological companion to D.8. |
| Jimenez et al., [*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*](https://arxiv.org/abs/2310.06770), ICLR 2024 | paper | intermediate | A widely used agent benchmark whose success test is code (the repository's tests): an example of D.1's first rule at scale. Read the task definition and evaluation sections. |
| Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), TACL 2024 | paper | intermediate | Evidence that models use information unevenly across a long context: background for D.6's advice to restate essentials rather than rely on a long history. |
| OWASP, [*Top 10 for Large Language Model Applications*](https://owasp.org/www-project-top-10-for-large-language-model-applications/), entry "Excessive Agency" | reference | introductory | The risk class that D.7's rules address: too much functionality, too many permissions, too much autonomy. |

Historical versus current: the loop itself is decades old and stable. Agent *frameworks*
and benchmarks change month to month, and reported success rates are highly sensitive to
budgets and evaluation details. Prefer primary descriptions of a loop you can read in
full (as here) to framework documentation.
