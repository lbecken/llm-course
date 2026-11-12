# Module 16 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| nanochat, [`nanochat/engine.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py) (`use_calculator`, `RowState`, `Engine.generate`) and [`nanochat/execution.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/execution.py) | source code | intermediate | The primary source for section F: tool use as a state machine over sampled tokens, and a candidly documented evaluation sandbox. |
| Schick et al., [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761), NeurIPS 2023 | paper | intermediate | Tool calls as special text inside the generated sequence, with results inserted by the system: the design nanochat follows. Also shows how training data with tool calls can be produced automatically. |
| Cobbe et al., [*Training Verifiers to Solve Math Word Problems*](https://arxiv.org/abs/2110.14168) (GSM8K), 2021 | paper | introductory | The data set whose `<<12/60=0.2>>` calculator annotations nanochat converts into tool-call training data. Section 3 describes the annotations. |
| Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629), ICLR 2023 | paper | intermediate | The application-level pattern of alternating model output, action and observation; the basis of Module 18. Read section 2 for the formulation. |
| Greshake et al., [*Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173) (2023) | paper | intermediate | Establishes indirect prompt injection (through retrieved content and tool results) as a class of attack, with a taxonomy of consequences. The systematic version of lab part B4. |
| Willison, [*The lethal trifecta for AI agents: private data, untrusted content, and external communication*](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) (2025) | article | introductory | A short, practical statement of the risky combination named in D.7, with many documented incidents linked from the author's prompt-injection series. |
| OWASP, [*Top 10 for Large Language Model Applications*](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | reference | introductory | A maintained checklist of application-level risks (prompt injection, excessive agency, insecure output handling). Useful as a review list for Exercise 4. |
| Chen et al., [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) (HumanEval/Codex), 2021, section 2.3 | paper | intermediate | Why generated code must be executed in a sandbox to be evaluated, and how the authors isolated it; nanochat's `execution.py` cites the accompanying code. |

Historical versus current: the mechanism (request, execute, observe) is stable. The
*interfaces* by which hosted models express tool calls change frequently and differ
between providers; the security analysis does not depend on them. Prompt injection has
no general solution at the time of writing: treat any claim of one with the
evaluation discipline of Module 10.
