# Module 19 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/19-multi-agent.md`](../../solutions/19-multi-agent.md).

1. **(Concept)** Define role, message, shared state and orchestrator. Which of them involve a model?

2. **(Concept)** "The planner tells the executor what to do." Restate this sentence in terms of function calls and prompts.

3. **(Method)** A paper shows a three-agent system outperforming "the base model". Name the two baselines that are missing.

4. **(Interpretation)** In the lab the team beat a single agent that used the same number of calls for majority voting. What produced the gain, and how would you prove it is not the multi-agent structure?

5. **(Concept)** Which reasons for division of labour among people transfer to copies of one model, and which do not?

6. **(Calculation)** A solver is right 60 % of the time. A reviewer approves right answers with probability 0.95 and wrong ones with probability 0.25. What fraction of approved answers is wrong?

7. **(Interpretation)** With half-systematic errors, a same-model reviewer and an independent reviewer gave nearly the same final accuracy, but 37 % versus 6 % wrongly approved answers. Why is accuracy the same, and why does the difference matter?

8. **(Concept)** Name four failure modes that exist in a team and not in a single call.

9. **(Design)** Why should messages between agents be validated in code where possible? Give the lab's evidence.

10. **(Security)** Agent A reads web pages; agent B can send email. A passes summaries to B. Describe the attack and one structural defence.

11. **(Design)** Give three legitimate reasons to use more than one agent.

12. **(Critical thinking)** A system uses the same model as "author" and as "critic", and reports that the critic approved 98 % of outputs. What does this number tell you about quality?

## Mastery criteria

- [ ] Implement a planner–executor–reviewer protocol with a complete message log.
- [ ] Choose and run the fair baselines for a multi-agent claim.
- [ ] Explain and measure shared blind spots; distinguish accuracy from wrong approvals.
- [ ] Validate inter-agent messages and name team-specific failure modes.
- [ ] Decide, for a concrete task, between a single call, a workflow, a loop and a team.
- [ ] At least 10 of 12 quiz questions correct.
