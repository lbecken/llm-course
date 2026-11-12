# Module 18 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/18-agentic-loop.md`](../../solutions/18-agentic-loop.md).

1. **(Concept)** Write the agent loop as six steps. Mark the one step in which a model is involved.

2. **(Concept)** Why must the success test be implemented outside the model? Give an example of what goes wrong otherwise.

3. **(Concept)** Name five distinct outcome labels an episode can end with in the course's loop, and say what each suggests fixing.

4. **(Calculation)** Steps fail fatally and independently with probability 0.1. What is the success probability of a 10-step task? Of a 30-step task?

5. **(Interpretation)** With a 20 % per-step mistake rate, measured success on 6-move tasks was 58 %, well above $0.8^6 = 26\,\%$. Explain.

6. **(Design)** Search solved every task in the number game with no model calls. State the conditions under which a search or planner should replace a model as the policy.

7. **(Concept)** What is the ReAct pattern, and how does its "thought" relate to Module 17?

8. **(Diagnosis)** An agent runs well for a dozen steps, then starts repeating an action that makes no sense for the current state. Name a likely cause related to context, and the standard remedy.

9. **(Interpretation)** Loosening all budgets raised success from 15 % to 58 % and introduced 46 episodes labelled `failed`. Is the loose configuration better? What does the new label mean?

10. **(Design)** List four things that must *not* be in an agent's action space, however capable its model.

11. **(Method)** Give four measurements, besides the success rate, that an agent evaluation should report.

12. **(Code reading)** In nanochat's `chat_cli.py`, the loop appends `<|assistant_end|>` when generation stopped without it. Which principle of this module does that illustrate? What does the script do about a conversation that outgrows the context window?

## Mastery criteria

- [ ] Write an agent loop with validation, budgets, named stop conditions and a trace.
- [ ] Diagnose an agent from its outcome distribution and traces.
- [ ] Compute compounded success probabilities and explain the effect of retries.
- [ ] Choose between search, rules and a model as policy for a given task.
- [ ] Manage context so that goal, legal actions and current state are always visible.
- [ ] State and justify the safety constraints on a loop that acts.
- [ ] At least 10 of 12 quiz questions correct.
