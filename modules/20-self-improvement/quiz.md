# Module 20 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/20-self-improvement.md`](../../solutions/20-self-improvement.md).

1. **(Concept)** Name four levels at which a system can be said to "improve itself", and for each the measurement it depends on.

2. **(Concept)** List the five steps of one trial in the bounded experiment loop, and what happens after the last trial.

3. **(Concept)** Why must the loop measure seed-to-seed noise before accepting any change?

4. **(Calculation)** Thirty candidate configurations are in truth identical. A loop accepts every candidate that beats the best score so far. About how many acceptances do you expect? (The $i$-th of a sequence of independent draws is a record with probability $1/i$; $\sum_{i=2}^{31} 1/i \approx 3.0$.)

5. **(Interpretation)** A search over a parameter that only reseeds the run reported a validation gain of +0.0053 and a test change of −0.0058. Explain both numbers.

6. **(Concept)** What is the "winner's curse" in this setting, and which reported number is not affected by it?

7. **(Design)** Name five things that must be outside the reach of an improvement loop, and for one of them describe concretely what goes wrong otherwise.

8. **(Interpretation)** A proposer asks to evaluate on the training set. It was not built to cheat. Why does it propose this, and what does that imply for how constraints must be implemented?

9. **(Concept)** Why does the proposer require no trust in this design, even if it is a model that can be wrong or manipulated?

10. **(Critical reading)** nanochat's leaderboard shows validation bpb of 0.71854, 0.71808 and 0.71800 for three successive entries, with time-to-target falling from 2.02 h to 1.65 h. What can you conclude from the bpb column alone? What makes the overall claim credible?

11. **(Concept)** Training a model on its own outputs can improve it or degrade it. What decides which?

12. **(Critical thinking)** State two bottlenecks that a claim of open-ended recursive self-improvement must explain away.

## Mastery criteria

- [ ] Distinguish the levels of self-improvement and the measurement each needs.
- [ ] Implement and run a bounded experiment loop with a ledger.
- [ ] Demonstrate selection effects in adaptive search and apply the countermeasures.
- [ ] Specify what an improvement loop must not be able to change.
- [ ] Read an automated-research result critically.
- [ ] Complete **Milestone 5**: a tool-using agent with budgets, traces, an independent success test and an equal-cost baseline.
- [ ] At least 10 of 12 quiz questions correct.
