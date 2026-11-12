# Module 17 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/17-reasoning.md`](../../solutions/17-reasoning.md).

1. **(Concept)** Why can a Transformer with a fixed number of layers fail to compute, in one forward pass, an answer that it computes easily when allowed to generate intermediate tokens?

2. **(Concept)** Name the four distinct things this module separates under the word "reasoning", and say for each whether the model's weights change.

3. **(Interpretation)** Two identical networks are trained on the same problems; the one trained to answer directly is at chance even on one-step problems, the one trained with worked steps is perfect. A third, trained directly on one- and two-step problems only, solves one-step problems perfectly. What do the three results together show?

4. **(Calculation)** A scratchpad format rewrites the remaining expression after each of $n$ operations (operands and results are single characters, steps separated by one character). How many characters are generated for $n = 4$?

5. **(Calculation)** Samples are right with probability 0.7, independently; wrong samples are all different from each other. What is the probability that the most common answer among 3 samples is the right one? (Count the case of three different answers as a failure.)

6. **(Concept)** Give the condition under which majority voting returns a wrong answer with high confidence. Which strategies of this module fail in the same situation, and what is needed to escape it?

7. **(Interpretation)** A 25-sample vote reaches the same accuracy as greedy decoding. Is self-consistency useless for this model? Name something the samples provide that greedy decoding does not.

8. **(Concept)** Compare a model used as a checker with an exact checker in verify-and-retry. What bounds the benefit in each case?

9. **(Method)** List three requirements for a fair comparison between a reasoning strategy and a baseline.

10. **(Concept)** What must exist for tree search over reasoning steps to beat independent sampling? Give one domain where it exists and one where it usually does not.

11. **(Concept)** How does reinforcement learning with a final-answer reward shape a model's reasoning text, and what risk from Module 15 carries over?

12. **(Critical thinking)** A model answers a question and, when asked, explains its answer step by step. A colleague cites the explanation as proof of how the model arrived at the answer. Give two reasons for caution.

## Mastery criteria

- [ ] Explain intermediate tokens as additional serial computation, and demonstrate it with a trained model.
- [ ] Implement voting, verify-and-retry and decomposition, and report accuracy with cost.
- [ ] Predict from the nature of a model's errors which strategy can help.
- [ ] Use sample agreement as a confidence estimate and design an abstention policy.
- [ ] Evaluate a reasoning claim with the checklist of D.8.
- [ ] At least 10 of 12 quiz questions correct.
