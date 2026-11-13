# Module 22 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/22-beyond-the-baseline.md`](../../solutions/22-beyond-the-baseline.md).

1. **(Concept)** List the seven questions this module asks of any proposed improvement.

2. **(Concept)** Which limitation of the baseline does retrieval-augmented generation address, and what changes: the model, or the system around it?

3. **(Interpretation)** After every fact changed, the closed-book model was right 0 % of the time and gave the old value 100 % of the time. Why is this a more serious failure than "it does not know"?

4. **(Interpretation)** The plain open-book model repeated an irrelevant passage's value in 100 % of cases. What does this imply about the relation between retrieval quality and answer quality?

5. **(Concept)** A model trained to abstain on irrelevant passages still repeats a wrong value attached to the right key. Why can no amount of such training fix this from the passage alone?

6. **(Method)** Name three things to evaluate separately in a retrieval-augmented system.

7. **(Concept)** What is a mixture-of-experts layer, and what is its intended advantage?

8. **(Interpretation)** nanochat's MoE experiment improved validation loss per step and was rejected. On what grounds? State the general lesson in one sentence.

9. **(Concept)** What does a state-space or recurrent alternative give up relative to attention, and why does that matter for the tool use and retrieval of this course?

10. **(Concept)** How does an image enter a Transformer? Name two things that do not change relative to a text-only model and one security consequence.

11. **(Concept)** What is distillation, and why can a student trained on a teacher's output distribution learn more per example than one trained on the data alone?

12. **(Critical thinking)** What does a negative result establish, and what does it not? Why keep it?

## Mastery criteria

- [ ] Apply the seven questions to an unfamiliar proposal.
- [ ] Explain retrieval, its benefit and its three characteristic failure conditions, with measurements.
- [ ] Explain why equal-FLOP and equal-time comparisons can disagree.
- [ ] Place long-context, multimodal and distillation methods relative to the baseline architecture.
- [ ] Read primary sources, including negative results, and state the scope of their evidence.
- [ ] At least 10 of 12 quiz questions correct, and the **final checklist of abilities** in `instructor/assessments.md` reviewed.
