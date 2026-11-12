# Module 16 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/16-tool-use.md`](../../solutions/16-tool-use.md).

1. **(Concept)** Name the five steps of the tool-use cycle and say which component performs each.

2. **(Concept)** In nanochat's token-stream mechanism, what does it mean that tokens are "forced"? Why must the forced tokens also be fed through the model?

3. **(Concept)** Which parts of a tool-using assistant turn are training targets, and which are not? Give the reason for the part that is not.

4. **(Interpretation)** A tool-trained model scores 100 % with its calculator and 0 % without. A colleague concludes the model "forgot how to add". Give a better description of what happened.

5. **(Diagnosis)** A transcript shows `<|output_start|>2264<|output_end|>` after a calculator request for `23*87`. How can the application tell whether the calculator produced this? What training mistake makes the bad case likely?

6. **(Interpretation)** With a calculator that returns wrong results, the model's answers follow the tool 100 % of the time. Is this a defect of the model? What does it imply for system design?

7. **(Concept)** State three rules for handling a model's tool request at the application level before anything is executed.

8. **(Concept)** Define prompt injection. Why can it not be eliminated by training the model to distinguish instructions from data?

9. **(Design)** Give the permission class (read / write / dangerous) of: a calculator; `search_web`; `append_to_notes`; `send_email`; `execute_python`. For one of them, argue that the classification depends on context.

10. **(Critical reading)** nanochat's `use_calculator` checks that an expression contains only the characters `0123456789*+-/.() ` and no `**`, then calls `eval` with empty built-ins. Name one strength and one weakness of this design compared with evaluating a parsed syntax tree.

11. **(Analysis)** A tool-using model is evaluated on arithmetic with numbers twice as long as in training and scores 0 %, although the calculator is exact. Where is the error, and which metric would have revealed it?

12. **(Design)** An assistant can read the user's files and browse the web. Name the property that makes this combination risky, and two different mitigations.

## Mastery criteria

- [ ] Trace the token-stream tool mechanism by hand, including forced tokens and masks.
- [ ] Explain why tool results are masked in training, with the measured consequence of not doing so.
- [ ] Implement parsing, validation, permissions and budgets for an application-level tool loop.
- [ ] Evaluate a tool-using system by step (request, execution, use of result) and by condition (tool present, absent, wrong).
- [ ] Write a threat model for a tool set, including prompt injection through observations.
- [ ] At least 10 of 12 quiz questions correct.
