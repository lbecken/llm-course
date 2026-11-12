# Module 21 — Knowledge check

Closed book, about 25 minutes. Solutions: [`solutions/21-full-system.md`](../../solutions/21-full-system.md).

1. **(Concept)** List the stages of the life cycle of a language-model system in order, and the artefact each hands to the next.

2. **(Concept)** Which two artefacts must be identical in every stage from training to serving? What is the typical symptom when one of them is not?

3. **(Diagnosis)** A deployed assistant answers fluently but never stops generating. Name two upstream causes in different stages.

4. **(Concept)** Which resource limits each of: pretraining, prefill, decode at batch 1, an agent task of 30 steps?

5. **(Calculation)** A pipeline's stages take 2 h (data and tokenizer), 40 h (pretraining), 3 h (fine-tuning), 1 h (evaluation). Doubling pretraining speed and doubling fine-tuning speed each cost a week of work. What does each save, as a fraction of the total?

6. **(Concept)** Distinguish reproducible from replicable. Which one can be checked automatically, and how did the lab check it?

7. **(Concept)** Name six things a run manifest should record.

8. **(Design)** Why should a serving system log the rendered token sequence and not only the user's text?

9. **(Diagnosis)** An evaluation shows 92 % accuracy; users report far worse. Give three explanations located in different stages.

10. **(Code reading)** What are the three kinds of file in a nanochat checkpoint directory, and what does `chat_sft.py` take from the metadata file?

11. **(Design)** Translate "8 GPUs, device batch 16, FP8" from nanochat's speedrun into a plan for one GPU without FP8 support. What stays the same and what changes?

12. **(Critical thinking)** "It ran without errors, so the pipeline works." Give two counter-examples from this course.

## Mastery criteria

- [ ] Draw the full life cycle with artefacts and interfaces from memory.
- [ ] Find a pipeline's bottleneck by measurement and name the binding resource per stage.
- [ ] Produce a manifest that makes a run reproducible and comparable.
- [ ] Specify logs and traces for training, serving, tools and agents.
- [ ] Trace an end-of-pipeline symptom to candidate upstream causes.
- [ ] At least 10 of 12 quiz questions correct.
