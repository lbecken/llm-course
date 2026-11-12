# Module 10 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/10-evaluation-and-generation.md`](../../solutions/10-evaluation-and-generation.md).

1. **(Calculation)** A model's summed loss over an evaluation text is 1.2 × 10⁶ nats; the text is 2.5 × 10⁶ bytes and 6.0 × 10⁵ tokens. Give nats per token, perplexity and bits per byte.

2. **(Concept)** Why does `evaluate_bpb` give special tokens a byte length of 0 and also drop their loss?

3. **(Concept)** Describe how one multiple-choice CORE item is scored. Why does no text need to be generated?

4. **(Calculation)** A 4-way multiple-choice task has accuracy 0.40. What is its centred accuracy? What would 0.25 give?

5. **(Critical reading)** In nanochat's categorical chat evaluation, logits are restricted to the answer letters. Name one thing this measures less strictly than free generation would.

6. **(Calculation)** Logits $(0, 0, \ln 2)$. Give the probabilities at temperature 1 and at temperature 0.5.

7. **(Concept)** Top-k with $k=40$ and top-p with $p=0.9$: in which situation does top-p keep far fewer than 40 tokens, and in which far more?

8. **(Diagnosis)** A deployed model repeats the same sentence endlessly. Give the most likely cause and two different fixes.

9. **(Concept)** Give two reasons a language model produces fluent false statements, one that lowering the temperature mitigates and one that it does not.

10. **(Calculation)** Model X answers 130 of 200 questions correctly. Give the accuracy, its standard error and an approximate 95 % interval. Model Y scores 142/200 on *different* questions. Is Y better?

11. **(Method)** The same question as 10, but X and Y were tested on the *same* 200 questions. What additional information do you need, and why could the conclusion change?

12. **(Design)** You want to claim that a new activation function improves nanochat. List the minimum set of runs and controls.

## Mastery criteria

- [ ] Convert between loss units and state when two losses are comparable.
- [ ] Explain CORE's three task formats and centring.
- [ ] Implement temperature, top-k and top-p sampling and predict their effects.
- [ ] Compute a standard error and a bootstrap interval, and run a paired comparison.
- [ ] Complete **Milestone 2** with a record that states its own uncertainty.
- [ ] At least 10 of 12 quiz questions correct.
