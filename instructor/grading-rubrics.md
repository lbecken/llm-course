# Grading rubrics

Five rubrics cover everything assessed in the course. Each has four levels. **Proficient**
is the standard for acceptance; *exemplary* describes work that could be used as a model
for others. For self-study, grade yourself row by row, in writing, and revisit any row
below proficient before moving on. For a taught course, give the rubric to learners
before they start the work.

Levels: **not yet** (the element is missing or wrong) · **developing** (present but with
errors or gaps that affect the conclusion) · **proficient** (correct and complete for the
purpose) · **exemplary** (also anticipates objections, quantifies uncertainty, or
generalises correctly).

## Rubric 1: conceptual explanations

Used for written explanations, end-of-part "explain" questions, and the architectural
explanation of Capstone A.

| Criterion | Not yet | Developing | Proficient | Exemplary |
|---|---|---|---|---|
| Correctness | contains false statements about the mechanism | correct in outline, wrong in a detail that matters (a shape, a direction of dependence) | every statement correct | also states the conditions under which it would be false |
| Mechanism, not vocabulary | names components without saying what they compute | describes operations but not why they are there | says what is computed, from what, and why it is needed | derives the behaviour from the computation (e.g. causality from the mask) |
| Concreteness | no example | an example that does not exercise the mechanism | a small numerical or shape-level example | a worked example that would fail if the explanation were wrong |
| Grounding in code | no reference to an implementation | cites a file | cites the lines and reads them correctly | notes where the implementation differs from the textbook form and why |
| Limits | none stated | generic caveat | states what the explanation does not cover | identifies an observation that the explanation cannot account for |

## Rubric 2: code

Used for coding exercises, milestone code and capstone code.

| Criterion | Not yet | Developing | Proficient | Exemplary |
|---|---|---|---|---|
| Correctness | does not run or gives wrong results | works on the example only | correct, with a test that would fail if it were not | tested against an independent reference (a naive version, nanochat, a by-hand case) |
| Tests | none | assert that code runs | check behaviour, including one edge case | include a negative test (the bug is injected and caught) |
| Clarity | cannot be followed | works but needs its author to explain | readable; shapes and invariants stated where not obvious | another learner could extend it without asking |
| Reproducibility | unseeded, undocumented | seeded, command missing | one command, seeded, dependencies stated | a manifest records configuration, revision and environment |
| Safety (Part V) | executes model output; unbounded loops | limits exist but can be bypassed by the model | explicit budgets, validation and permissions in code, tested | threat model written; an attack demonstrated and stopped |

## Rubric 3: experiments

Used for every lab write-up, milestone experiment and capstone experiment.

| Criterion | Not yet | Developing | Proficient | Exemplary |
|---|---|---|---|---|
| Question and prediction | none | question stated after the fact | question, and a prediction written before the run | prediction with a stated result that would refute it |
| Design | several things changed at once | one variable, but an uncontrolled difference remains | one independent variable, controls listed | includes the control that rules out the most plausible alternative explanation |
| Measurement | wrong metric, or a metric computed incorrectly | right metric, no uncertainty | uncertainty stated (seeds, sample size, run-to-run variation) | paired or otherwise efficient comparison; noise measured, not assumed |
| Labelling | measured, reported and estimated numbers mixed | mostly labelled | every number labelled, with hardware and date | separates what was verified from what is inferred, sentence by sentence |
| Interpretation | conclusion not supported by the data | supported, but overstated | conclusion matches the evidence and its scope | says what the experiment cannot show and what would |
| Negative results | hidden | mentioned | reported with the same care as positive ones | analysed for what they rule out |

## Rubric 4: benchmarks and performance claims

Used for Milestone 3 and Capstone B.

| Criterion | Not yet | Developing | Proficient | Exemplary |
|---|---|---|---|---|
| Correctness column | absent | present for some rows | every optimisation has an equality or quality check | a case where the check fails is found and explained |
| Method | cold start, no synchronisation, single run | warm-up and repeats, but conditions unstated | warm-up, synchronisation, repeats, all conditions stated | separates phases (prefill/decode), reports distribution not only mean |
| Baseline | none, or an unfair one | fair but untuned | the strongest simple baseline at equal conditions | several baselines; shows where the optimisation loses |
| Resource analysis | none | names a resource | identifies the binding resource with a bound or estimate | prediction from the bound compared with measurement |
| Scope | implies generality | hardware stated | states hardware, sizes and where the result should not be extrapolated | tests one extrapolation |

## Rubric 5: agent and system design

Used for Milestone 5 and Capstone C.

| Criterion | Not yet | Developing | Proficient | Exemplary |
|---|---|---|---|---|
| Structure | the model controls the loop | loop in code, success judged by the model | loop, validation, budgets and success test in code | simplest structure that works, with the alternatives considered |
| Trust boundaries | none identified | identified informally | every source of text classified; permissions per task; untrusted text never authorises actions | injection attempted against the system and the result reported |
| Observability | final answers only | logs without structure | trace per step with outcome labels | traces used to diagnose and fix a failure |
| Evaluation | anecdotes | success rate | success rate, outcome distribution, cost, equal-cost baseline | variance over repetitions; analysis by task difficulty |
| Honesty about simulation | simulated results presented as real | labelled in places | every simulated component and number labelled | a real model used where it matters, with its failures reported |

## Using the rubrics

- A milestone or capstone is **accepted** when every row of each applicable rubric is at least proficient.
- When a row is below proficient, the feedback is the cell to the right: it says what is missing.
- Rubric 3's "labelling" row and Rubric 5's "honesty about simulation" row are not negotiable. Work that presents estimated or simulated numbers as measurements is returned regardless of its other qualities. This mirrors the rule the course material itself follows.
