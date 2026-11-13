# Solutions — Module 22

## Lab exercises

### Exercise 1: the seven questions

1. **RAG over internal documents.** Limitation: the model's weights contain neither private nor current information and cannot cite. Changes: the system (an index, a retriever, a prompt layout), possibly fine-tuning to use context. Gain: answers grounded in documents; compare against the same model without retrieval and against a keyword search shown to the user, at equal latency budget. Scale: depends on corpus size and on retrieval recall, not on model size. Cost: indexing pipeline, access control per document, freshness, longer prompts. Evidence: retrieval recall@k; answer accuracy given correct, missing and misleading passages. New failures: irrelevant or wrong passages repeated as fact; injection through documents; leakage of documents a user may not see.
2. **GQA.** Limitation: KV-cache memory and bandwidth in decoding. Changes: architecture (fewer key/value heads); must be chosen at training time. Gain: cache bytes per token proportional to the number of key/value heads; compare quality at equal training budget and decode throughput at equal batch. Scale: Module 12 found no quality loss on a toy task; evidence at the target scale is needed. Cost: small; kernels support it. New failure: possible loss of capacity for tasks needing many distinct look-ups per position.
3. **nanochat's MoE experiment.** Limitation: every token uses every MLP parameter. Changes: architecture (router plus experts), optimizer handling, kernels. Gain claimed and measured: better loss per step at equal active FLOPs. At equal wall-clock time: worse (MFU about 46 % → 35 % at depth 18, reported upstream). Scale: approximately GPT-2 capability on 8 GPUs. Cost: substantial code, no FP8 path. Evidence: a negative result, reported with reasons. New failures: load imbalance, kernel constraints.

"30 % fewer FLOPs": ask for (a) wall-clock training time to the same validation loss on the same hardware with tuned kernels for both; (b) inference latency and throughput (prefill and decode separately) and memory; (c) the comparison at more than one scale and with seeds. FLOPs are one of three resources (Module 12 C), and often not the binding one.

### Exercise 2: weights versus context

Reference table: lesson D.2.

2. Without the shuffle the wrong keys are **y and z**, e.g. `y`: trained value 18, answered 64. The documents are generated grouped by key and the course's `split_tokens` holds out the *last* 10 % of the stream for validation, so every document of the last keys lands in the validation split and those facts are never trained on. (The author's first version of this lab had exactly this bug and reported 0.92.) In a larger system the same mistake, a split that is not random with respect to some grouping, produces a model that is "mysteriously" weak on one slice of the data, with nothing failing. The check: evaluate per group, and inspect what is in each split.
3. With probability about 1 in 100 (here somewhat more: the 26 values contain duplicates) the *other* key's value happens to equal the right one. The model is not right; the passage coincides. Counting such cases as correct slightly flatters retrieval systems in general: match the *source*, not only the string, when you can.
4. No. A statement "k=57" carries no mark of truth or falsehood; the model has nothing to compare it with (its weights hold only stale values, and it was trained to prefer the context). Detecting it needs information from outside the passage: a second independent source to cross-check, provenance and trust metadata for documents, signatures, or consistency checks against known constraints. That is a property of the system, not of the reader.
5. The closed-book rows change (its accuracy on the newest table stays at 0 and it still returns the first table's values); the open-book rows do not change (1.00 with a correct retriever on any table). The system's knowledge is in the index; the model contributes the ability to read.

### Exercise 3: a retriever that can fail

1. Expected and, within sampling noise, observed: plain open-book: accuracy ≈ $p$, abstention 0 (a wrong passage is repeated). Checking model: accuracy ≈ $p$, abstention ≈ $1-p$. Answer accuracy is bounded by retrieval accuracy for both; the difference is what happens in the remaining $1-p$: confident wrong answers versus explicit abstentions.
2. Accuracy rises to $1-(1-p)^3$ (0.999, 0.973, 0.875 for $p$ = 0.9, 0.7, 0.5) at an expected $1 + (1-p) + (1-p)^2$ retrieval calls. This is verify-and-retry (Module 17) with the model's relevance check as verifier, inside a bounded loop (Module 18). It works because abstention is a *reliable signal* here (1.00); with an unreliable one the calculation of Module 17, Exercise 1.5 applies.
3. None of the accuracy measurements on clean data detects it, and for the affected key the system is confidently wrong: the passage is relevant and well-formed. Controls: treat indexed content as untrusted (Module 16 D.6); restrict who can write to the index; record provenance and show sources with answers; cross-check high-stakes answers against a second source.

### Exercise 4: a negative result as evidence

1. Per step: validation loss improved. Per FLOP: by construction equal (iso-FLOP sizing of the experts). Per second: MFU from about 46 % to about 35 % at depth 18, net negative on wall-clock time. Also qualitative: code size, missing FP8 support.
2. Precision (FP8 unsupported for the grouped matmul; bf16 required): Module 13. Kernels and layout (grouped matmul, sorting, padding to alignment): Module 12. Optimizer (Muon on 3-D expert tensors, second-momentum buffer shape): Module 07. Scaling-law accounting (active versus total parameters): Module 09. Compilation (graph breaks checked): Module 13.
3. "At roughly GPT-2 scale on that hardware, with the kernels available in early 2026, a careful MoE implementation improved loss per step but lost on wall-clock time and added much complexity, so it was not adopted. It says nothing against MoE at larger scale or with a fused kernel."
4. A fused kernel that performs routing, dispatch and the matmuls in one operation ("like FlashAttention did for attention"). A negative result should be revisited when the stated bottleneck is removed: here, when such a kernel exists or the scale of interest changes. Recording the bottleneck is what makes the result reusable.
5. Open-ended. Typical gaps in positive claims: the unit of equality (question 3), the scale (4), variance over seeds (6), and failure modes (7).

### Exercise 5: your own extension

Open-ended; assessed with the capstone rubric
([`instructor/grading-rubrics.md`](../instructor/grading-rubrics.md)). A good design
states the effect size that would count *before* running, uses equal wall-clock time or
equal tokens as the unit, sizes runs from a measured throughput, plans at least three
seeds per arm, and commits to recording a negative outcome.

### Challenge: distillation in miniature

Outline. Student loss: `KL(softmax(teacher_logits / T) || softmax(student_logits / T))`
averaged over positions, with the teacher frozen. Expected, to be verified by your own
runs over three seeds: the distribution-matching student reaches a given validation loss
in fewer steps, because each position supplies a full distribution over the vocabulary
(for Pocket: which of several continuations are possible and how likely) instead of one
sampled token, i.e. far more bits per example; the final loss of both is bounded by the
student's capacity, and the distilled student cannot exceed the teacher. *The author has
not run this challenge; no reference values are given.*

### Written questions

1. "Open-book / fact about another key" (0.02 right, the irrelevant value repeated in 100 % of cases) and "wrong value for the right key" (repeated in 100 %, by the checking model too). Retrieval replaces one source of error (stale or missing knowledge) by another (the retriever and the corpus), and the model does not tell you which you got.
2. In each, text arrives in the context from a source the model treats as authoritative, and the model acts on it: copies the number, follows the instruction, states the claim. The model has no channel for verifying its context, so the integrity of whatever fills the context is a responsibility of the system: validation, provenance, permissions, independent checks.
3. Open-ended. A defensible answer for "still true": next-token prediction and cross-entropy as the training principle; the three-resource cost model; evaluation independent of optimisation with attention to noise; systems (model plus tools plus checks) as the unit of capability. For "obsolete": specific hyperparameters and architectural details of the pinned nanochat revision; today's benchmark numbers and hardware figures; the particular tool-call syntaxes and serving frameworks.

## Quiz

1. Which limitation does it address; what exactly changes; what gain, on what metric, against which baseline, at equal what; at what scale; at what cost in complexity and interactions; how strong is the evidence; what new failure modes.
2. Knowledge frozen at training time, not updatable, not citable, not private. The system changes: a retriever and an index supply passages in the context. The model at most needs to be good at using context.
3. Because nothing signals it. The model answers fluently and confidently with information that used to be true; neither the user nor the system can tell a stale answer from a current one without an external check.
4. Answer quality is bounded by retrieval quality, and a wrong retrieval is worse than none: the model does not fall back on "I don't know", it repeats what it was given.
5. Relevance (is this passage about the question?) can be judged from the passage and the question. Truth cannot: a false statement about the right key is indistinguishable in form from a true one, and the model has no other evidence. Verification needs an independent source.
6. Retrieval alone (is the right passage among the top k?); generation given correct passages; behaviour given missing, irrelevant and misleading passages (abstention, repetition); plus latency and cost of the longer prompts.
7. The MLP of a block is replaced by several expert MLPs and a router that sends each token to a few of them. Intended advantage: more parameters (capacity) at the same arithmetic per token.
8. On wall-clock time: dispatch and sorting overhead lowered hardware utilisation so much that training to a given loss took longer, and the code and kernel constraints (no FP8) were costly. Lesson: an improvement must be judged in the unit you pay in; per-step and per-FLOP gains can vanish per second.
9. Exact, direct access to any earlier token: the state is a fixed-size summary. Copying a specific string from far back (an expression into a tool call, a value out of a retrieved passage or tool result) is precisely what attention does well and what a compressed state can lose.
10. It is cut into patches, each mapped to a vector (linear projection or a vision encoder) and placed in the sequence like token embeddings. Unchanged: the Transformer blocks, the next-token objective for text, the KV cache and cost model. Security: instructions can be embedded in an image, so images are untrusted input like any retrieved text.
11. Training a smaller student to reproduce a larger teacher's behaviour. A teacher's output distribution says how likely *every* continuation is at each position, whereas data give one sampled continuation: many more bits of supervision per example, including which mistakes are near-misses.
12. It establishes that one implementation, at one scale, on one hardware and with a given amount of tuning did not beat a stated baseline. It does not establish that the idea cannot work. Keeping it saves repeating the experiment, records where the cost was (so one knows when to revisit), and calibrates trust in positive claims that lack the same comparison.
