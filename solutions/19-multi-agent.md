# Solutions — Module 19

## Lab exercises

### Exercise 1: cost and approval arithmetic

1. Per round: 1 plan + 6 steps + 1 review = **8** calls. Minimum 8 (approved in round 1), maximum **24**.
2. Approved: $0.7\cdot0.9 + 0.3\cdot0.1 = 0.66$ of first-round answers. Correct among approved: $0.63/0.66 = $ **0.955**.
3. Wrong and approved: $0.2\cdot0.9 + 0.1\cdot0.1 = 0.19$. Approved in total: $0.63 + 0.19 = 0.82$. Correct among approved: $0.63/0.82 = $ **0.768**. The reviewer's stamp went from "95 % reliable" to "77 % reliable" without any change in its nominal error rate, only because its errors are correlated with the solver's.
4. $0.9^3 = $ **0.729**. With the check: stage 1 correct with probability $0.9 + 0.1\cdot0.95\cdot0.9 = 0.9855$; final $0.9855\cdot0.81 = $ **0.798**. One cheap check at the first boundary recovers a quarter of the lost reliability; the remaining loss is in the unchecked stages.

### Exercise 2: the fair comparison

Reference table: lesson D.3.

2. The error entered at the executor's second step (`-4-27 = -30` instead of −31). Steps three and four were correct *given their input* (−30·2 = −60; −60−26 = −86). The reviewer, which saw the original problem and the final answer, was the only role positioned to catch it, and approved (one of its 10 % wrong reviews). Nothing downstream of an error re-derives it; only a check against the original task can.
3. Sketch:

   ```python
   def workflow(llm, problem, max_rounds=3):          # one agent, no roles, no messages
       calls = 0
       for rnd in range(1, max_rounds + 1):
           t = 1.0 if rnd > 1 else 0.0
           plan = llm.complete(f"PLAN: {problem}", temperature=t, seed=rnd); calls += 1
           value = None
           for step in plan.split(";"):
               value = llm.complete(f"STEP: {step if value is None else value + step}", temperature=t, seed=rnd); calls += 1
           calls += 1
           if llm.complete(f"CHECK: {problem}={value}") == "yes":
               break
       return value, calls
   ```

   Measured: 0.940 at 8.5 calls (4 operations) and 0.915 at 15.3 calls (8 operations): identical to the team, to every digit. The "team" is this function with labels on its calls. What produced the result is the workflow: decompose, verify, retry.
4. Measured with `p_step = p_direct = 0.15`: 4 operations: team 0.787 at 10.8 calls, cost-matched voting (k = 11) **0.963**; 8 operations: team 0.550 at 21.9 calls, voting (k = 22) **0.720**. The team now *loses* to voting at equal cost. The earlier result rested entirely on the assumption that a single step is three times more reliable than a step inside a long direct answer. Whether that holds for a real model on a real task is an empirical question (Module 17's part A suggests it often does for small models), and it is the question to settle before building a decomposition pipeline.

### Exercise 3: reviewers and independence

1. Relative to no reviewer (0.735 at 7.0 calls): same model, 10 %: +20.3 points for +4.7 calls (0.23 calls per point); same model, 30 %: +9.0 points for +6.6 calls (0.73 per point); exact tool: +24.8 points for +2.4 calls (0.10 per point). The careless reviewer costs three times as much per point as the careful one, and seven times as much as the tool.
2. Accuracy: the solver's systematic errors are reproduced on every retry, so rejecting them does not produce a right answer; and its random errors are caught about equally by any reviewer. Wrong approvals: the same-model reviewer shares the systematic errors and approves them. If approved answers are acted on automatically, wrong approvals are the damage; unapproved answers can be held back or escalated. The second number is the one that matters operationally.
3. Measured (same-model reviewer; accuracy, calls, wrong answers approved): share 0: 0.682, 4.0, 0.102; share 0.5: 0.545, 2.9, 0.370; share 1: 0.380, 2.3, **0.580**. With fully systematic errors, 58 of the 62 % wrong answers are approved: self-review is worthless as a filter, and the falling number of calls shows it stops *looking* (it approves at once).
4. Measured at share 0.5: independent reviewer with 20 % check errors: wrong approvals 0.152; same-model reviewer with only 2 % check errors: 0.352. A mediocre independent reviewer lets through less than half as many wrong answers as an excellent one that shares the blind spot. Effort spent on *independence* (a different model, a tool, different evidence) buys more than effort spent on making a correlated reviewer more careful.

### Exercise 4: harden the protocol

1. Per-step review costs one extra call per step instead of one per round, and localises errors: a rejected step can be redone alone instead of repeating the round. With independent step errors this raises accuracy at somewhat higher cost; it also shortens the propagation path to zero. With a same-model reviewer and systematic step errors it inherits D.5's weakness at every step.
2. Sketch: set `run.outcome = "approved"` on break, else `"rejected_out_of_rounds"` after the loop. Expect unapproved answers to be much less accurate than approved ones (they are the ones the reviewer rejected three times); that is exactly why the label is valuable to whatever consumes the answer.
3. Sketch: a `permissions = {"planner": set(), "executor": {"calculator"}, "reviewer": set()}` table in the orchestrator; any tool request parsed from a role's output is executed only if the tool is in that role's set, otherwise the refusal becomes that role's observation. Test: a scripted planner that emits a calculator request receives "not permitted for role planner" and no tool call is logged.
4. Keep the previous rejected answer; if the new answer equals it and is rejected again, stop with `stalemate`. Under systematic errors this saves the remaining rounds and reports honestly that no approved answer exists.

### Exercise 5: a team for a real task

No single right answer. A strong design for (a), questions over documents: retrieval is
**code** (search), not an agent; one model call drafts an answer with citations; a
**code** check verifies that every cited passage exists and contains the quoted text
(independent, exact); an optional second *different* model judges whether the passages
support the claims. Untrusted text enters with the documents, so the drafting role gets
no tools beyond reading. Success test: citation check passes and, on an evaluation set,
answers match references. Baseline to beat: one call with the retrieved passages in the
prompt, at the same token budget. Many "multi-agent" designs for this task reduce, after
this exercise, to a workflow with one model call and two checks in code.

### Challenge: debate

Outline and expectation. With *random* errors, two independent samples disagree when at
least one is wrong, and revision after seeing the other's answer acts like a small vote:
some gain, comparable to voting at equal cost. With *systematic* errors, both copies give
the same wrong answer, never disagree, and the debate never starts: zero gain, and full
confidence. Debate between copies of one model adds something only to the extent that
their errors are independent, which sampling temperature provides for random slips and
nothing provides for shared misconceptions. Debate between *different* models, or with
different evidence, is a different experiment.

### Written questions

1. "…the reviewer has information or error patterns the author lacks: an exact tool (0.983 against 0.735 with no reviewer), a different model (wrong approvals 6 % against 37 %), or a check in code; and when its own error rate is low (10 % wrong reviews gave 0.938, 30 % gave 0.825)."
2. For example: What does a single agent achieve with the same token budget, and with the same workflow? Are the five agents different models or one model with five prompts, and where does independent information enter? What is the measured rate of wrong approvals, and what is logged?
3. At the action with the largest irreversible consequence, as late as possible but before it executes, with the concrete action and its provenance shown (Module 16). Not at the plan (too early: the damage is done later, by something the human never saw) and not at a model-written summary.

## Quiz

1. Role: a prompt (possibly with its own model, tools, permissions) — involves a model. Message: a typed record passed from one role to another — data. Shared state: data all roles can read and write, such as a blackboard — data. Orchestrator: the code deciding who is called when and when to stop — code.
2. The orchestrator calls the model with the planning prompt, takes the returned text, and inserts it (or parts of it) into the prompt of a second call that asks for one step to be carried out.
3. A single agent given the same total budget (calls or tokens), e.g. voting or retries; and a single agent executing the same workflow (same sequence of prompts) without the multi-agent framing.
4. The workflow: decomposition into single steps, which the assumed solver performs more reliably than whole chains, plus verification and retry. Proof: a single function making the identical sequence of calls reaches identical accuracy and cost; and when single steps are made no more reliable than direct answering, the team loses to voting.
5. Transfers: focus from a short task-specific context (partly), parallelism for wall-clock time, separation of privileges. Does not transfer to copies of one model: different knowledge, and independent judgement (they share blind spots).
6. Approved: $0.6\cdot0.95 + 0.4\cdot0.25 = 0.67$. Wrong among approved: $0.10/0.67 = $ **15 %**.
7. Accuracy is the same because re-asking the solver reproduces its systematic errors whoever rejects them; no reviewer can make the solver right. The difference matters because approved answers are treated as verified: with the same-model reviewer over a third of all problems carry a false stamp of approval, while the independent reviewer leaves most wrong answers visibly unapproved, so they can be escalated.
8. Any four: error propagation between roles; false approval by a reviewer; cost multiplication by roles and rounds; ping-pong between agents; diffusion of responsibility for the success test; prompt injection from one agent to another; loss of context at hand-offs.
9. A structural check is exact, cheap and independent of the model, and it stops an error before other roles build on it. Lab: validating the plan raised accuracy from 0.943 to 0.990 and *lowered* cost from 10.7 to 9.5 calls.
10. A web page contains instructions; A's summary carries them (or is itself the instruction) to B, which sends mail on the attacker's behalf, possibly including private data. Structural defence: B never receives free text from A as instructions; A returns data in a constrained schema; B's send permission requires human confirmation of the concrete message; or the agent that reads untrusted content has no path to the agent that can send.
11. Independence (a different model, tool or data for checking); parallel execution of independent sub-tasks; a task too large for one context; separation of privileges between reading untrusted input and acting.
12. Very little. A critic sharing the author's weights shares its blind spots and tends to favour its own outputs; a 98 % approval rate measures agreement of the model with itself. Quality needs an independent measurement: exact checks, a different judge, or human evaluation on a sample.
