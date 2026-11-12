# Module 19 — Lab: a team, its message log, and its fair baseline

Script: [`labs/agents/19_multi_agent.py`](../../labs/agents/19_multi_agent.py) ·
Code under study: [`mlle/agents/multi.py`](../../mlle/agents/multi.py), [`mlle/agents/reasoning.py`](../../mlle/agents/reasoning.py), [`mlle/agents/llm.py`](../../mlle/agents/llm.py) ·
Solutions: [`solutions/19-multi-agent.md`](../../solutions/19-multi-agent.md)

The lab runs in a few seconds. All roles are played by the simulated solver of Module 17;
its error rates are assumptions you can and should vary.

## G. Exercises

### Exercise 1 (by hand): cost and approval arithmetic

1. A problem has 6 operations. One round of planner + executor + model reviewer costs how many calls? With up to 3 rounds, what are the minimum and maximum cost?
2. An executor answers correctly with probability 0.7. A reviewer approves a correct answer with probability 0.9 and a wrong one with probability 0.1. What fraction of *approved* answers is correct? What fraction of all first-round answers is approved?
3. Same executor, but 20 percentage points of its 30 % errors are systematic and the reviewer (same model) approves those with probability 0.9, the other wrong answers with probability 0.1. Recompute the fraction of approved answers that is correct.
4. Three agents work in a pipeline; each passes on a correct result with probability 0.9 if its input was correct, and never repairs a wrong input. What is the probability of a correct final result? What changes if a code check after stage 1 catches 95 % of stage-1 errors and triggers one retry?

**Success criterion:** numerical answers.

### Exercise 2 (experimental): the fair comparison

```bash
python labs/agents/19_multi_agent.py
```

1. Answer the lesson's "predict before you run" question. Record table 2.
2. Read the single run of part 1 message by message. Where did the error enter, which steps were *correct given their input*, and which role had the chance to catch it?
3. The team beats cost-matched voting. Write a single-agent function (no roles, no messages) that performs the same sequence of calls as `run_team`, using `decompose` and `llm_checker` from `mlle.agents.reasoning`. Show that its accuracy and cost match the team's. What do you conclude about the word "team"?
4. In `NoisySolver`, set `p_step = p_direct = 0.15` (decomposition no longer easier per step). Re-run `compare_systems`. Does the team still beat voting? What assumption was carrying the result?

**Success criterion:** recorded table; the function of item 3 with matching numbers; a written answer to item 4.

### Exercise 3 (experimental): reviewers and independence

Use parts 3 and 4 of the output and your own variations.

1. From part 3, compute for each reviewer the extra calls spent per percentage point of accuracy gained over "no reviewer".
2. In part 4, accuracy is almost the same for all three reviewers while "wrong answers approved" ranges from 0.37 to 0.00. Explain both facts. Which number would matter more if approved answers were acted on automatically?
3. Vary the systematic share (0.0, 0.5, 1.0) for the same-model reviewer and tabulate "wrong answers approved". At which share does self-review become nearly worthless?
4. Give the "independent model" reviewer a 20 % error rate on its checks (`p_check_err=0.2`). Is a mediocre independent reviewer better or worse than a careful same-model reviewer, in terms of wrong approvals? What does this say about how to spend effort?

**Success criterion:** two tables (items 1 and 3) and written answers to items 2 and 4.

### Exercise 4 (coding): harden the protocol

1. Add to `run_team` a blackboard entry with the full execution trace and make the model reviewer check *each step* (`CHECK: <step>=<result>`) instead of only the final answer. Compare accuracy, cost and the location at which errors are caught.
2. Add an outcome label to `TeamRun`: `approved`, `rejected_out_of_rounds`. Report, for part 2's team, how often the returned answer was never approved and how accurate those answers are.
3. Implement per-role permissions: the executor may call a `calculator` tool, the planner may not call any tool, the reviewer may only read. Show a test in which a planner message containing a tool request is refused by the orchestrator.
4. Detect ping-pong: if the reviewer rejects the same answer twice, stop with outcome `stalemate` instead of spending the remaining rounds.

**Success criterion:** working code with one test per item.

### Exercise 5 (design): a team for a real task

Choose one: (a) answering questions over a set of documents with citations; (b) fixing a
failing unit test in a small code base; (c) producing a weekly summary from a mailbox.

1. Draw the roles, messages and shared state (a Mermaid diagram like the lesson's).
2. For each role: model or code? Which tools and permissions? Where does *independent* information enter?
3. Name the success test and who implements it.
4. Mark every boundary where untrusted text crosses from one role to another, and the role holding the most dangerous permission. Can injected text reach it?
5. State the single-agent baseline you would have to beat, and the budget at which you would compare.

**Success criterion:** a one-page design that answers all five points.

### Challenge (optional): debate

Implement a two-solver debate: both answer; if they disagree, each sees the other's
answer and may revise once; a judge (model check or tool) decides. Compare, at equal
cost, with voting, for random and for systematic errors. Under which error model does
debate between copies of one model add anything?

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Separate the effect of organisation (roles, review, validation) from the effect of budget, and quantify the value of an independent check |
| **Hypotheses** | (H1) The team beats a single call by a large margin and cost-matched voting by a smaller one. (H2) A single agent running the same workflow matches the team exactly. (H3) An exact tool reviewer is more accurate and cheaper than a model reviewer. (H4) With systematic errors, a same-model reviewer approves at least five times as many wrong answers as an independent one. (H5) Validating the plan message in code raises accuracy and lowers cost. |
| **Hardware / software** | CPU; course package; no model |
| **Tier** | A: about 5 seconds |
| **Metrics** | accuracy; model calls per problem; fraction of problems whose wrong answer was approved |
| **Independent variables** | system (single call, team, cost-matched vote); reviewer type; systematic share of errors; plan validation |
| **Controls** | the same 400 problems per comparison; seeded solver; cost matching by rounding the team's mean calls |
| **Interpretation** | All numbers are consequences of the solver's assumed error model. They establish relationships ("if errors are shared, self-review approves them"), not magnitudes for real models. |
| **Common failure modes** | Comparing a team with a single call. Counting only accuracy and not wrong approvals. Calling a same-model review "verification". Omitting the message log. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.3–D.6.

| System | 4 ops: accuracy | calls | 8 ops: accuracy | calls | predicted winner? |
|---|---:|---:|---:|---:|---|
| single call | | | | | |
| team, model reviewer | | | | | |
| team, tool reviewer | | | | | |
| vote, cost-matched | | | | | |
| single agent, same workflow (Exercise 2.3) | | | | | |

| Reviewer (systematic share 0.5) | accuracy | calls | wrong answers approved |
|---|---:|---:|---:|
| same model | | | |
| independent model | | | |
| exact tool | | | |

**Questions to answer in writing**

1. Complete the sentence with evidence from your tables: "Adding a reviewer agent helps when …".
2. A framework advertises "a team of five specialised agents". Which three questions do you ask before adopting it?
3. Where in a multi-agent system would you place the one human approval step you can afford, and why there?
