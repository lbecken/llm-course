# Module 16 — Lab: a calculator in the token stream, and an application that says no

Script: [`labs/agents/16_tool_assistant.py`](../../labs/agents/16_tool_assistant.py) ·
Code under study: [`mlle/agents/tool_engine.py`](../../mlle/agents/tool_engine.py), [`mlle/agents/tools.py`](../../mlle/agents/tools.py), [`mlle/agents/assistant.py`](../../mlle/agents/assistant.py) ·
Solutions: [`solutions/16-tool-use.md`](../../solutions/16-tool-use.md)

Part A fine-tunes three small assistants from the cached Part IV base model (about
3 minutes on a 4-core CPU). Part B needs no model and runs in seconds:
`python labs/agents/16_tool_assistant.py --part b`.

## G. Exercises

### Exercise 1 (by hand): the state machine

The model samples, in this order: `<|python_start|>`, `7`, `*`, `8`, `<|python_end|>`,
and afterwards always wants to say `5`, `6`, `<|assistant_end|>`.

1. Write the complete token sequence that ends up in the context, marking each token S (sampled) or F (forced). How many forward passes of the model are needed, and in how many of them is the model's own prediction discarded?
2. Which of these tokens would be training targets if this sequence were fine-tuning data? Which carry a policy gradient in the RL of Module 15?
3. The same model now samples `<|python_start|>`, `7`, `*`, `*`, `8`, `<|python_end|>`. What does the course engine do? What does it *not* do, and why is that the right choice?
4. For an assistant that can read a user's emails and send emails, classify every text source by trust (lesson D.6) and name the one tool that must be gated by human confirmation.

**Success criterion:** item 1 matches `generate_with_tools` on a scripted model (see `test_engine_forces_the_calculator_result_and_masks_it`).

### Exercise 2 (experimental): what the tool adds

```bash
python labs/agents/16_tool_assistant.py --part a
```

1. Answer the lesson's "predict before you run" question, then record both tables.
2. The direct model reaches 65 % on multiplications but 2.5 % on 2-digit × 2-digit ones. What does that say about reporting one accuracy for "multiplication"? Which products is it getting right?
3. In the row "tool, unmasked / tool on", accuracy on additions is 0.97, not 1.00. Inspect the failures (write ten lines using `generate_with_tools`): what did the model do in those cases?
4. The tool-trained model fails on longer operands because its *request* is wrong. Propose two remedies on the data side and one on the application side. Which could you test in this lab?
5. Sample with `temperature=1.0` instead of greedy decoding for the tool-trained model (pass `temperature` and a seeded generator). Does the tool-call rate change? Does accuracy? Why is a tool-using model less sensitive to sampling temperature in the *answer* than a direct one?

**Success criterion:** recorded tables and written answers to items 2–4.

### Exercise 3 (coding): extend the application

```bash
python labs/agents/16_tool_assistant.py --part b
```

1. Record the four traces. In B2, which component produced each of the two error messages?
2. Add a tool `word_length(word: str) -> int` to a registry and write a scripted conversation that calls it with (a) a correct call, (b) an integer instead of a string, (c) an extra argument. Verify that (b) and (c) are refused with informative observations.
3. Implement human confirmation: give `ToolRegistry` an optional `confirm: Callable[[ToolCall], bool]`; a tool with permission `write` or `dangerous` runs only if the permission is allowed **and** `confirm(call)` returns true. Show the injection of B4 being stopped by a confirmation callback even when the dangerous permission is granted. What must the callback show the human for the confirmation to be meaningful?
4. In B4 the injected text asked for a *dangerous* tool. Rewrite the injection so that it only uses *read* tools but still harms the user (hint: what could a read tool reveal, and where would the model put it?). What does this say about permissions as a defence?

**Success criterion:** working code for items 2 and 3 with a test each; a concrete attack for item 4.

### Exercise 4 (analysis): threat model

An assistant has these tools: `search_web(query)`, `read_file(path)` in a project
directory, `write_file(path, text)` in the same directory, `run_tests()`.

1. List every channel through which text not written by the user can reach the model.
2. For each tool give its permission class and the worst outcome if an injected instruction triggers it.
3. Specify a default policy (permissions per task type, budgets, confirmations) and name the residual risk you accept.
4. `run_tests()` executes code that `write_file` may have just written. What does that imply for the classification of `write_file`?

**Success criterion:** a one-page threat model with a table for item 2.

### Exercise 5 (nanochat): read the calculator and the sandbox

```bash
cd external/nanochat
python -c "
from nanochat.engine import use_calculator
for e in ['12*12', '1,000+1', '2**100', '\"strawberry\".count(\"r\")', '__import__(\"os\").getcwd()', '1/0']:
    print(repr(e), '->', repr(use_calculator(e)))"
python -m pytest tests/test_execution.py tests/test_engine.py -q
```

1. Record the outputs. For each `None`, name the line of `use_calculator` that produced it.
2. `use_calculator` evaluates `'9'*20 + '*' + '9'*20` (two 20-digit numbers) without complaint. Is that a problem? What bound would you add, and what does the course's `safe_calculate` do with the same input?
3. Which protections of `execute_code` are listed as *not covered*? For each, name the mechanism that would cover it.
4. In `Engine.generate`, what happens if the model emits `<|python_end|>` without a preceding `<|python_start|>`? If it emits `<|output_start|>` itself? Compare with the course engine.
5. *(With your SFT model on a GPU.)* Ask it `How many r are in strawberry?` and a two-step arithmetic word problem through `scripts.chat_cli`. Does it call the calculator? Is the request right?

**Success criterion:** written answers to 1–4. Item 5: *awaiting validation on GPU, no reference values.*

### Challenge (optional): a second tool in the token stream

Extend `generate_with_tools` and the fine-tuning data with a second tool that shares the
`<|python_start|>` syntax: `len(word)` for Pocket words (the answer to "how long is
robots"). Train one model on a mixture of calculator and length questions. Measure
whether it chooses the right tool, and what it does for a question that needs neither.

## H. Experimental procedure

| | |
|---|---|
| **Objective** | Quantify what a tool adds to a small trained model; demonstrate the failure modes of absence, fabrication, blind trust and wrong requests; exercise application-side validation and permissions |
| **Hypotheses** | (H1) Tool-trained accuracy on held-out 2-digit × 2-digit products exceeds 0.95 while the direct model stays below 0.10. (H2) With the tool off, the tool-trained model's accuracy is near 0. (H3) A model trained with tool outputs as targets fabricates output blocks when the tool is off. (H4) With a lying tool, the final answer equals the tool's output in more than 95 % of cases. (H5) On operands longer than in training, the request is miscopied and accuracy collapses although the tool is exact. (H6) An injected instruction requesting a dangerous tool is refused under a read-only policy and executed when the permission is granted. |
| **Hardware / software** | CPU; course package. Exercise 5: nanochat environment |
| **Tier** | A: part A ≈ 3 min, part B seconds |
| **Metrics** | exact-match accuracy by question type; tool-call rate; fabricated-output rate (output block opened by a *sampled* token); share of answers equal to the tool output; share of requests equal to the question's expression |
| **Independent variables** | training format (direct, tool, tool unmasked); tool condition (on, off, lying); operand length; permission policy |
| **Controls** | the same base model, questions, steps and learning rate for all three assistants; greedy decoding; held-out questions |
| **Interpretation** | 200 questions per cell (standard error up to 0.035). One seed. The gullible model of part B is scripted: it demonstrates the mechanism, not the rate at which real models are fooled. |
| **Common failure modes** | Evaluating a tool-using model without its tool, or with a different tool implementation from training. Reporting one accuracy over a mixture of easy and hard cases. Treating a model-written output block as a tool result. |

## I. Results and interpretation

Reference values (CPU, 2026-11): lesson D.4.

| Model | Condition | additions | multiplications | 2-digit × 2-digit | fabricates | answer = tool |
|---|---|---:|---:|---:|---:|---:|
| direct | — | | | | | |
| tool | on | | | | | |
| tool | off | | | | | |
| tool, unmasked | off | | | | | |
| tool | lying | | | | | |

| Longer operands | direct | tool | request exact | answer = tool |
|---|---:|---:|---:|---:|
| 4-digit + 4-digit | | | | |
| 3-digit × 3-digit | | | | |

**Questions to answer in writing**

1. "Our model can now do arithmetic." Rewrite this sentence so that it is accurate for the tool-trained system.
2. Why is the loss mask on tool outputs a *safety* property and not only a training detail?
3. Explain to a colleague why "we told the model to ignore instructions in documents" is not a sufficient defence against prompt injection, using the lab's B4 traces.
