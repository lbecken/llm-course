"""
A tool-using assistant: the smallest complete tool loop (Module 16).

    user question
        -> model output
             a) plain text            -> that is the answer; stop
             b) a tool call           -> validate, execute, append "OBSERVATION: ...", ask the model again
             c) a malformed tool call -> append the error as an observation, ask again
    ... until an answer, or the budget of tool calls is used up.

The model decides WHAT to call. This loop decides WHETHER it runs, runs it, and bounds
the whole interaction. The trace records every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlle.agents.llm import LLM
from mlle.agents.tools import ToolCall, ToolRegistry, parse_json_tool_call

SYSTEM_TEMPLATE = """You can use tools. To call one, reply with ONLY a JSON object:
{{"tool": "<name>", "args": {{...}}}}
Available tools:
{tools}
When you know the answer, reply with the answer as plain text (no JSON)."""


@dataclass
class TraceStep:
    kind: str  # "model" | "tool" | "error" | "answer" | "stopped"
    content: str


@dataclass
class AssistantResult:
    answer: str | None
    trace: list[TraceStep] = field(default_factory=list)
    tool_calls: int = 0
    stopped_reason: str = "answered"  # answered | tool_budget | error_budget


class ToolAssistant:
    def __init__(self, llm: LLM, registry: ToolRegistry, max_tool_calls: int = 4, max_errors: int = 2):
        self.llm, self.registry, self.max_tool_calls, self.max_errors = llm, registry, max_tool_calls, max_errors

    def run(self, question: str) -> AssistantResult:
        transcript = SYSTEM_TEMPLATE.format(tools=self.registry.describe()) + f"\n\nUSER: {question}\n"
        result = AssistantResult(answer=None)
        errors = 0
        while True:
            output = self.llm.complete(transcript + "ASSISTANT:", max_tokens=128).strip()
            result.trace.append(TraceStep("model", output))
            parsed = parse_json_tool_call(output)
            if parsed is None:  # plain text: the final answer
                result.answer = output
                result.trace.append(TraceStep("answer", output))
                return result
            if isinstance(parsed, str):  # malformed call: tell the model, do not guess
                errors += 1
                observation = f"ERROR: {parsed}"
                result.trace.append(TraceStep("error", observation))
            else:
                if result.tool_calls >= self.max_tool_calls:
                    result.stopped_reason = "tool_budget"
                    result.trace.append(TraceStep("stopped", "tool-call budget exhausted"))
                    return result
                result.tool_calls += 1
                tool_result = self.registry.execute(parsed)
                observation = tool_result.output
                if not tool_result.ok:
                    errors += 1
                result.trace.append(TraceStep("tool" if tool_result.ok else "error", f"{parsed.name}({parsed.args}) -> {observation}"))
            if errors > self.max_errors:
                result.stopped_reason = "error_budget"
                result.trace.append(TraceStep("stopped", "too many errors"))
                return result
            # The observation is DATA for the model to read. It is clearly delimited and never
            # treated by this loop as an instruction.
            transcript += f"ASSISTANT: {output}\nOBSERVATION: {observation}\n"


def format_trace(result: AssistantResult) -> str:
    lines = [f"{i:2d}. [{s.kind:7s}] {s.content}" for i, s in enumerate(result.trace, 1)]
    lines.append(f"    => answer: {result.answer!r} | tool calls: {result.tool_calls} | stopped: {result.stopped_reason}")
    return "\n".join(lines)
