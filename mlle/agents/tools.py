"""
Tools for language models: definitions, parsing, validation, restricted execution (Module 16).

A "tool call" is text produced by the model in an agreed format. The model never
executes anything. The APPLICATION parses the text, decides whether the call is allowed,
runs it, and puts the result back into the model's context. Every step here is ordinary,
testable code, and it is where safety lives:

    parse      malformed output is an error to report back, not something to guess at
    validate   unknown tool, wrong argument names or types -> rejected before execution
    permit     each tool has a permission level; the registry enforces an allow-list
    execute    restricted implementations (the calculator walks an AST; it never calls eval)
    observe    results and errors are returned as plain text observations
"""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ----------------------------------------------------------------------------
# A calculator that cannot execute code

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str, max_length: int = 100, max_abs: float = 1e15):
    """Evaluate + - * / // % with parentheses on numbers. Raises ValueError on anything else.

    The expression is parsed into a syntax tree and only number literals and the listed
    operators are accepted: names, calls, attributes, strings, comprehensions and the power
    operator are all rejected, so no Python code can run and no huge numbers can be built.
    (nanochat's calculator uses character allow-lists plus eval with empty builtins; walking
    the tree is the stricter approach.)"""
    if len(expression) > max_length:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not a valid expression: {e.msg}") from None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            value = _BINARY[type(node.op)](left, right)
            if abs(value) > max_abs:
                raise ValueError("result out of range")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed in a calculation: {type(node).__name__}")

    return walk(tree)


# ----------------------------------------------------------------------------
# Tool definitions and the registry

READ, WRITE, DANGEROUS = "read", "write", "dangerous"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, type]  # argument name -> required Python type (the "schema")
    fn: Callable[..., Any]
    permission: str = READ

    def schema_text(self) -> str:
        args = ", ".join(f"{k}: {t.__name__}" for k, t in self.parameters.items())
        return f"{self.name}({args}) - {self.description} [{self.permission}]"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    output: str  # the observation given back to the model: a result or an error message


@dataclass
class ToolRegistry:
    """Holds the tools an application offers and enforces what may run.

    `allowed` is the set of permission levels the CURRENT task may use. A tool whose
    permission is outside it is refused whatever the model says, including when a tool
    result or a document "instructs" the model to call it (prompt injection)."""

    tools: dict[str, Tool] = field(default_factory=dict)
    allowed: set[str] = field(default_factory=lambda: {READ})
    log: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def describe(self) -> str:
        return "\n".join(t.schema_text() for t in self.tools.values())

    def validate(self, call: ToolCall) -> str | None:
        """Returns an error message, or None if the call is well-formed and permitted."""
        tool = self.tools.get(call.name)
        if tool is None:
            return f"unknown tool '{call.name}'. Available: {', '.join(self.tools)}"
        if tool.permission not in self.allowed:
            return f"tool '{call.name}' requires permission '{tool.permission}', which is not granted for this task"
        missing = [k for k in tool.parameters if k not in call.args]
        extra = [k for k in call.args if k not in tool.parameters]
        if missing or extra:
            return f"wrong arguments for {call.name}: missing {missing}, unexpected {extra}"
        for k, t in tool.parameters.items():
            if not isinstance(call.args[k], t) or isinstance(call.args[k], bool) and t is not bool:
                return f"argument '{k}' of {call.name} must be {t.__name__}"
        return None

    def execute(self, call: ToolCall) -> ToolResult:
        error = self.validate(call)
        if error is not None:
            result = ToolResult(False, f"ERROR: {error}")
        else:
            try:
                result = ToolResult(True, str(self.tools[call.name].fn(**call.args)))
            except Exception as e:  # noqa: BLE001  a failing tool is an observation, not a crash
                result = ToolResult(False, f"ERROR: {type(e).__name__}: {e}")
        self.log.append((call, result))
        return result


# ----------------------------------------------------------------------------
# Parsing model output into tool calls

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_tool_call(text: str) -> ToolCall | None | str:
    """Parse a call of the form {"tool": "name", "args": {...}} anywhere in `text`.

    Returns a ToolCall, None if the text contains no JSON object (a normal answer), or an
    error STRING if something looks like a call but is malformed. The error is fed back to
    the model so it can correct itself: never silently repair or guess."""
    match = _JSON_RE.search(text)
    if match is None:
        if "{" in text and '"tool"' in text:  # it tried to call a tool but the JSON is incomplete
            return "malformed tool call (incomplete JSON)"
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return f"malformed tool call (invalid JSON: {e.msg})"
    if not isinstance(obj, dict) or "tool" not in obj:
        return 'malformed tool call (expected {"tool": ..., "args": {...}})'
    args = obj.get("args", {})
    if not isinstance(obj["tool"], str) or not isinstance(args, dict):
        return "malformed tool call ('tool' must be a string and 'args' an object)"
    return ToolCall(obj["tool"], args)


def default_registry(notes: dict[str, str] | None = None) -> ToolRegistry:
    """A small, restricted tool set used by the labs: calculator, letter counting, a read-only
    notes store, and one deliberately dangerous tool that the default policy never allows."""
    notes = notes if notes is not None else {}
    deleted: list[str] = []
    reg = ToolRegistry()
    reg.register(Tool("calculator", "evaluate an arithmetic expression", {"expression": str}, lambda expression: safe_calculate(expression)))
    reg.register(Tool("count_letter", "count occurrences of a letter in a word", {"word": str, "letter": str},
                      lambda word, letter: word.lower().count(letter.lower())))
    reg.register(Tool("read_note", "read a stored note by its key", {"key": str}, lambda key: notes[key]))
    reg.register(Tool("delete_all_notes", "delete every note", {}, lambda: (deleted.extend(notes), notes.clear(), "deleted")[2], permission=DANGEROUS))
    reg.deleted = deleted  # type: ignore[attr-defined]  exposed for tests
    return reg
