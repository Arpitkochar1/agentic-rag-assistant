"""Calculator tool. The LLM only *translates* the question into an expression;
evaluation is done by a strict AST whitelist (never eval())."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from rag_agent.domain.interfaces import LLMClient
from rag_agent.domain.models import ChatMessage, ToolResult

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_POW = 1000


class CalculatorError(ValueError):
    pass


def safe_eval(expression: str) -> float | int:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {expression!r}") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
                raise CalculatorError("Exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")

    try:
        return ev(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero") from exc
    except (OverflowError, ValueError, TypeError) as exc:
        raise CalculatorError(str(exc)) from exc


_EXTRACT_PROMPT = (
    "Convert the user's maths question into ONE arithmetic expression. "
    "Allowed: numbers, + - * / // % ** ( ), functions sqrt log log10 exp sin cos tan abs round, "
    "constants pi and e. Express percentages as fractions (15% of 80 -> 0.15*80). "
    "Output ONLY the expression, nothing else."
)


class CalculatorTool:
    name = "calculator"
    description = "Evaluate arithmetic and numeric questions exactly."

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, question: str) -> ToolResult:
        try:
            raw = self._llm.invoke(
                [ChatMessage(role="system", content=_EXTRACT_PROMPT), ChatMessage(role="user", content=question)]
            )
            expression = raw.strip().strip("`").strip().splitlines()[0]
            value = safe_eval(expression)
            if isinstance(value, float):
                value = round(value, 10)
            return ToolResult(text=f"{expression} = {value}")
        except Exception as exc:
            return ToolResult(ok=False, text=f"Calculator error: {exc}")
